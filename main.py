"""
Computron — a talk-to-it-and-it-talks-back assistant, backed by Claude Code.

Flow per turn:
  1. Press Enter to start recording, press Enter again to stop.
  2. faster-whisper transcribes your speech locally (free, no API call).
  3. The transcript goes to a persistent `claude` process (stream-json in/out)
     that stays alive for the whole run — see claude_code_backend.py for why.
  4. Claude's reply is spoken back via Piper (local, free) + afplay.

Run: python main.py
"""
import subprocess
import sys
import tempfile
import threading
import warnings
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import sounddevice as sd

# faster-whisper's mel-filter step can hit divide-by-zero/overflow on
# very short or silent recordings; harmless, but noisy in the terminal.
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="faster_whisper.feature_extractor"
)

from faster_whisper import WhisperModel

import config
from claude_code_backend import ClaudeCodeSession, get_voice_speed

SAMPLE_RATE = 16000

# ANSI colors so "You" and "Computron" are visually distinct in the terminal.
COLOR_YOU = "\033[36m"       # cyan
COLOR_COMPUTRON = "\033[32m"  # green
COLOR_DIM = "\033[2m"         # dim, for the cost tag
COLOR_RESET = "\033[0m"

print("Loading local speech-to-text model (first run downloads it, ~1 min)...")
whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8")

# Tracks the currently-playing afplay process (if any) so stop_speaking()
# can interrupt it — e.g. pressing the push-to-talk hotkey again while
# Computron is still talking (barge-in), from menubar_app.py.
_current_playback: Optional[subprocess.Popen] = None
# Serializes actual playback — without this, two speak() calls fired from
# different threads at once (a voice reply and, e.g., auto-read narration)
# would start two overlapping afplay processes and garble the audio.
# Concurrent callers block here and play sequentially instead.
_playback_lock = threading.Lock()


def _play(path: Path) -> None:
    """Plays an audio file via afplay, tracked so it can be interrupted."""
    global _current_playback
    with _playback_lock:
        proc = subprocess.Popen(["afplay", str(path)])
        _current_playback = proc
        returncode = proc.wait()
        if _current_playback is proc:
            _current_playback = None
    # A positive returncode is a real afplay failure; a negative one means
    # it was killed by a signal (stop_speaking(), or an external kill) —
    # an intentional interruption, not an error, so don't raise for that.
    if returncode and returncode > 0:
        raise subprocess.CalledProcessError(returncode, ["afplay", str(path)])


def stop_speaking() -> bool:
    """Interrupts current playback, if any. Returns whether anything was
    actually playing (so a caller can decide whether it's worth logging)."""
    global _current_playback
    proc = _current_playback
    if proc is None or proc.poll() is not None:
        return False
    proc.terminate()
    _current_playback = None
    return True


def record_audio() -> Path:
    """Records from the mic until the user presses Enter again."""
    print("Recording... press Enter to stop.")
    frames = []

    def callback(indata, frames_count, time_info, status):
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback
    )
    with stream:
        input()  # blocks until Enter is pressed again

    audio = np.concatenate(frames, axis=0) if frames else np.zeros((0, 1), dtype="int16")
    out_path = Path(tempfile.gettempdir()) / "ev_input.wav"
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return out_path


def transcribe(wav_path: Path) -> str:
    segments, _ = whisper_model.transcribe(str(wav_path))
    return " ".join(seg.text.strip() for seg in segments).strip()


def _speak_piper(text: str) -> Optional[Path]:
    """Synthesizes text with Piper (local, free) and plays it with afplay."""
    out_wav = Path(tempfile.gettempdir()) / "ev_output.wav"
    piper_cmd = [
        config.PIPER_BIN,
        "--model", config.PIPER_VOICE,
        "--length-scale", str(config.PIPER_LENGTH_SCALE),
        "--output_file", str(out_wav),
    ]
    try:
        subprocess.run(piper_cmd, input=text.encode("utf-8"), check=True)
        _play(out_wav)
        return out_wav
    except FileNotFoundError:
        print("[Piper or afplay not found — printing reply instead]")
        print(f"{COLOR_COMPUTRON}Computron: {text}{COLOR_RESET}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"[Voice playback failed: {e} — printing reply instead]")
        print(f"{COLOR_COMPUTRON}Computron: {text}{COLOR_RESET}")
        return None


def _speak_elevenlabs(text: str) -> Optional[Path]:
    """Synthesizes text via the ElevenLabs API and plays the result.

    Falls back to Piper on any failure (network error, bad key, exhausted
    quota) — a paid-API hiccup shouldn't mean Computron goes silent, only
    less natural-sounding for that one reply.
    """
    out_mp3 = Path(tempfile.gettempdir()) / "ev_output.mp3"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{config.ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": config.ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": config.ELEVENLABS_MODEL,
        # Live from personality.json, not the static .env default — lets
        # Computron actually self-adjust its own speed on request, same
        # mechanism as the humor/sarcasm/bluntness dials.
        "voice_settings": {"speed": get_voice_speed()},
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        out_mp3.write_bytes(resp.content)
        _play(out_mp3)
        return out_mp3
    except (requests.RequestException, OSError) as e:
        print(f"[ElevenLabs failed ({e}) — falling back to Piper]")
        return _speak_piper(text)


def speak(text: str) -> Optional[Path]:
    """Synthesizes text and plays it aloud. Provider is chosen by
    config.TTS_PROVIDER ("elevenlabs" or "piper", default "piper") —
    ElevenLabs is only used if an API key is actually configured, otherwise
    this silently stays on Piper regardless of TTS_PROVIDER.

    Returns the path to the audio file actually played (wav or mp3, a
    fixed name overwritten each call) so a caller can replay it later via
    replay() without re-synthesizing — or None if synthesis/playback failed
    outright, meaning there's nothing valid to replay.
    """
    if config.TTS_PROVIDER == "elevenlabs" and config.ELEVENLABS_API_KEY:
        return _speak_elevenlabs(text)
    return _speak_piper(text)


def replay(wav_path: Optional[Path]) -> bool:
    """Re-plays an already-synthesized reply's wav file with afplay,
    skipping Piper entirely. Returns False if there's nothing to replay
    (no reply spoken yet this run, or the temp file is gone)."""
    if wav_path is None or not wav_path.exists():
        return False
    try:
        _play(wav_path)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def read_clipboard_aloud() -> Optional[Path]:
    """Reads the current macOS clipboard contents aloud via the same TTS
    pipeline as spoken replies — a mechanical action, bypasses Claude
    entirely (no reasoning needed to read text back verbatim). Returns the
    audio path (so it can be replayed via 'r'/"Replay last reply") or None
    if the clipboard was empty or unreadable."""
    try:
        result = subprocess.run(["pbpaste"], capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    text = result.stdout.strip()
    if not text:
        return None
    return speak(text)


def main():
    print("Starting Claude Code session...")
    session = ClaudeCodeSession(model=config.CLAUDE_MODEL)
    print("Computron is ready. Press Enter to talk, or type 'r' + Enter to replay the last reply. Ctrl+C to quit.\n")

    last_reply_wav = None
    try:
        while True:
            try:
                command = input("Press Enter to talk (or 'r' to replay)... ").strip().lower()
                if command == "r":
                    if not replay(last_reply_wav):
                        print("(Nothing to replay yet.)")
                    continue

                wav_path = record_audio()
                transcript = transcribe(wav_path)
                if not transcript:
                    print("(Didn't catch that — try again.)")
                    continue
                print(f"{COLOR_YOU}You: {transcript}{COLOR_RESET}")

                reply, turn_cost = session.ask(transcript)
                print(f"{COLOR_COMPUTRON}Computron: {reply}{COLOR_RESET}{COLOR_DIM}  [+${turn_cost:.4f}]{COLOR_RESET}")
                last_reply_wav = speak(reply)
            except KeyboardInterrupt:
                print("\nGoodbye.")
                break
    finally:
        session.close()


if __name__ == "__main__":
    main()
