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
import warnings
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

# faster-whisper's mel-filter step can hit divide-by-zero/overflow on
# very short or silent recordings; harmless, but noisy in the terminal.
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="faster_whisper.feature_extractor"
)

from faster_whisper import WhisperModel

import config
from claude_code_backend import ClaudeCodeSession

SAMPLE_RATE = 16000

# ANSI colors so "You" and "Computron" are visually distinct in the terminal.
COLOR_YOU = "\033[36m"       # cyan
COLOR_COMPUTRON = "\033[32m"  # green
COLOR_DIM = "\033[2m"         # dim, for the cost tag
COLOR_RESET = "\033[0m"

print("Loading local speech-to-text model (first run downloads it, ~1 min)...")
whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8")


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


def speak(text: str):
    """Synthesizes text with Piper and plays it with afplay (macOS)."""
    out_wav = Path(tempfile.gettempdir()) / "ev_output.wav"
    piper_cmd = [
        config.PIPER_BIN,
        "--model", config.PIPER_VOICE,
        "--length-scale", str(config.PIPER_LENGTH_SCALE),
        "--output_file", str(out_wav),
    ]
    try:
        subprocess.run(piper_cmd, input=text.encode("utf-8"), check=True)
        subprocess.run(["afplay", str(out_wav)], check=True)
    except FileNotFoundError:
        print("[Piper or afplay not found — printing reply instead]")
        print(f"{COLOR_COMPUTRON}Computron: {text}{COLOR_RESET}")
    except subprocess.CalledProcessError as e:
        print(f"[Voice playback failed: {e} — printing reply instead]")
        print(f"{COLOR_COMPUTRON}Computron: {text}{COLOR_RESET}")


def main():
    print("Starting Claude Code session...")
    session = ClaudeCodeSession(model=config.CLAUDE_MODEL)
    print("Computron is ready. Press Enter to start talking, Ctrl+C to quit.\n")

    try:
        while True:
            try:
                input("Press Enter to talk...")
                wav_path = record_audio()
                transcript = transcribe(wav_path)
                if not transcript:
                    print("(Didn't catch that — try again.)")
                    continue
                print(f"{COLOR_YOU}You: {transcript}{COLOR_RESET}")

                reply, turn_cost = session.ask(transcript)
                print(f"{COLOR_COMPUTRON}Computron: {reply}{COLOR_RESET}{COLOR_DIM}  [+${turn_cost:.4f}]{COLOR_RESET}")
                speak(reply)
            except KeyboardInterrupt:
                print("\nGoodbye.")
                break
    finally:
        session.close()


if __name__ == "__main__":
    main()
