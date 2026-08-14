"""
Computron menu bar app — runs persistently in the macOS menu bar with a
global hold-to-talk hotkey (Right Option by default), so it works without
a focused terminal window.

Shares STT/TTS/backend logic with main.py (imports transcribe, speak,
SAMPLE_RATE from it) — only the trigger (a held key instead of Enter) and
the UI (a menu bar icon instead of terminal prompts) are different.

Requires Accessibility permission for whatever process runs this (Terminal,
or the Python interpreter directly) — macOS will prompt for it on first
run; without it, the global hotkey silently never fires. Grant it under
System Settings -> Privacy & Security -> Accessibility.

Run: python menubar_app.py
"""
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np
import rumps
import sounddevice as sd
from pynput import keyboard
from PyObjCTools import AppHelper

import config
from claude_code_backend import ClaudeCodeSession
from main import SAMPLE_RATE, replay, speak, transcribe

try:
    HOTKEY = getattr(keyboard.Key, config.HOTKEY_KEY)
except AttributeError:
    print(f"Unknown COMPUTRON_HOTKEY_KEY '{config.HOTKEY_KEY}' — falling back to alt_r (Right Option).")
    HOTKEY = keyboard.Key.alt_r

ICON_IDLE = "\U0001F399"     # microphone
ICON_RECORDING = "\U0001F534"  # red circle
ICON_THINKING = "⏳"      # hourglass
ICON_SPEAKING = "\U0001F50A"  # speaker


class ComputronApp(rumps.App):
    def __init__(self):
        super().__init__("Computron", title=ICON_IDLE)
        self.session = None
        self._recording = False
        self._frames = []
        self._stream = None
        self._last_reply_wav = None
        self._lock = threading.Lock()
        self.menu = ["Talk now", "Replay last reply", None, "Quit Computron"]

    def _set_tooltip(self, text):
        # AppKit calls must happen on the main thread (macOS's Main Thread
        # Checker traps otherwise, crashing the process) — _process_turn runs
        # on a background thread, so this hops over via AppHelper.callAfter
        # rather than touching nsstatusitem directly.
        def _apply():
            try:
                self._nsapp.nsstatusitem.setToolTip_(text)
            except AttributeError:
                pass
        AppHelper.callAfter(_apply)

    def start_session(self):
        self.title = ICON_THINKING
        print("Starting Claude Code session...")
        self.session = ClaudeCodeSession(model=config.CLAUDE_MODEL)
        self.title = ICON_IDLE
        print(f"Computron menu bar app ready. Hold {config.HOTKEY_KEY} to talk, or use the menu.")

    # --- menu item: same trigger as the hotkey, toggled by click instead of hold ---
    @rumps.clicked("Talk now")
    def talk_now(self, _):
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording_and_process()

    # --- global hotkey: real hold-to-talk ---
    def on_key_press(self, key):
        if key == HOTKEY and not self._recording:
            self._start_recording()

    def on_key_release(self, key):
        if key == HOTKEY and self._recording:
            self._stop_recording_and_process()

    def _start_recording(self):
        with self._lock:
            if self._recording:
                return
            self._recording = True
        self._frames = []
        self.title = ICON_RECORDING
        self._set_tooltip("Listening...")
        print("Recording...")

        def callback(indata, frames_count, time_info, status):
            self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=callback
        )
        self._stream.start()

    def _stop_recording_and_process(self):
        with self._lock:
            if not self._recording:
                return
            self._recording = False
        self._stream.stop()
        self._stream.close()
        self.title = ICON_THINKING

        audio = (
            np.concatenate(self._frames, axis=0)
            if self._frames
            else np.zeros((0, 1), dtype="int16")
        )
        wav_path = Path(tempfile.gettempdir()) / "computron_menubar_input.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio.tobytes())

        # Off the hotkey-listener thread so a quick re-press isn't blocked
        # while transcription/Claude/TTS are running.
        threading.Thread(target=self._process_turn, args=(wav_path,), daemon=True).start()

    def _process_turn(self, wav_path: Path):
        transcript = transcribe(wav_path)
        if not transcript:
            print("(Didn't catch that — try again.)")
            self.title = ICON_IDLE
            self._set_tooltip("Didn't catch that — try again.")
            return
        print(f"You: {transcript}")
        self._set_tooltip(f"You: {transcript}\n\nThinking...")

        reply, turn_cost = self.session.ask(transcript)
        print(f"Computron: {reply}  [+${turn_cost:.4f}]")
        self._set_tooltip(f"You: {transcript}\n\nComputron: {reply}")

        self.title = ICON_SPEAKING
        self._last_reply_wav = speak(reply)
        self.title = ICON_IDLE

    @rumps.clicked("Replay last reply")
    def replay_last(self, _):
        # Off the main/UI thread — afplay blocks for the duration of
        # playback, and this handler fires on the main thread (same reason
        # _stop_recording_and_process hands _process_turn off to a thread).
        threading.Thread(target=self._do_replay, daemon=True).start()

    def _do_replay(self):
        self.title = ICON_SPEAKING
        if not replay(self._last_reply_wav):
            print("(Nothing to replay yet.)")
            self._set_tooltip("Nothing to replay yet.")
        self.title = ICON_IDLE

    @rumps.clicked("Quit Computron")
    def quit_app(self, _):
        if self.session:
            self.session.close()
        rumps.quit_application()


def main():
    app = ComputronApp()
    app.start_session()

    listener = keyboard.Listener(on_press=app.on_key_press, on_release=app.on_key_release)
    listener.daemon = True
    listener.start()

    app.run()


if __name__ == "__main__":
    main()
