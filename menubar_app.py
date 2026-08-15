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
import time
import wave
from pathlib import Path

import numpy as np
import rumps
import sounddevice as sd
from pynput import keyboard
from PyObjCTools import AppHelper

import config
from claude_code_backend import AFFIRMATIVE_PHRASES, ClaudeCodeSession
from main import SAMPLE_RATE, read_clipboard_aloud, replay, speak, stop_speaking, transcribe
from server import start_server
from terminal_watcher import start_watching

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
        # Guards every session.ask() call (voice turns and HTTP turns from
        # the VS Code extension) so concurrent callers can't interleave on
        # the one persistent claude subprocess's stdin/stdout.
        self.session_lock = threading.Lock()
        self.last_turn = None
        self.http_server = None
        # Kept fresh by the VS Code extension via POST /editor-state
        # (active file path + cursor line) so voice turns can be
        # editor-aware too, not just typed Ask commands.
        self.editor_state = None
        # Auto-Read mode: tails the attached workspace's `claude` terminal
        # session (terminal_watcher.py) and speaks new responses aloud. The
        # watcher thread always runs (started in start_session below) but
        # only speaks while this is True — off by default, opt-in, since
        # it's a standing background behavior with real ongoing TTS cost.
        self.auto_read_enabled = False
        self.menu = [
            "Talk now", "Replay last reply", "Read Clipboard",
            "Auto-Read: Off", None, "Quit Computron",
        ]

    # Used by server.py so an HTTP-driven ask (from the VS Code extension)
    # flips the icon to "thinking" too, not just voice turns — the same
    # menu bar state server.py's /status reports back to the extension.
    def set_state(self, state: str):
        self.title = {
            "idle": ICON_IDLE, "recording": ICON_RECORDING,
            "thinking": ICON_THINKING, "speaking": ICON_SPEAKING,
        }[state]

    def _compose_with_editor_context(self, text: str) -> str:
        """Prepends the active VS Code file/cursor line (kept fresh by the
        extension via POST /editor-state) to a voice transcript — without
        this, a spoken "what's this line doing" has no way to know what's
        on screen and answers from stale conversation history instead.
        No-op if no editor state has been reported yet (or VS Code isn't
        running/connected).

        Also a no-op for bare confirmations ("yes", "go ahead", etc.) —
        real bug, caught live: ClaudeCodeSession._looks_like_confirmation()
        does an exact match against AFFIRMATIVE_PHRASES on whatever text
        it receives. Wrapping a plain "yes" in the file-context prefix
        made it no longer match, so write access silently stopped
        granting the moment editor_state started being populated — the
        first couple of confirmations in a session worked (before VS Code
        had reported anything yet), then every one after quietly failed.
        """
        state = self.editor_state
        if not state or not state.get("path"):
            return text
        normalized = text.strip().lower().rstrip(".!")
        if normalized in AFFIRMATIVE_PHRASES:
            return text
        line = state.get("line")
        line_part = f", cursor on line {line}" if line else ""
        return f"Jorge's active file in VS Code: {state['path']}{line_part}.\nJorge asks: {text}"

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
        self.http_server = start_server(self, config.SERVER_PORT)
        start_watching(self)
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
        # Barge-in: pressing the hotkey (or clicking "Talk now") while
        # Computron is still talking interrupts playback immediately,
        # rather than recording your next question over the top of it.
        # No-op if nothing's currently playing.
        if stop_speaking():
            print("(Interrupted — go ahead.)")
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

        with self.session_lock:
            reply, turn_cost = self.session.ask(self._compose_with_editor_context(transcript))
        self.last_turn = {
            "id": time.time(), "text": transcript, "reply": reply,
            "cost": turn_cost, "source": "voice",
        }
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

    @rumps.clicked("Read Clipboard")
    def read_clipboard(self, _):
        threading.Thread(target=self._do_read_clipboard, daemon=True).start()

    def _do_read_clipboard(self):
        self.title = ICON_SPEAKING
        wav = read_clipboard_aloud()
        if wav is None:
            print("(Clipboard empty or unreadable.)")
            self._set_tooltip("Clipboard empty or unreadable.")
        else:
            self._last_reply_wav = wav
        self.title = ICON_IDLE

    @rumps.clicked("Auto-Read: Off")
    def toggle_auto_read(self, sender):
        self.auto_read_enabled = not self.auto_read_enabled
        sender.title = "Auto-Read: On" if self.auto_read_enabled else "Auto-Read: Off"
        # Interrupt any narration/reply immediately on turning it off —
        # otherwise it'd keep speaking whatever's already queued/playing.
        if not self.auto_read_enabled:
            stop_speaking()

    @rumps.clicked("Quit Computron")
    def quit_app(self, _):
        if self.http_server:
            self.http_server.shutdown()
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
