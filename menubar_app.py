"""
Computron menu bar app — runs persistently in the macOS menu bar with a
global hold-to-talk hotkey (Right Option by default), so it works without
a focused terminal window. A second hotkey (Left Option by default) does
the same hold-to-talk, but attaches a screenshot to that turn only —
"Ask About Screen" — and needs the separate Screen Recording permission
(System Settings -> Privacy & Security -> Screen Recording), which only
takes effect after the granted process's parent (e.g. Terminal) relaunches.

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
import waveform_settings
import waveform_window
from claude_code_backend import AFFIRMATIVE_PHRASES, ClaudeCodeSession
from main import (
    SAMPLE_RATE, capture_screenshot, read_clipboard_aloud, replay,
    set_playback_listener, speak, stop_speaking, transcribe,
)
from server import start_server
from terminal_watcher import start_watching

try:
    HOTKEY = getattr(keyboard.Key, config.HOTKEY_KEY)
except AttributeError:
    print(f"Unknown COMPUTRON_HOTKEY_KEY '{config.HOTKEY_KEY}' — falling back to alt_r (Right Option).")
    HOTKEY = keyboard.Key.alt_r

try:
    SCREEN_HOTKEY = getattr(keyboard.Key, config.SCREEN_HOTKEY_KEY)
except AttributeError:
    print(f"Unknown COMPUTRON_SCREEN_HOTKEY_KEY '{config.SCREEN_HOTKEY_KEY}' — falling back to alt_l (Left Option).")
    SCREEN_HOTKEY = keyboard.Key.alt_l

ICON_IDLE = "\U0001F399"     # microphone
ICON_RECORDING = "\U0001F534"  # red circle
ICON_THINKING = "⏳"      # hourglass
ICON_SPEAKING = "\U0001F50A"  # speaker


class ComputronApp(rumps.App):
    def __init__(self):
        # quit_button=None: rumps otherwise auto-adds its own bare "Quit"
        # item bound straight to rumps.quit_application(), bypassing our
        # own "Quit Computron" handler's cleanup (HTTP server shutdown,
        # closing the Claude subprocess) — two quit items, only one safe.
        super().__init__("Computron", title=ICON_IDLE, quit_button=None)
        self.session = None
        self._recording = False
        self._frames = []
        self._stream = None
        # Set at the start of an "Ask About Screen" turn (SCREEN_HOTKEY or
        # its menu item), cleared once handed off to _process_turn. None
        # means the in-progress recording is a normal, screenshot-less turn.
        self._recording_screenshot = None
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
        # Both default on (matching current behavior) — these are opt-out,
        # not opt-in, since neither has Auto-Read's ongoing-TTS-cost reason
        # to default off. Exists so someone running Computron who doesn't
        # want screen capture (privacy, or no Screen Recording permission
        # granted) or the floating waveform window (visual distraction, or
        # just doesn't want it) can turn either off without losing the rest.
        self.screen_awareness_enabled = True
        self.waveform_enabled = True

        # --- "Waveform Settings" submenu: color/opacity/size/bars/gap/margin
        # via text-prompt dialogs, position via a checkmarked preset list.
        # All write through waveform_settings.update(), the same file the
        # `python waveform_settings.py` CLI edits — see that module's
        # docstring for why no restart or file-watcher is needed for either
        # surface to take effect.
        waveform_menu = rumps.MenuItem("Waveform Settings")
        current_settings = waveform_settings.read()
        style_menu = rumps.MenuItem("Style")
        self._waveform_style_items = {}
        for style in waveform_settings.STYLES:
            item = rumps.MenuItem(style.capitalize(), callback=self._set_waveform_style)
            item.state = style == current_settings["style"]
            style_menu.add(item)
            self._waveform_style_items[style] = item
        waveform_menu.add(style_menu)
        waveform_menu.add(None)
        presets_menu = rumps.MenuItem("Presets")
        for preset_name in waveform_settings.PRESETS:
            presets_menu.add(rumps.MenuItem(preset_name, callback=self._apply_waveform_preset))
        waveform_menu.add(presets_menu)
        waveform_menu.add(None)
        waveform_menu.add(rumps.MenuItem("Color...", callback=self._edit_waveform_color))
        waveform_menu.add(rumps.MenuItem("Opacity...", callback=self._edit_waveform_opacity))
        waveform_menu.add(rumps.MenuItem("Size...", callback=self._edit_waveform_size))
        waveform_menu.add(rumps.MenuItem("Bar Count...", callback=self._edit_waveform_bars))
        waveform_menu.add(rumps.MenuItem("Bar Gap...", callback=self._edit_waveform_bar_gap))
        waveform_menu.add(rumps.MenuItem("Margin...", callback=self._edit_waveform_margin))
        waveform_menu.add(None)
        self._waveform_bg_toggle_item = rumps.MenuItem(
            "Background: On" if current_settings["bg_enabled"] else "Background: Off",
            callback=self._toggle_waveform_background,
        )
        waveform_menu.add(self._waveform_bg_toggle_item)
        waveform_menu.add(rumps.MenuItem("Background Color...", callback=self._edit_waveform_bg_color))
        waveform_menu.add(rumps.MenuItem("Background Opacity...", callback=self._edit_waveform_bg_opacity))
        waveform_menu.add(None)
        position_menu = rumps.MenuItem("Position")
        current_position = current_settings["position"]
        self._waveform_position_items = {}
        for pos in waveform_settings.POSITIONS:
            item = rumps.MenuItem(pos, callback=self._set_waveform_position)
            item.state = pos == current_position
            position_menu.add(item)
            self._waveform_position_items[pos] = item
        waveform_menu.add(position_menu)
        waveform_menu.add(None)
        waveform_menu.add(rumps.MenuItem("Reset to Defaults", callback=self._reset_waveform_settings))

        self.menu = [
            "Talk now", "Ask About Screen", "Replay last reply",
            "Read Clipboard", "Auto-Read: Off", "Screen Awareness: On",
            "Waveform: On", waveform_menu, None, "Quit Computron",
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
        # Floating waveform: main.py calls this with the audio path right
        # as _play() starts, and None right as it stops. main.py itself
        # never touches AppKit, so the hop to the main thread happens here.
        # Gated on waveform_enabled here (not inside waveform_window) so
        # main.py's playback path and the cost of _play()'s call stay
        # identical either way — only whether the window actually shows
        # changes.
        set_playback_listener(
            lambda path: AppHelper.callAfter(
                waveform_window.on_playback_change,
                path if self.waveform_enabled else None,
            )
        )
        self.title = ICON_IDLE
        print(
            f"Computron menu bar app ready. Hold {config.HOTKEY_KEY} to talk, "
            f"hold {config.SCREEN_HOTKEY_KEY} to ask about your screen, or use the menu."
        )

    # --- menu items: same triggers as the hotkeys, toggled by click instead of hold ---
    @rumps.clicked("Talk now")
    def talk_now(self, _):
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording_and_process()

    @rumps.clicked("Ask About Screen")
    def ask_about_screen(self, _):
        if not self._recording:
            self._start_recording(with_screenshot=self.screen_awareness_enabled)
        else:
            self._stop_recording_and_process()

    # --- global hotkeys: real hold-to-talk ---
    def on_key_press(self, key):
        if self._recording:
            return
        if key == HOTKEY:
            self._start_recording()
        elif key == SCREEN_HOTKEY:
            # Falls back to a normal (screenshot-less) voice turn when the
            # toggle is off, rather than doing nothing on a held key — a
            # silently dead hotkey is more confusing than a plain answer.
            self._start_recording(with_screenshot=self.screen_awareness_enabled)

    def on_key_release(self, key):
        if key in (HOTKEY, SCREEN_HOTKEY) and self._recording:
            self._stop_recording_and_process()

    def _start_recording(self, with_screenshot: bool = False):
        with self._lock:
            if self._recording:
                return
            self._recording = True
        # Barge-in: pressing a hotkey (or clicking a menu item) while
        # Computron is still talking interrupts playback immediately,
        # rather than recording your next question over the top of it.
        # No-op if nothing's currently playing.
        if stop_speaking():
            print("(Interrupted — go ahead.)")
        self._frames = []
        if with_screenshot:
            self._recording_screenshot = capture_screenshot()
            if self._recording_screenshot is None:
                print("(Screenshot capture failed — Screen Recording permission granted and Terminal relaunched? Continuing without one.)")
        else:
            self._recording_screenshot = None
        self.title = ICON_RECORDING
        self._set_tooltip("Listening (screen attached)..." if with_screenshot else "Listening...")
        print("Recording (screen attached)..." if with_screenshot else "Recording...")

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
        screenshot = self._recording_screenshot
        self._recording_screenshot = None
        threading.Thread(target=self._process_turn, args=(wav_path, screenshot), daemon=True).start()

    def _process_turn(self, wav_path: Path, screenshot: Path = None):
        transcript = transcribe(wav_path)
        if not transcript:
            print("(Didn't catch that — try again.)")
            self.title = ICON_IDLE
            self._set_tooltip("Didn't catch that — try again.")
            return
        print(f"You: {transcript}" + (" [+screen]" if screenshot else ""))
        self._set_tooltip(f"You: {transcript}\n\nThinking...")

        with self.session_lock:
            text = self._compose_with_editor_context(transcript)
            reply, turn_cost = self.session.ask(text, image_path=screenshot)
        self.last_turn = {
            "id": time.time(), "text": transcript, "reply": reply,
            "cost": turn_cost, "source": "screen" if screenshot else "voice",
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

    @rumps.clicked("Screen Awareness: On")
    def toggle_screen_awareness(self, sender):
        self.screen_awareness_enabled = not self.screen_awareness_enabled
        sender.title = "Screen Awareness: On" if self.screen_awareness_enabled else "Screen Awareness: Off"

    @rumps.clicked("Waveform: On")
    def toggle_waveform(self, sender):
        self.waveform_enabled = not self.waveform_enabled
        sender.title = "Waveform: On" if self.waveform_enabled else "Waveform: Off"
        # Hide immediately if turned off mid-playback — otherwise it'd keep
        # showing whatever's already animating until the next turn.
        if not self.waveform_enabled:
            AppHelper.callAfter(waveform_window.on_playback_change, None)

    def _apply_waveform_preset(self, sender):
        # Presets only ever touch color + opacity (see waveform_settings.py's
        # PRESETS comment) — position/background menu state never changes
        # from this, so nothing here needs re-syncing afterward.
        waveform_settings.apply_preset(sender.title)

    def _toggle_waveform_background(self, sender):
        new_state = not waveform_settings.read()["bg_enabled"]
        waveform_settings.update(bg_enabled=new_state)
        sender.title = "Background: On" if new_state else "Background: Off"

    def _edit_waveform_bg_color(self, _):
        self._prompt_waveform_value(
            "Background Color", "Hex RGB (e.g. 000000), leading # optional:", "bg_color", str,
        )

    def _edit_waveform_bg_opacity(self, _):
        self._prompt_waveform_value("Background Opacity", "0 to 1 (e.g. 0.5):", "bg_opacity", float)

    def _prompt_waveform_value(self, title: str, message: str, key: str, parse):
        current = waveform_settings.read()[key]
        response = rumps.Window(
            message=message, title=title, default_text=str(current),
            ok="Save", cancel="Cancel",
        ).run()
        if not response.clicked:
            return
        try:
            waveform_settings.update(**{key: parse(response.text.strip())})
        except ValueError as e:
            rumps.alert(title=f"Invalid {title}", message=str(e))

    def _edit_waveform_color(self, _):
        self._prompt_waveform_value(
            "Waveform Color", "Hex RGB (e.g. 00CFFF), leading # optional:", "color", str,
        )

    def _edit_waveform_opacity(self, _):
        self._prompt_waveform_value("Waveform Opacity", "0 to 1 (e.g. 0.85):", "opacity", float)

    def _edit_waveform_bars(self, _):
        self._prompt_waveform_value("Waveform Bar Count", "Number of bars (e.g. 9):", "bars", int)

    def _edit_waveform_bar_gap(self, _):
        self._prompt_waveform_value("Waveform Bar Gap", "Pixels between bars (e.g. 4):", "bar_gap", float)

    def _edit_waveform_margin(self, _):
        self._prompt_waveform_value(
            "Waveform Margin", "Pixels from the screen edge (e.g. 80):", "margin", int,
        )

    def _edit_waveform_size(self, _):
        current = waveform_settings.read()
        response = rumps.Window(
            message="Width and height in pixels, separated by a space (e.g. 220 60):",
            title="Waveform Size",
            default_text=f"{current['width']} {current['height']}",
            ok="Save", cancel="Cancel",
        ).run()
        if not response.clicked:
            return
        try:
            parts = response.text.strip().split()
            if len(parts) != 2:
                raise ValueError("Enter two numbers separated by a space, e.g. '220 60'.")
            waveform_settings.update(width=int(parts[0]), height=int(parts[1]))
        except ValueError as e:
            rumps.alert(title="Invalid Size", message=str(e))

    def _set_waveform_position(self, sender):
        try:
            waveform_settings.update(position=sender.title)
        except ValueError as e:
            rumps.alert(title="Invalid Position", message=str(e))
            return
        for pos, item in self._waveform_position_items.items():
            item.state = pos == sender.title

    def _set_waveform_style(self, sender):
        style = sender.title.lower()
        try:
            waveform_settings.update(style=style)
        except ValueError as e:
            rumps.alert(title="Invalid Style", message=str(e))
            return
        for name, item in self._waveform_style_items.items():
            item.state = name == style

    def _reset_waveform_settings(self, _):
        defaults = waveform_settings.reset()
        for pos, item in self._waveform_position_items.items():
            item.state = pos == defaults["position"]
        for name, item in self._waveform_style_items.items():
            item.state = name == defaults["style"]
        self._waveform_bg_toggle_item.title = "Background: On" if defaults["bg_enabled"] else "Background: Off"

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
