"""
Loads settings from .env so nothing sensitive is hardcoded.
"""
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # unused by the Claude Code backend — it authenticates via `claude`'s own login (Claude Pro). Kept for a future raw-API mode.
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
PIPER_BIN = os.getenv("PIPER_BIN", "piper")
PIPER_VOICE = os.getenv("PIPER_VOICE", "./voices/en_US-lessac-medium.onnx")
# Piper's phoneme-length multiplier: 1.0 = normal, lower = faster (e.g. 0.8 = ~20% faster), higher = slower.
PIPER_LENGTH_SCALE = float(os.getenv("PIPER_LENGTH_SCALE", "1.0"))
# Global push-to-talk key for the menu bar app (menubar_app.py) — hold to
# record, release to send. Must be a name from pynput.keyboard.Key, e.g.
# alt_r, alt_l, cmd_r, ctrl_r. Right Option by default: rarely bound to
# anything else system-wide, easy to reach, unlikely to fight other apps.
HOTKEY_KEY = os.getenv("COMPUTRON_HOTKEY_KEY", "alt_r")
# Second push-to-talk hotkey, same hold-to-record/release-to-send shape,
# but attaches a screenshot to that turn only ("Ask About Screen"). Left
# Option by default — a distinct physical key from HOTKEY_KEY so a normal
# voice turn never pays for a screenshot capture + larger payload.
SCREEN_HOTKEY_KEY = os.getenv("COMPUTRON_SCREEN_HOTKEY_KEY", "alt_l")

# TTS provider: "piper" (local, free — the default) or "elevenlabs" (paid
# API, more natural voice). Opt-in only — set TTS_PROVIDER=elevenlabs in
# .env to switch. If ELEVENLABS_API_KEY is missing, main.speak() falls back
# to Piper regardless of this setting.
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "piper")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# "Rachel" — one of ElevenLabs' widely-used premade voices; swap via .env.
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
# Turbo model: optimized for low latency, which matters more than peak
# quality for a real-time voice assistant — eleven_multilingual_v2 is
# higher-fidelity but noticeably slower to generate.
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5")
# Playback speed multiplier: 1.0 = normal. ElevenLabs' valid range is
# roughly 0.7-1.2 — values outside that get rejected by their API.
ELEVENLABS_SPEED = float(os.getenv("ELEVENLABS_SPEED", "1.0"))

# Local-only HTTP bridge (127.0.0.1) the menu bar app exposes so the
# companion VS Code extension can ask questions and attach a workspace
# directory. No auth — both processes run under the same user session.
SERVER_PORT = int(os.getenv("COMPUTRON_SERVER_PORT", "4317"))

# Auto-Read mode: fenced code blocks at or under this many lines are read
# aloud as-is; longer ones are replaced with a short spoken placeholder
# instead (see terminal_watcher.py) — short snippets are worth hearing,
# a whole function isn't.
AUTO_READ_CODE_LINE_LIMIT = int(os.getenv("AUTO_READ_CODE_LINE_LIMIT", "3"))

# Floating waveform visualizer look — these are only the *seed* defaults,
# used the first time waveform_settings.json is created (same relationship
# config.ELEVENLABS_SPEED has to personality.json's "speed"). After that,
# waveform_settings.json is the live source of truth: edit it from the menu
# bar's "Waveform Settings" submenu or via `python waveform_settings.py`,
# both take effect on the next spoken reply with no restart needed. Defaults
# below match the window's original hardcoded appearance.
# One of: bars (spectrum equalizer bars), line (a filled amplitude ribbon
# that scrolls left), orb (a Siri-style pulsing circle).
WAVEFORM_STYLE = os.getenv("COMPUTRON_WAVEFORM_STYLE", "bars")
WAVEFORM_COLOR = os.getenv("COMPUTRON_WAVEFORM_COLOR", "FFFFFF")  # hex RGB, '#' optional
WAVEFORM_OPACITY = float(os.getenv("COMPUTRON_WAVEFORM_OPACITY", "0.85"))  # 0-1
WAVEFORM_WIDTH = int(os.getenv("COMPUTRON_WAVEFORM_WIDTH", "220"))
WAVEFORM_HEIGHT = int(os.getenv("COMPUTRON_WAVEFORM_HEIGHT", "60"))
WAVEFORM_BARS = int(os.getenv("COMPUTRON_WAVEFORM_BARS", "9"))
WAVEFORM_BAR_GAP = float(os.getenv("COMPUTRON_WAVEFORM_BAR_GAP", "4.0"))
# One of: bottom-center, bottom-left, bottom-right, top-center, top-left,
# top-right, center. Falls back to bottom-center (with a printed warning)
# on anything else.
WAVEFORM_POSITION = os.getenv("COMPUTRON_WAVEFORM_POSITION", "bottom-center")
# Distance in px from whichever screen edge(s) the chosen position hugs —
# meaningless for "center", which ignores it.
WAVEFORM_MARGIN = int(os.getenv("COMPUTRON_WAVEFORM_MARGIN", "80"))
