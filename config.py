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

# Local-only HTTP bridge (127.0.0.1) the menu bar app exposes so the
# companion VS Code extension can ask questions and attach a workspace
# directory. No auth — both processes run under the same user session.
SERVER_PORT = int(os.getenv("COMPUTRON_SERVER_PORT", "4317"))

# Auto-Read mode: fenced code blocks at or under this many lines are read
# aloud as-is; longer ones are replaced with a short spoken placeholder
# instead (see terminal_watcher.py) — short snippets are worth hearing,
# a whole function isn't.
AUTO_READ_CODE_LINE_LIMIT = int(os.getenv("AUTO_READ_CODE_LINE_LIMIT", "3"))
