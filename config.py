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
