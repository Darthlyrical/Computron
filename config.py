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
