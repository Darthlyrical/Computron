"""
Auto-Read mode: tails the currently-attached VS Code project's `claude`
CLI session transcript (written by Claude Code itself under
~/.claude/projects/) and speaks new assistant text aloud as it's written —
so Jorge doesn't have to copy/paste to hear a response.

Scoped to the attached workspace, not "whatever .jsonl changed most
recently anywhere" — that would eventually pick up Computron's own
`claude -p --resume` subprocess talking to itself (it writes the exact
same kind of session file, under the vault's project folder) and read
that aloud too. Deriving the target directory from the same workspace_dir
the VS Code extension already reports is what naturally avoids that,
without needing a special-case exclusion.
"""
import json
import re
import threading
import time
from pathlib import Path
from typing import Optional

import config
from main import speak

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"

CODE_BLOCK_PLACEHOLDER = "View code block provided."

# Captures the code body only — the (optional) language tag right after
# the opening fence and the newline before it are consumed but not kept.
_CODE_FENCE_RE = re.compile(r"```\w*\n?(.*?)```", re.DOTALL)
_HEADER_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_BOLD_RE = re.compile(r"(\*\*|__)(.*?)\1")
_ITALIC_RE = re.compile(r"(\*|_)(.*?)\1")
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def _replace_code_fence(match: "re.Match[str]") -> str:
    """Short snippets (<= AUTO_READ_CODE_LINE_LIMIT lines) are worth
    hearing — read as-is. Longer ones are replaced with a short spoken
    placeholder instead of either reading a whole function aloud or
    silently vanishing."""
    code = match.group(1).strip("\n")
    lines = code.splitlines()
    if len(lines) <= config.AUTO_READ_CODE_LINE_LIMIT:
        return code
    return CODE_BLOCK_PLACEHOLDER


def _strip_markdown_for_speech(text: str) -> str:
    """Handles fenced code blocks per _replace_code_fence (small ones read
    as-is, large ones become a placeholder) and strips common markdown
    syntax so auto-read narration doesn't come out as literal
    asterisks/backticks. Inline code (`like this`) already gets its
    content read — only the backtick markers are stripped, not the text.
    Order matters: bold (**/__) before italic (*/_), since italic's regex
    would otherwise chew through a bold pair's asterisks first."""
    text = _CODE_FENCE_RE.sub(_replace_code_fence, text)
    text = _HEADER_RE.sub("", text)
    text = _BOLD_RE.sub(r"\2", text)
    text = _ITALIC_RE.sub(r"\2", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _BLANK_RUN_RE.sub("\n\n", text)
    return text.strip()


def _project_session_dir(workspace_dir: str) -> Path:
    """Claude Code's own convention: every non-alphanumeric character in the
    project path becomes a hyphen, one-for-one — not just '/'. Verified
    empirically against a real path with a space in it (the vault:
    '.../Obsidian/The Triforce' -> '...-Obsidian-The-Triforce'), which a
    naive slash-only replace gets wrong."""
    sanitized = re.sub(r"[^a-zA-Z0-9]", "-", workspace_dir)
    return CLAUDE_PROJECTS_DIR / sanitized


def _active_session_file(workspace_dir: Optional[str]) -> Optional[Path]:
    """The most-recently-modified *.jsonl directly inside the attached
    workspace's project folder — Jorge runs one `claude` terminal session
    per project in practice, so "most recently written to" reliably means
    "the one currently in use." Excludes anything under subagents/ (those
    are sub-conversations, not top-level sessions) by only globbing direct
    children, not walking the tree. Returns None if no workspace is
    attached or the project folder doesn't exist (e.g. Jorge has never run
    `claude` there)."""
    if not workspace_dir:
        return None
    session_dir = _project_session_dir(workspace_dir)
    if not session_dir.is_dir():
        return None
    candidates = [p for p in session_dir.glob("*.jsonl") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _extract_speakable_texts(raw_line: bytes) -> list[str]:
    """Parses one JSONL line; returns stripped text worth speaking for each
    text-type content block in an assistant message (empty list for
    anything else — thinking/tool_use blocks, non-assistant events, blocks
    that are pure code and strip to nothing, or malformed lines)."""
    try:
        event = json.loads(raw_line)
    except json.JSONDecodeError:
        return []
    if event.get("type") != "assistant":
        return []
    results = []
    for block in event.get("message", {}).get("content", []):
        if block.get("type") != "text":
            continue
        spoken = _strip_markdown_for_speech(block.get("text", ""))
        if spoken:
            results.append(spoken)
    return results


def watch_and_speak(app) -> None:
    """Runs forever on a daemon thread. Re-derives the target session file
    every cycle from app.session.workspace_dir, so switching the attached
    VS Code workspace automatically retargets which session is watched.
    Only speaks while app.auto_read_enabled is True — otherwise just keeps
    tracking the read position so toggling on mid-session doesn't dump a
    backlog of everything missed while off."""
    current_file: Optional[Path] = None
    read_position = 0

    while True:
        time.sleep(1)
        if app.session is None:
            continue

        target = _active_session_file(app.session.workspace_dir)
        if target != current_file:
            # New session (workspace changed, or a fresh `claude` invocation
            # started a new session file in the same project) — tail from
            # here on, never replay backlog.
            current_file = target
            read_position = target.stat().st_size if target else 0
            continue
        if current_file is None:
            continue

        try:
            with open(current_file, "rb") as f:
                f.seek(read_position)
                chunk = f.read()
        except FileNotFoundError:
            current_file = None
            continue
        if not chunk:
            continue

        # Only consume complete lines — a trailing partial line (the writer
        # hasn't flushed the rest yet) is left for the next poll instead of
        # being parsed now and silently dropped forever.
        last_newline = chunk.rfind(b"\n")
        if last_newline == -1:
            continue
        complete = chunk[: last_newline + 1]
        read_position += len(complete)

        if not app.auto_read_enabled:
            continue  # position already advanced; just not speaking right now

        for raw_line in complete.split(b"\n"):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            for spoken in _extract_speakable_texts(raw_line):
                if not app.auto_read_enabled:
                    break  # toggled off mid-batch
                speak(spoken)


def start_watching(app) -> threading.Thread:
    """Starts the watcher on a daemon thread. Safe to call once at app
    startup regardless of whether auto-read is enabled yet — the loop
    itself gates on app.auto_read_enabled."""
    thread = threading.Thread(target=watch_and_speak, args=(app,), daemon=True)
    thread.start()
    return thread
