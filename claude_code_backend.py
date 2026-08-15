"""
Persistent Claude Code session for Computron

Spawns one long-running `claude` process per Computron run (not one per turn) so
the system prompt, tool list, and vault/memory context stay warm in
Anthropic's prompt cache instead of being reloaded and re-billed on every
exchange. Authenticates via whatever `claude` is already logged into on this
machine (Jorge's Claude Pro subscription) — no separate API key involved.

Read-only by default: Computron can read files (including the Obsidian vault) and
search the web, but cannot edit, write, or run git commands (add/commit/push
only — no other bash) without an explicit voice confirmation. That's a
deliberate safety choice for a voice-driven assistant with no way to render
a normal permission prompt — see the write-confirmation flow in
ClaudeCodeSession.ask() below.
"""
import base64
import hashlib
import json
import os
import subprocess
from typing import Optional

import config

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SESSION_FILE = os.path.join(PROJECT_DIR, ".session_id")
CAPABILITIES_FILE = os.path.join(PROJECT_DIR, ".capabilities_fingerprint")
PERSONALITY_FILE = os.path.join(PROJECT_DIR, "personality.json")

# ElevenLabs' documented valid range for voice_settings.speed — values
# outside this get rejected by their API, so any value Jorge or Computron
# writes to personality.json gets clamped to it before ever reaching a
# request.
_ELEVENLABS_SPEED_RANGE = (0.7, 1.2)

# Not a secret, so — unlike .env — Computron can actually edit this itself
# through the normal write-confirm flow: no built-in sensitive-file
# classifier stands in the way. Read fresh on every process spawn (not
# fingerprinted alongside SAFE_TOOLS/WRITE_TOOLS below) so a value Jorge
# just changed takes effect on the very next respawn — which happens
# automatically right after any confirmed write turn — without needing the
# tool-permission drift machinery, which is a separate concern. "speed"
# seeds from config.ELEVENLABS_SPEED (the .env default) the first time
# personality.json is created, so Jorge's existing .env value is honored
# as the starting point rather than silently reset to 1.0.
DEFAULT_PERSONALITY = {"humor": 50, "sarcasm": 25, "bluntness": 70, "speed": config.ELEVENLABS_SPEED}


def _read_personality() -> dict:
    try:
        with open(PERSONALITY_FILE) as f:
            values = json.load(f)
        return {**DEFAULT_PERSONALITY, **values}
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(DEFAULT_PERSONALITY)


if not os.path.exists(PERSONALITY_FILE):
    with open(PERSONALITY_FILE, "w") as f:
        json.dump(DEFAULT_PERSONALITY, f, indent=2)


def _personality_prompt_segment() -> str:
    p = _read_personality()
    return (
        f"\n\nPersonality settings (0-100 each, current values — Jorge can "
        f"ask you to change any of these mid-conversation): Humor: "
        f"{p['humor']}/100 — how often genuine wit, wordplay, or a playful "
        f"aside makes it into a reply; higher means being willing to be "
        f"funny even when it's not strictly needed, lower means staying "
        f"matter-of-fact. Sarcasm: {p['sarcasm']}/100 — how much dry, "
        f"needling edge colors your tone; higher leans into ribbing him a "
        f"little, lower stays warm and sincere. Bluntness: {p['bluntness']}"
        f"/100 — how much you cushion an honest but unwelcome take before "
        f"giving it; higher means the unfiltered version straight away, "
        f"lower means softening it first. Actually let these numbers shape "
        f"how you talk, don't just acknowledge them. "
        f"There's a fourth setting, Speed: {p['speed']} — this one is "
        f"different from the other three: it's not a 0-100 scale, it's "
        f"the literal ElevenLabs TTS playback-speed multiplier (1.0 = "
        f"normal), valid roughly {_ELEVENLABS_SPEED_RANGE[0]}-"
        f"{_ELEVENLABS_SPEED_RANGE[1]} — values outside that get clamped "
        f"before use. This one genuinely is something you can adjust "
        f"yourself: if Jorge asks you to talk faster/slower, don't say "
        f"it's not something you can control — it is, through this exact "
        f"mechanism. If Jorge asks you to change any of these four (e.g. "
        f"'turn your sarcasm up to 80,' 'be less blunt,' or 'speak a "
        f"little faster'), treat it exactly like any other self-edit: "
        f"propose the specific new number, and on confirmation write the "
        f"full updated JSON to personality.json in your project directory "
        f"through the write-confirm flow above. Don't claim the change is "
        f"in effect before the file is actually written, and don't expect "
        f"it to color your reply (or, for speed, your voice) on the same "
        f"turn you write it — it takes hold starting your next reply, "
        f"once the file is saved and the process respawns."
    )


def get_voice_speed() -> float:
    """Public accessor for main.py's TTS code — the current speed value
    from personality.json, clamped to ElevenLabs' valid range regardless
    of what's on disk (a stray out-of-range value, written by hand or by
    Computron, should degrade to the nearest valid speed, not break TTS
    outright)."""
    speed = _read_personality()["speed"]
    lo, hi = _ELEVENLABS_SPEED_RANGE
    return max(lo, min(hi, speed))

# Configurable so this works for anyone, not just on a machine with this
# specific vault. COMPUTRON_VAULT_PATH lets a different vault (or no vault
# at all) be used; if it's unset or the path doesn't exist, fall back to
# running from the project directory instead of crashing on a missing cwd.
_configured_vault_path = os.path.expanduser(
    os.getenv("COMPUTRON_VAULT_PATH", "~/Obsidian/The Triforce")
)
if os.path.isdir(_configured_vault_path):
    VAULT_PATH = _configured_vault_path
else:
    VAULT_PATH = PROJECT_DIR
    print(
        f"No vault found at {_configured_vault_path} — running from the "
        f"project directory instead. Set COMPUTRON_VAULT_PATH in .env to "
        f"point at your own vault, if you have one."
    )

SAFE_TOOLS = "Read,Glob,Grep,WebSearch"
# Bash is scoped to git only — not bare Bash — so a confirmed write turn
# can commit and push (e.g. vault memory sync) without opening up
# arbitrary shell command execution off a misheard "yes". Was
# Bash(git add:*),Bash(git commit:*),Bash(git push:*) — narrower on paper,
# but the permission matcher does a literal string-prefix check, and the
# checkpoint commit needs `git -C <dir> ...` (no `cd`, since cwd is the
# vault, not the target project — see the system prompt below), which
# starts with "git -C", not "git add"/"git commit". Verified empirically:
# the three narrow patterns denied every checkpoint attempt 100% of the
# time regardless of confirmation; Bash(git:*) with `git -C` succeeded
# immediately. Broader than add/commit/push (any git subcommand reachable
# during the one granted turn), but still git-only, still single-turn.
WRITE_TOOLS = "Read,Glob,Grep,WebSearch,Edit,Write,Bash(git:*)"

# Short, unambiguous confirmations only — anything longer or more equivocal
# ("maybe", "let me think", a new unrelated question) does NOT grant write
# access. Deliberately conservative: a misheard "yes" is much cheaper than a
# misheard "no".
AFFIRMATIVE_PHRASES = {
    "yes", "yeah", "yep", "yup", "sure", "confirm", "confirmed",
    "go ahead", "do it", "please do", "yes please", "yes go ahead",
    "correct", "affirmative", "sounds good", "do that",
}

# Must match the closing line mandated in SYSTEM_PROMPT_ADDITION exactly —
# used both to instruct the model and, in ask(), to detect whether a
# confirmation window is actually open (see _pending_write_request below).
CONFIRMATION_PROMPT = "Should I go ahead — just say yes."

SYSTEM_PROMPT_ADDITION = (
    "You are being used as Computron, a spoken voice assistant for Jorge "
    "(Computron — a lighthearted Office 'The Banker' callback — is your "
    "name, full stop, in conversation and everywhere else) — but "
    "this should feel like an ongoing conversation with the same Claude he "
    "already collaborates with in Claude Code, not a lookup service. Be "
    "genuinely curious: ask a follow-up question when something's ambiguous "
    "or worth exploring further, offer an angle he might not have "
    "considered, and let the conversation go where it needs to instead of "
    "just answering and stopping. "
    "Critical rule about persistence: whenever Jorge asks you to remember, "
    "save, write down, note, record, or otherwise persist ANYTHING — even "
    "phrased casually, like 'make sure to write that down' or 'keep that in "
    "mind for the project' — you must actually attempt an Edit or Write "
    "tool call for it. Never say 'done', 'saved', 'noted', or anything "
    "implying something was written or persisted unless a tool call for it "
    "actually ran. If you don't have write permission yet, the tool call "
    "will be denied automatically — when that happens, briefly describe "
    f"exactly what you'd change, then end your reply with exactly this and "
    f"nothing after it: '{CONFIRMATION_PROMPT}' That precise closing line — "
    "word for word, every single time — is what actually opens the "
    "confirmation window on the backend; a paraphrase does not. This "
    "applies just as much when you're re-asking after an interruption (a "
    "dropped mic, an unrelated reply, anything that isn't a clear yes) as "
    "it does the first time — always close with the exact line again, not "
    "a rephrased version, or the window silently never opens. Don't bury "
    "it inside a longer sentence, don't pair it with a second unrelated "
    "question in the same breath. A vague conversational "
    "acknowledgment with no tool call is a lie by omission — never do that, "
    "even to be agreeable or keep the conversation moving. And don't "
    "respond to a denial by pivoting to a workaround instead of asking — "
    "no 'here's a copy-paste command for a session that has write access,' "
    "no 'can you check your permission settings,' no suggesting he do it "
    "himself. The confirm-then-retry flow above is the actual mechanism, "
    "not a fallback to route around; always give him the chance to just say "
    "yes before suggesting anything else. "
    "Special case: if a confirmed write actually edits a file inside your "
    "own project directory (your own source code, not a vault note), "
    "immediately after the edit succeeds, commit it as a checkpoint so a "
    f"bad self-edit is always one git revert away from undone — but do "
    f"NOT cd there first (your shell's cwd is the vault, and a `cd ... && "
    f"git ...` command gets denied; that's a real, confirmed bug, not a "
    f"guess). Instead run two separate commands using git's -C flag: "
    f"`git -C {PROJECT_DIR} add <the file(s) you changed>`, then "
    f"`git -C {PROJECT_DIR} commit -m \"<short message describing the "
    f"change>\"`. Never run git push as part of this — pushing is a "
    "separate, more visible action Jorge hasn't asked for, so leave the "
    "commit local and unpushed unless he explicitly says to push. "
    "Permissions are re-evaluated fresh on every turn and can change "
    "between turns or across resumed conversations — Jorge may have granted "
    "a capability since you last checked. If earlier in this conversation "
    "you concluded something was permanently blocked, treat that as stale, "
    "not settled: retry the actual tool call and ask him to confirm again "
    "rather than repeating a past refusal from memory. "
    "Your text output is converted to speech and played aloud, so: never "
    "use markdown, code blocks, bullet points, or headers — plain spoken "
    "sentences only — and keep any single reply to a natural conversational "
    "length (a few sentences, not a monologue) so it doesn't turn into a "
    "lecture. Don't narrate which tool you're about to use, just answer or "
    "ask."
)

# A change to any of these three is a change to what the model is allowed
# or told to do — fingerprint them so a resumed conversation can detect
# "the code changed since we last talked" and say so explicitly, instead of
# relying on the model to reliably infer that from a static prompt while
# arguing against its own prior turns (confirmed unreliable in practice —
# see CAPABILITY_DRIFT_NOTE below).
def _capabilities_fingerprint() -> str:
    fingerprint_input = f"{SAFE_TOOLS}|{WRITE_TOOLS}|{SYSTEM_PROMPT_ADDITION}"
    return hashlib.sha256(fingerprint_input.encode()).hexdigest()[:16]


CAPABILITY_DRIFT_NOTE = (
    "\n\nOne more thing, and treat this as more important than anything "
    "above: your tool permissions or instructions changed since the last "
    "time this conversation ran. If you told Jorge earlier in this "
    "conversation that something was blocked, denied, or impossible — "
    "especially anything about write access, git, or committing/pushing — "
    "that statement is now void. Do not repeat it or reason from it. "
    "Actually attempt the relevant tool call fresh before saying anything "
    "about whether it's possible."
)


class ClaudeCodeSession:
    """Wraps a persistent `claude -p --input-format stream-json` process.

    Resumes the previous Computron conversation automatically (via a stored
    session ID) so quitting and relaunching doesn't lose context — only a
    fresh `claude` process per run, not a fresh conversation.

    Write access (Edit/Write tools, plus git add/commit/push via Bash) is
    off by default. If a turn tries to write and gets denied, the model
    naturally explains what it wanted to do and asks Jorge to confirm (see
    SYSTEM_PROMPT_ADDITION). If his next utterance is an unambiguous "yes",
    the *next* turn only is respawned with write access so the model can
    retry and actually make the change; it then drops straight back to
    read-only for every turn after that.
    """

    def __init__(self, model: str, project_dir: str = None):
        self.model = model
        self.project_dir = project_dir or VAULT_PATH
        self._last_total_cost = 0.0
        self._resumed = False
        self._first_call_done = False
        self._pending_write_request = False
        self._current_tools = SAFE_TOOLS
        # Set via set_workspace() by the VS Code extension (through the HTTP
        # bridge in server.py) — an extra --add-dir so Computron can read
        # (and, once confirmed, write) whatever project is open there,
        # alongside its own project directory. None means no workspace
        # attached yet.
        self.workspace_dir: Optional[str] = None
        self.proc = self._spawn(SAFE_TOOLS)

    def _spawn(self, allowed_tools: str) -> subprocess.Popen:
        resume_id = self._read_session_id()

        current_fingerprint = _capabilities_fingerprint()
        stored_fingerprint = self._read_capabilities_fingerprint()
        # Personality is read fresh here, every spawn, on purpose — it's
        # deliberately outside the fingerprint/drift-note machinery above,
        # which exists for a different problem (a resumed conversation
        # arguing from stale beliefs about tool access). Personality values
        # just need to reflect whatever's on disk right now.
        system_prompt = SYSTEM_PROMPT_ADDITION + _personality_prompt_segment()
        if self.workspace_dir:
            system_prompt += (
                f"\n\nA VS Code workspace directory is currently attached: "
                f"{self.workspace_dir}. You have the same read access there "
                f"as your own project directory. The same self-edit "
                f"git-checkpoint habit applies to it too: if a confirmed "
                f"write edits a file inside {self.workspace_dir}, "
                f"immediately after the edit succeeds, commit it — same as "
                f"for your own project directory: do NOT cd there first "
                f"(your shell's cwd is the vault; a `cd ... && git ...` "
                f"command gets denied), instead run `git -C "
                f"{self.workspace_dir} add <file(s)>` then `git -C "
                f"{self.workspace_dir} commit -m \"<message>\"` as two "
                f"separate commands. "
                f"Never run git push there either, unless Jorge explicitly "
                f"says to. If anything earlier in this conversation claimed "
                f"a different attached workspace (or none at all), that "
                f"claim is now stale — this line is the current, correct "
                f"one. Don't reason from what you said before about which "
                f"workspace is attached; this line always wins."
            )
        if resume_id and stored_fingerprint and stored_fingerprint != current_fingerprint:
            system_prompt += CAPABILITY_DRIFT_NOTE
            print("Permissions/prompt changed since last run — flagging it for this session.")
        self._write_capabilities_fingerprint(current_fingerprint)

        add_dirs = [PROJECT_DIR] + ([self.workspace_dir] if self.workspace_dir else [])
        cmd = [
            "claude", "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--model", self.model,
            "--allowedTools", allowed_tools,
            "--permission-mode", "dontAsk",
            "--add-dir", *add_dirs,
            "--append-system-prompt", system_prompt,
        ]
        self._resumed = bool(resume_id)
        if resume_id:
            cmd += ["--resume", resume_id]
            print(f"Resuming previous conversation ({resume_id[:8]}...)")

        # config.py's load_dotenv() puts ANTHROPIC_API_KEY into this process's
        # env; strip it (and any auth token) before spawning so `claude` falls
        # back to its own login (Claude Pro) instead of API-key billing.
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        # Claude Code's built-in auto-memory writer bypasses --allowedTools
        # entirely — confirmed empirically, it wrote a real file to the local
        # memory bank during a live session with Edit/Write excluded from the
        # allowlist. Disabling it here closes that gap; memory-bank *reading*
        # (via Read/Glob/Grep) is unaffected — verified separately.
        env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
        return subprocess.Popen(
            cmd,
            cwd=self.project_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

    @staticmethod
    def _read_session_id() -> Optional[str]:
        try:
            with open(SESSION_FILE) as f:
                return f.read().strip() or None
        except FileNotFoundError:
            return None

    @staticmethod
    def _write_session_id(session_id: str):
        with open(SESSION_FILE, "w") as f:
            f.write(session_id)

    @staticmethod
    def _clear_session_id():
        try:
            os.remove(SESSION_FILE)
        except FileNotFoundError:
            pass

    @staticmethod
    def _read_capabilities_fingerprint() -> Optional[str]:
        try:
            with open(CAPABILITIES_FILE) as f:
                return f.read().strip() or None
        except FileNotFoundError:
            return None

    @staticmethod
    def _write_capabilities_fingerprint(fingerprint: str):
        with open(CAPABILITIES_FILE, "w") as f:
            f.write(fingerprint)

    @staticmethod
    def _looks_like_confirmation(text: str) -> bool:
        normalized = text.strip().lower().rstrip(".!")
        return normalized in AFFIRMATIVE_PHRASES

    def _switch_tools(self, allowed_tools: str):
        if allowed_tools == self._current_tools:
            return
        self.close()
        self.proc = self._spawn(allowed_tools)
        self._current_tools = allowed_tools
        # total_cost_usd is scoped to the process, not the conversation —
        # confirmed empirically (a resumed process starts back at 0 even
        # though the resumed conversation has prior cost). Reset the
        # baseline on every respawn or turn_cost deltas go negative.
        self._last_total_cost = 0.0

    def set_workspace(self, path: str):
        """Attaches (or switches) the VS Code workspace directory Computron
        can read/write. Respawns the subprocess with an extra --add-dir,
        same pattern as _switch_tools — a no-op if it's already attached."""
        normalized = os.path.abspath(os.path.expanduser(path))
        if normalized == self.workspace_dir:
            return
        self.workspace_dir = normalized
        self.close()
        self.proc = self._spawn(self._current_tools)
        self._last_total_cost = 0.0

    def ask(self, text: str, image_path: Optional[str] = None) -> tuple[str, float]:
        """Sends a message, blocks for the reply. Returns (reply_text, this_turn_cost_usd).

        image_path, if given, attaches that image (e.g. a screenshot for an
        "Ask About Screen" turn) to this message only — never persisted or
        reused on later turns."""
        grant_write = self._pending_write_request and self._looks_like_confirmation(text)
        self._switch_tools(WRITE_TOOLS if grant_write else SAFE_TOOLS)

        try:
            reply, cost, is_error, saw_denied = self._ask_once(text, image_path)
        except (RuntimeError, OSError):
            # OSError covers BrokenPipeError: the subprocess can die before
            # we even finish writing to its stdin (e.g. --resume-ing a
            # session ID that only exists on another machine), which is a
            # different failure point than the read-side RuntimeError below
            # but means the same thing — treat it the same way so the
            # stale-resume retry logic actually gets a chance to run.
            reply, cost, is_error, saw_denied = None, 0.0, True, False

        # A resume attempt can fail without the process crashing (an
        # is_error result for a stale/unknown session ID) or by crashing
        # outright — either way, only on the *first* turn of a resumed
        # session do we treat it as "resume failed" and retry fresh; a
        # later-turn error is a real error, not a bad resume.
        if is_error and self._resumed and not self._first_call_done:
            print("Resume failed — starting a new conversation instead.")
            self._clear_session_id()
            self.close()
            self.proc = self._spawn(self._current_tools)
            self._last_total_cost = 0.0
            try:
                reply, cost, is_error, saw_denied = self._ask_once(text, image_path)
            except (RuntimeError, OSError) as e:
                reply, cost = f"Something went wrong: {e}", 0.0

        self._first_call_done = True
        # Write access is single-turn only — drop back to read-only right
        # after a granted turn, regardless of whether it actually wrote
        # anything, so a stray follow-up "yes" later can't reuse it.
        if grant_write:
            self._switch_tools(SAFE_TOOLS)
        # The confirmation window should stay open across a turn that didn't
        # itself trigger a fresh permission_denied — e.g. a dropped-mic turn
        # where Computron re-asks without retrying the tool call. Keying this
        # off the mandated closing line (CONFIRMATION_PROMPT), not just
        # saw_denied, means any turn that ends with that exact question
        # (first ask or a re-ask) correctly arms the next "yes" — confirmed
        # this was a real bug: a mic-drop turn silently reset saw_denied to
        # False, closing the window before Jorge's actual "yes" landed.
        self._pending_write_request = saw_denied or (
            reply is not None and reply.rstrip().endswith(CONFIRMATION_PROMPT)
        )

        return reply if reply is not None else "Something went wrong.", cost

    def _ask_once(self, text: str, image_path: Optional[str] = None) -> tuple[str, float, bool, bool]:
        content = []
        if image_path:
            # screencapture (the only current source of image_path) always
            # writes PNG, so the media type isn't derived from the path.
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("ascii")
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": encoded},
            })
        content.append({"type": "text", "text": text})
        message = {
            "type": "user",
            "message": {"role": "user", "content": content},
        }
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

        saw_permission_denied = False
        while True:
            line = self.proc.stdout.readline()
            if not line:
                stderr_tail = self.proc.stderr.read(2000) if self.proc.stderr else ""
                raise RuntimeError(f"Claude Code process ended unexpectedly: {stderr_tail}")
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if event.get("type") == "system" and event.get("subtype") == "permission_denied":
                saw_permission_denied = True
                continue

            if event.get("type") == "result":
                session_id = event.get("session_id")
                if session_id:
                    self._write_session_id(session_id)
                if event.get("is_error"):
                    result_text = event.get("result") or "unknown error"
                    return f"Something went wrong: {result_text}", 0.0, True, saw_permission_denied
                total_cost = event.get("total_cost_usd", 0.0)
                turn_cost = total_cost - self._last_total_cost
                self._last_total_cost = total_cost
                return event.get("result", ""), turn_cost, False, saw_permission_denied

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
