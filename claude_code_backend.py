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
import hashlib
import json
import os
import subprocess
from typing import Optional

VAULT_PATH = os.path.expanduser("~/Obsidian/The Triforce")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SESSION_FILE = os.path.join(PROJECT_DIR, ".session_id")
CAPABILITIES_FILE = os.path.join(PROJECT_DIR, ".capabilities_fingerprint")

SAFE_TOOLS = "Read,Glob,Grep,WebSearch"
# Bash is scoped to git add/commit/push only — not bare Bash — so a
# confirmed write turn can commit and push (e.g. vault memory sync) without
# opening up arbitrary shell command execution off a misheard "yes".
WRITE_TOOLS = (
    "Read,Glob,Grep,WebSearch,Edit,Write,"
    "Bash(git add:*),Bash(git commit:*),Bash(git push:*)"
)

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
        self.proc = self._spawn(SAFE_TOOLS)

    def _spawn(self, allowed_tools: str) -> subprocess.Popen:
        resume_id = self._read_session_id()

        current_fingerprint = _capabilities_fingerprint()
        stored_fingerprint = self._read_capabilities_fingerprint()
        system_prompt = SYSTEM_PROMPT_ADDITION
        if resume_id and stored_fingerprint and stored_fingerprint != current_fingerprint:
            system_prompt += CAPABILITY_DRIFT_NOTE
            print("Permissions/prompt changed since last run — flagging it for this session.")
        self._write_capabilities_fingerprint(current_fingerprint)

        cmd = [
            "claude", "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--model", self.model,
            "--allowedTools", allowed_tools,
            "--permission-mode", "dontAsk",
            "--add-dir", PROJECT_DIR,
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

    def ask(self, text: str) -> tuple[str, float]:
        """Sends a message, blocks for the reply. Returns (reply_text, this_turn_cost_usd)."""
        grant_write = self._pending_write_request and self._looks_like_confirmation(text)
        self._switch_tools(WRITE_TOOLS if grant_write else SAFE_TOOLS)

        try:
            reply, cost, is_error, saw_denied = self._ask_once(text)
        except RuntimeError:
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
                reply, cost, is_error, saw_denied = self._ask_once(text)
            except RuntimeError as e:
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

    def _ask_once(self, text: str) -> tuple[str, float, bool, bool]:
        message = {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": text}]},
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
