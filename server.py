"""
Local-only HTTP bridge so the companion VS Code extension can talk to the
already-running Computron menu bar app — ask questions as text, attach the
current workspace directory for file access, and mirror voice replies as
text in-editor.

Bound to 127.0.0.1 only. No auth: both processes run under the same user
session on the same machine, so there's nothing to authenticate.

The app (ComputronApp in menubar_app.py) owns the one ClaudeCodeSession and
a threading.Lock shared between voice turns and HTTP turns — every call
into session.ask() goes through that lock so concurrent voice/HTTP requests
can't interleave on the subprocess's stdin/stdout.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from main import speak

# Mirrors ICON_IDLE/ICON_RECORDING/ICON_THINKING/ICON_SPEAKING in
# menubar_app.py — derived from self.app.title (the single source of
# truth for the menu bar icon) rather than tracked separately, so this
# can't drift out of sync with the actual icon.
_STATE_BY_ICON = {
    "\U0001F399": "idle",
    "\U0001F534": "recording",
    "⏳": "thinking",
    "\U0001F50A": "speaking",
}


class ComputronRequestHandler(BaseHTTPRequestHandler):
    # Set by start_server() before the server is created — every handler
    # instance reaches back into the running ComputronApp through this.
    app = None

    def log_message(self, fmt, *args):
        pass  # BaseHTTPRequestHandler logs every request to stderr by default — noisy, skip it.

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def do_GET(self):
        if self.path == "/status":
            state = _STATE_BY_ICON.get(self.app.title, "unknown")
            self._send_json(200, {"ready": self.app.session is not None, "state": state})
        elif self.path == "/last-turn":
            self._send_json(200, self.app.last_turn or {})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return

        if self.path == "/ask":
            text = (body.get("text") or "").strip()
            if not text:
                self._send_json(400, {"error": "missing 'text'"})
                return
            if self.app.session is None:
                self._send_json(503, {"error": "Computron session not ready yet"})
                return
            with self.app.session_lock:
                self.app.set_state("thinking")
                try:
                    reply, cost = self.app.session.ask(text)
                finally:
                    self.app.set_state("idle")
            turn = {"id": time.time(), "text": text, "reply": reply, "cost": cost, "source": "http"}
            self.app.last_turn = turn
            # Speaking happens outside session_lock — it's just audio
            # playback, doesn't touch the claude subprocess, and holding
            # the lock through TTS would needlessly stall the next ask()
            # (voice or HTTP) until playback finishes.
            if body.get("speak"):
                self.app.set_state("speaking")
                try:
                    speak(reply)
                finally:
                    self.app.set_state("idle")
            self._send_json(200, {"reply": reply, "cost": cost})

        elif self.path == "/editor-state":
            path = (body.get("path") or "").strip()
            line = body.get("line")
            self.app.editor_state = {"path": path, "line": line} if path else None
            self._send_json(200, {"ok": True})

        elif self.path == "/workspace":
            path = (body.get("path") or "").strip()
            if not path:
                self._send_json(400, {"error": "missing 'path'"})
                return
            if self.app.session is None:
                self._send_json(503, {"error": "Computron session not ready yet"})
                return
            with self.app.session_lock:
                self.app.session.set_workspace(path)
            self._send_json(200, {"workspace": path})

        else:
            self._send_json(404, {"error": "not found"})


def start_server(app, port: int) -> ThreadingHTTPServer:
    """Starts the HTTP bridge on a daemon thread and returns the server
    instance (so it can be shut down cleanly on quit)."""
    ComputronRequestHandler.app = app
    httpd = ThreadingHTTPServer(("127.0.0.1", port), ComputronRequestHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print(f"Computron HTTP bridge listening on http://127.0.0.1:{port} (local only)")
    return httpd
