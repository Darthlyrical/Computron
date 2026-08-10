# Computron — Claude Code-powered voice assistant

Talk to it, it talks back. Speech-to-text and text-to-speech run **locally and
free**. The "thinking" is a persistent `claude` (Claude Code) process kept
alive for the whole run — not a fresh API call per turn — so it authenticates
via your Claude Pro login and stays warm in the prompt cache instead of
reloading its system prompt and tools every exchange. It reads files
(including your Obsidian vault) and searches the web freely; it can also
write/edit files, but never without asking first out loud and getting a
clear "yes" — see "Write access" below for how that actually works. No Bash
ever, under any circumstances.

## What's in here

| File | Purpose |
|---|---|
| `main.py` | The voice loop: record → transcribe → ask Claude Code → speak |
| `claude_code_backend.py` | Spawns and talks to the persistent `claude` process (stream-json in/out) |
| `tools.py` | Not currently used — leftover from an earlier raw-API design. Kept in case that mode comes back. |
| `config.py` | Loads settings from `.env` (model choice, Piper paths) |
| `.env.example` | Template `.env` — `ANTHROPIC_API_KEY` is optional here, unused by the current backend |

## One-time setup (macOS, Apple Silicon)

### 1. Python deps
```bash
cd Computron
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Audio input library
```bash
brew install portaudio
```

### 3. Piper (local text-to-speech)

The `rhasspy/piper` GitHub release binary is stale (last shipped Nov 2023) and
the `piper_macos_aarch64.tar.gz` asset is actually broken — it's an x86_64
binary mislabeled as arm64, missing its required dylibs. Use the maintained
pip package instead, installed straight into the venv:

```bash
pip install piper-tts

# Download a voice model (this one's a good default — natural, medium size)
curl -L -o voices/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -L -o voices/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

`PIPER_BIN` in `.env` should just be `piper` — it resolves to the venv's
`piper` console script as long as the venv is activated.

### 4. Log into Claude Code (if you haven't already)
Computron authenticates through whatever `claude` is logged into on this machine —
run `claude` once interactively and log in if you haven't. No API key needed;
`.env`'s `ANTHROPIC_API_KEY` is unused by this backend (see `.env.example`).

### 5. Run it
```bash
python main.py
```
Or, from any terminal: `computron` (a small launcher script at
`~/.local/bin/computron`, already on PATH, that `cd`s in, activates the venv,
and runs this for you). "Computron" — an Office "The Banker" callback — is
both the spoken persona and the project's name.

Press Enter, talk, press Enter again to stop recording. Computron transcribes, thinks, and speaks back.

## Write access — confirm-before-acting, not a permission prompt

A voice interface can't show you Claude Code's normal "approve this edit?"
dialog — there's no UI to render it into. Instead, `claude_code_backend.py`
implements confirmation as an actual two-turn flow, not just a prompt asking
Claude to be polite about it:

1. Every turn defaults to read-only tools (`Read`, `Glob`, `Grep`,
   `WebSearch` — never `Edit`/`Write`/`Bash`).
2. If a turn tries to write something, the CLI denies it and tells Claude
   why. Claude — per its system prompt — explains what it wanted to do and
   asks you to confirm, out loud.
3. If your *very next* utterance is an unambiguous "yes" (a short list of
   exact phrases in `AFFIRMATIVE_PHRASES` — not "maybe" or a new unrelated
   question), that one turn only gets respawned with `Edit`/`Write` enabled
   so Claude can retry and actually make the change. It drops straight back
   to read-only immediately after, whether or not the write succeeded.

This means the enforcement is real, not just a suggestion in the prompt: the
CLI's own tool-permission system is the gate, and a vague or unrelated reply
never grants anything. `Bash` is never in the allowed list under any
circumstances — no confirmation flow unlocks it.

**One real gap found and closed:** Claude Code has a built-in "auto-memory"
feature that writes to its local memory bank through an internal mechanism
that bypasses `--allowedTools` entirely — confirmed live when Computron said
"done, saved to memory" for something it was never actually granted
permission to write. `claude_code_backend.py` sets
`CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` in the subprocess environment to close
this — verified empirically (file-count before/after, direct content
search) that it stops the silent write without affecting memory-bank
*reading*, which still works exactly as before.

Ideas worth asking it given your projects — no code changes needed, since it
can already read the vault and (with confirmation) write to it:
- "What's the status of PulseTag?"
- "What am I supposed to be working on today?" (reads `Daily Structure.md`)
- "Summarize my last journal entry"
- "Add a note to my next-steps list about X" (will ask to confirm first)

## Cost reality check

Computron's "thinking" runs through your Claude Pro subscription's included Claude
Code usage — not separate API billing — since `claude_code_backend.py`
explicitly strips `ANTHROPIC_API_KEY` from the subprocess environment so it
falls back to your Pro login. That means no extra charge unless you exceed
Pro's included usage, at which point overage billing kicks in. One real
consequence: Computron shares the same usage pool and rate-limit window as your
interactive Claude Code sessions — heavy Computron use competes with actual coding
work. If that becomes a problem, switch back to a real API key (add it back
to `.env` and stop stripping it in `claude_code_backend.py`) to isolate
Computron's billing and usage from your coding sessions.

## Conversation memory across launches

Computron remembers the actual conversation, not just facts from the vault.
`claude_code_backend.py` stores the Claude Code session ID in `.session_id`
after every turn and resumes it automatically on the next `computron`
launch — quitting and relaunching picks up where you left off instead of
starting cold. If that stored ID is ever stale (deleted, expired), Computron detects the
failure on the first turn and silently falls back to a fresh conversation —
you'll see a note printed in the terminal, but the answer itself won't be an
error message. To force a fresh start deliberately, just delete
`.session_id`.

## Known rough edges (v1)

- Push-to-talk only (press Enter) — no wake word or always-listening yet
- Piper's voice is clear but not super natural — swap in ElevenLabs later if you want more polish (costs a bit)
- `faster-whisper` uses the `small.en` model by default — good accuracy/speed balance on Apple Silicon CPU; bump to `medium.en` in `main.py` if you want better accuracy and don't mind slower transcription
- No running cost total for the session — only per-turn cost is shown
