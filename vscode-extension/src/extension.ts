import * as vscode from "vscode";

let outputChannel: vscode.OutputChannel;
let statusBarItem: vscode.StatusBarItem;
let lastSeenTurnId: number | undefined;
let pollTimer: ReturnType<typeof setInterval> | undefined;
let editorStateDebounce: ReturnType<typeof setTimeout> | undefined;
let lastReportedEditorState = "";

const STATUS_DISPLAY: Record<string, { icon: string; text: string }> = {
  idle: { icon: "$(mic)", text: "Computron: idle" },
  recording: { icon: "$(record)", text: "Computron: listening" },
  thinking: { icon: "$(sync~spin)", text: "Computron: thinking" },
  speaking: { icon: "$(unmute)", text: "Computron: speaking" },
};
const STATUS_NOT_RUNNING = { icon: "$(circle-slash)", text: "Computron: not running" };

interface Turn {
  id: number;
  text: string;
  reply: string;
  cost: number;
  source: "voice" | "http";
}

function serverBase(): string {
  const port = vscode.workspace.getConfiguration("computron").get<number>("serverPort", 4317);
  return `http://127.0.0.1:${port}`;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${serverBase()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const errBody = await res.json().catch(() => ({}));
    throw new Error((errBody as { error?: string }).error ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

async function pollLastTurn(): Promise<void> {
  try {
    const res = await fetch(`${serverBase()}/last-turn`);
    if (!res.ok) {
      return;
    }
    const turn = (await res.json()) as Partial<Turn>;
    if (!turn.id || turn.id === lastSeenTurnId) {
      return;
    }
    lastSeenTurnId = turn.id;
    // Only mirror voice turns here — HTTP turns (Ask/Ask About Selection)
    // are already printed immediately by ask() below, so mirroring them
    // too would double them up.
    if (turn.source === "voice") {
      outputChannel.appendLine(`You (voice): ${turn.text}`);
      outputChannel.appendLine(`Computron: ${turn.reply}  [+$${(turn.cost ?? 0).toFixed(4)}]\n`);
    }
  } catch {
    // Server not reachable — silently skip this tick, retried on the next poll.
  }
}

async function pollStatus(): Promise<void> {
  try {
    const res = await fetch(`${serverBase()}/status`);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const { state } = (await res.json()) as { ready: boolean; state?: string };
    const display = (state && STATUS_DISPLAY[state]) || STATUS_NOT_RUNNING;
    statusBarItem.text = `${display.icon} ${display.text}`;
  } catch {
    statusBarItem.text = `${STATUS_NOT_RUNNING.icon} ${STATUS_NOT_RUNNING.text}`;
  }
}

async function reportEditorState(): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  const path = editor?.document.uri.scheme === "file" ? editor.document.uri.fsPath : "";
  const line = editor ? editor.selection.active.line + 1 : undefined;
  const key = `${path}:${line}`;
  if (key === lastReportedEditorState) {
    return;
  }
  lastReportedEditorState = key;
  try {
    await postJson("/editor-state", { path, line });
  } catch {
    // Computron not running — harmless, just means voice questions won't
    // be editor-aware until it's back and the next change re-reports.
  }
}

function scheduleEditorStateReport(): void {
  // Debounced — onDidChangeTextEditorSelection fires on every cursor move
  // (including as a byproduct of typing), so reporting on every event
  // would spam the bridge. 500ms feels responsive without being chatty.
  clearTimeout(editorStateDebounce);
  editorStateDebounce = setTimeout(() => void reportEditorState(), 500);
}

async function attachCurrentWorkspace(): Promise<void> {
  const folder = vscode.workspace.workspaceFolders?.[0];
  if (!folder) {
    return;
  }
  try {
    await postJson("/workspace", { path: folder.uri.fsPath });
  } catch (err) {
    // Computron not running yet — not fatal, just means workspace attaches
    // on the next successful Ask instead. Surfaced quietly via the output
    // channel rather than an intrusive error popup.
    outputChannel.appendLine(`[Computron] Couldn't attach workspace yet: ${(err as Error).message}`);
  }
}

async function ask(displayText: string, sendText: string = displayText): Promise<void> {
  outputChannel.show(true);
  outputChannel.appendLine(`You: ${displayText}`);
  const speakReplies = vscode.workspace.getConfiguration("computron").get<boolean>("speakReplies", true);
  try {
    const { reply, cost } = await postJson<{ reply: string; cost: number }>("/ask", {
      text: sendText,
      speak: speakReplies,
    });
    outputChannel.appendLine(`Computron: ${reply}  [+$${cost.toFixed(4)}]\n`);
  } catch (err) {
    const message = (err as Error).message;
    outputChannel.appendLine(`[Error] ${message}\n`);
    vscode.window.showErrorMessage(
      `Computron: ${message}. Is the menu bar app running? (Computron/menubar_app.py)`
    );
  }
}

export function activate(context: vscode.ExtensionContext) {
  outputChannel = vscode.window.createOutputChannel("Computron");
  context.subscriptions.push(outputChannel);

  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
  statusBarItem.command = "computron.ask";
  statusBarItem.text = `${STATUS_NOT_RUNNING.icon} ${STATUS_NOT_RUNNING.text}`;
  statusBarItem.show();
  context.subscriptions.push(statusBarItem);

  void attachCurrentWorkspace();
  context.subscriptions.push(
    vscode.workspace.onDidChangeWorkspaceFolders(() => void attachCurrentWorkspace())
  );

  void reportEditorState();
  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(() => void reportEditorState()),
    vscode.window.onDidChangeTextEditorSelection(() => scheduleEditorStateReport())
  );

  // Polls continuously for the extension's lifetime rather than gating on
  // output-channel visibility — there's no public VS Code API for that, and
  // a local GET every 1.5s is cheap enough not to matter.
  pollTimer = setInterval(() => {
    void pollLastTurn();
    void pollStatus();
  }, 1500);
  void pollStatus();
  context.subscriptions.push({ dispose: () => clearInterval(pollTimer) });

  context.subscriptions.push(
    vscode.commands.registerCommand("computron.ask", async () => {
      const text = await vscode.window.showInputBox({
        prompt: "Ask Computron",
        placeHolder: "What do you want to know?",
      });
      if (!text) {
        return;
      }
      // Attach the active file (path + cursor line) as context when there
      // is one, so "what does line 67 do" resolves without making Jorge
      // name the file every time — Computron can Read the file itself once
      // it knows the path. Falls back to the bare question with no editor.
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        await ask(text);
        return;
      }
      const composed =
        `Jorge's active file in VS Code: ${editor.document.uri.fsPath}, cursor on line ` +
        `${editor.selection.active.line + 1}.\nJorge asks: ${text}`;
      await ask(text, composed);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("computron.askAboutSelection", async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage("Computron: no active editor to select from.");
        return;
      }
      const selection = editor.selection;
      const selectedText = editor.document.getText(selection);
      if (!selectedText) {
        vscode.window.showErrorMessage("Computron: no text selected.");
        return;
      }
      const question = await vscode.window.showInputBox({
        prompt: "Ask Computron about this selection",
        placeHolder: "e.g. Explain what this does, or Why might this fail?",
      });
      if (!question) {
        return;
      }
      const filePath = editor.document.uri.fsPath;
      const startLine = selection.start.line + 1;
      const endLine = selection.end.line + 1;
      const composed =
        `Regarding ${filePath} lines ${startLine}-${endLine}:\n` +
        "```\n" + selectedText + "\n```\n" +
        `Jorge asks: ${question}`;
      await ask(question, composed);
    })
  );
}

export function deactivate() {}
