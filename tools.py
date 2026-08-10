"""
Tools Computron can call. Add more functions here as you want Computron to do more —
each one needs: (1) a Python function, (2) a matching schema in TOOL_SCHEMAS,
(3) a dispatch entry in run_tool().
"""
import datetime
import os

VAULT_PATH = os.path.expanduser("~/Obsidian/The Triforce")


def get_current_time() -> str:
    return datetime.datetime.now().strftime("%A, %B %d, %Y — %I:%M %p")


def list_project_notes() -> str:
    """Lists files in the Projects/ folder of the Obsidian vault, if it exists."""
    projects_dir = os.path.join(VAULT_PATH, "Projects")
    if not os.path.isdir(projects_dir):
        return f"Couldn't find {projects_dir}. Check VAULT_PATH in tools.py."
    files = [f for f in os.listdir(projects_dir) if f.endswith(".md")]
    if not files:
        return "No project notes found."
    return "Project notes: " + ", ".join(files)


def read_project_note(filename: str) -> str:
    """Reads a specific markdown file from the Projects/ folder."""
    path = os.path.join(VAULT_PATH, "Projects", filename)
    if not os.path.isfile(path):
        return f"Couldn't find {filename} in Projects/."
    with open(path, "r") as f:
        return f.read()[:3000]  # keep it reasonably short for voice replies


TOOL_SCHEMAS = [
    {
        "name": "get_current_time",
        "description": "Get the current date and time.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_project_notes",
        "description": "List the markdown files in the user's Projects folder (Obsidian vault).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_project_note",
        "description": "Read the contents of a specific project note by filename.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The .md filename, e.g. 'Universal Music Link.md'",
                }
            },
            "required": ["filename"],
        },
    },
]


def run_tool(name: str, tool_input: dict) -> str:
    if name == "get_current_time":
        return get_current_time()
    if name == "list_project_notes":
        return list_project_notes()
    if name == "read_project_note":
        return read_project_note(tool_input.get("filename", ""))
    return f"Unknown tool: {name}"
