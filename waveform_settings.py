"""Live-editable waveform visualizer settings (waveform_window.py), stored
in waveform_settings.json next to this file — seeded from config.WAVEFORM_*
(the .env defaults) the first time it's created, same pattern as
claude_code_backend.py's personality.json seeding from config.ELEVENLABS_SPEED.

Read fresh from disk on every waveform_window.show_and_animate() call, not
cached at import, so an edit — from the menu bar's "Waveform Settings"
submenu or this module's own CLI (`python waveform_settings.py --help`) —
takes effect on the very next spoken reply. No restart, and no cross-process
polling needed to notice a CLI edit made while the menu bar app is already
running: the running app simply re-reads this file the next time it's about
to show the window.
"""
import argparse
import json
import os
import sys

import config

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(PROJECT_DIR, "waveform_settings.json")

POSITIONS = (
    "bottom-center", "bottom-left", "bottom-right",
    "top-center", "top-left", "top-right", "center",
)

STYLES = ("bars", "line", "orb", "cat")

DEFAULT_SETTINGS = {
    "style": config.WAVEFORM_STYLE,
    "color": config.WAVEFORM_COLOR,
    "opacity": config.WAVEFORM_OPACITY,
    "width": config.WAVEFORM_WIDTH,
    "height": config.WAVEFORM_HEIGHT,
    "bars": config.WAVEFORM_BARS,
    "bar_gap": config.WAVEFORM_BAR_GAP,
    "position": config.WAVEFORM_POSITION,
    "margin": config.WAVEFORM_MARGIN,
    # Off by default — the window is fully transparent otherwise, same as
    # before this existed. Opt-in backing panel so bar colors don't
    # disappear against a similarly-colored desktop/window behind them.
    "bg_enabled": False,
    "bg_color": "000000",
    "bg_opacity": 0.5,
}

# Curated color themes — one click/flag instead of typing hex. Presets only
# ever touch the bar's color + opacity, applied on top of whatever else is
# already set (size, bar count, gap, position, margin, background) — they're
# a palette choice, not a full-layout overwrite, so picking one never moves
# or resizes the window.
PRESETS = {
    "Classic": {"color": "FFFFFF", "opacity": 0.85},
    "Cyan Pulse": {"color": "00CFFF", "opacity": 0.9},
    "Retro Green": {"color": "33FF33", "opacity": 0.95},
    "Party": {"color": "FF00AA", "opacity": 0.9},
    "Sunset": {"color": "FF7A00", "opacity": 0.9},
    "Ice": {"color": "A0E9FF", "opacity": 0.85},
}


def parse_hex_color(hex_str: str) -> tuple:
    """Parses a 3- or 6-digit hex RGB string (leading '#' optional) into
    0-1 floats. Raises ValueError (with a message safe to show a user
    as-is) on anything malformed, rather than silently falling back to a
    color nobody asked for."""
    s = hex_str.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"'{hex_str}' isn't a valid hex color (expected 3 or 6 hex digits).")
    try:
        return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        raise ValueError(f"'{hex_str}' isn't a valid hex color (expected 3 or 6 hex digits).")


def _validate(settings: dict) -> None:
    if settings["style"] not in STYLES:
        raise ValueError(f"style must be one of: {', '.join(STYLES)}")
    parse_hex_color(settings["color"])  # raises on a bad color
    if not (0.0 <= settings["opacity"] <= 1.0):
        raise ValueError("opacity must be between 0 and 1.")
    if settings["width"] <= 0 or settings["height"] <= 0:
        raise ValueError("width/height must be positive.")
    if settings["bars"] < 1:
        raise ValueError("bars must be at least 1.")
    if settings["bar_gap"] < 0:
        raise ValueError("bar_gap must be non-negative.")
    if settings["position"] not in POSITIONS:
        raise ValueError(f"position must be one of: {', '.join(POSITIONS)}")
    if not isinstance(settings["bg_enabled"], bool):
        raise ValueError("bg_enabled must be true or false.")
    parse_hex_color(settings["bg_color"])  # raises on a bad color
    if not (0.0 <= settings["bg_opacity"] <= 1.0):
        raise ValueError("bg_opacity must be between 0 and 1.")
    if settings["margin"] < 0:
        raise ValueError("margin must be non-negative.")


def read() -> dict:
    try:
        with open(SETTINGS_FILE) as f:
            values = json.load(f)
        return {**DEFAULT_SETTINGS, **values}
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(DEFAULT_SETTINGS)


def _write(settings: dict) -> None:
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


if not os.path.exists(SETTINGS_FILE):
    _write(DEFAULT_SETTINGS)

# Styles whose art doesn't suit the generic bars-sized default window.
# "cat"'s bundled frames are portrait (800x984, ratio ~0.81) — at the
# generic 220x60 it'd be letterboxed down to a sliver, so switching to it
# bumps the window to fit instead of leaving it looking broken by default.
STYLE_DEFAULT_SIZES = {
    "cat": (200, 250),
}


def update(**changes) -> dict:
    """Merges changes into the current settings, validates the result, and
    writes it back — nothing is written if validation fails. Raises
    ValueError on an unknown key or an invalid value. Shared by the menu
    bar dialogs and the CLI below, so both go through the same validation."""
    current = read()
    for key, value in changes.items():
        if key not in DEFAULT_SETTINGS:
            raise ValueError(f"Unknown setting '{key}'.")
        current[key] = value
    # Switching TO a style with its own recommended size applies it —
    # unless this same call also set width/height explicitly, which always
    # wins over the implied default.
    if "style" in changes and changes["style"] in STYLE_DEFAULT_SIZES and not ({"width", "height"} & changes.keys()):
        current["width"], current["height"] = STYLE_DEFAULT_SIZES[changes["style"]]
    _validate(current)
    _write(current)
    return current


def reset() -> dict:
    _write(DEFAULT_SETTINGS)
    return dict(DEFAULT_SETTINGS)


def apply_preset(name: str) -> dict:
    """Applies a preset's color + opacity on top of whatever else is
    currently set — see the PRESETS comment above for why this is a patch
    through update(), not a full overwrite. Raises ValueError on an
    unknown name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Choices: {', '.join(PRESETS)}")
    return update(**PRESETS[name])


# --- terminal CLI: `python waveform_settings.py --preset "Cyan Pulse"` or
# `python waveform_settings.py --color 00CFFF --position top-right` ---

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Edit Computron's floating waveform visualizer settings. "
                     "Takes effect the next time Computron speaks -- no restart needed."
    )
    parser.add_argument("--preset", choices=list(PRESETS), help="Apply a preloaded look (see --list-presets).")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit.")
    parser.add_argument("--style", choices=STYLES, help="Visualizer style.")
    parser.add_argument("--color", help="Hex RGB, e.g. 00CFFF (leading # optional).")
    parser.add_argument("--opacity", type=float, help="0-1.")
    parser.add_argument("--width", type=int, help="Window width in px.")
    parser.add_argument("--height", type=int, help="Window height in px.")
    parser.add_argument("--bars", type=int, help="Number of bars.")
    parser.add_argument("--bar-gap", type=float, dest="bar_gap", help="Pixels between bars.")
    parser.add_argument("--position", choices=POSITIONS)
    parser.add_argument("--margin", type=int, help="Pixels from the chosen screen edge(s).")
    parser.add_argument(
        "--background", choices=("on", "off"),
        help="Solid backing panel behind the bars, so their color doesn't get lost against whatever's behind the window.",
    )
    parser.add_argument("--bg-color", dest="bg_color", help="Hex RGB for the background panel, e.g. 000000.")
    parser.add_argument("--bg-opacity", type=float, dest="bg_opacity", help="0-1.")
    parser.add_argument("--reset", action="store_true", help="Restore all settings to their .env defaults.")
    parser.add_argument("--show", action="store_true", help="Print current settings and exit.")
    args = parser.parse_args()

    if args.list_presets:
        for name, settings in PRESETS.items():
            print(f"{name}: {settings}")
        return

    if args.reset:
        reset()
        print("Waveform settings reset to .env defaults.")
        return

    # A preset can be combined with explicit field flags in the same
    # invocation (preset applied first, then overrides layered on top) —
    # e.g. `--preset "Cyan Pulse" --bars 12` for "that preset, but more bars."
    if args.preset:
        try:
            apply_preset(args.preset)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Applied preset '{args.preset}'.")

    changes = {
        key: value for key, value in vars(args).items()
        if key not in ("reset", "show", "preset", "list_presets", "background") and value is not None
    }
    if args.background is not None:
        changes["bg_enabled"] = args.background == "on"
    if changes:
        try:
            update(**changes)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    for key, value in read().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    _cli()
