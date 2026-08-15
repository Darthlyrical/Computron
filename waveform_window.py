"""
Floating waveform visualizer — a small borderless, always-on-top window
showing a live-looking animation while Computron talks. Four styles
(waveform_settings.py's "style" field): "bars" (spectrum equalizer bars,
the default), "line" (a filled amplitude ribbon that scrolls left), "orb"
(a Siri-style pulsing circle), and "cat" (a 2-frame Pop Cat mouth flip —
assets/cat/, chroma-keyed transparent, see _load_cat_image). All four
animate off the exact same precomputed data — see compute_fft_envelope
below.

Not truly live: main.py's actual playback (_play(), afplay) exposes no
access to audio samples as they're played, so instead the whole clip's
frequency envelope is precomputed upfront (a few milliseconds — benchmarked
at ~5.6ms for a 6-second clip, negligible next to the multi-second LLM+TTS
latency already present) and the window animates on a timer synced to
elapsed wall-clock time since playback started. Good enough for a
decorative visual; not attempting sample-exact sync with afplay.

Look and placement (style, color, opacity, size, bar count/spacing, screen
position/margin, background) live in waveform_settings.py — editable from
the menu bar's "Waveform Settings" submenu or from a terminal via
`python waveform_settings.py`. Read fresh from disk on every
show_and_animate() call (not cached), so an edit from either surface takes
effect on the very next spoken reply — no restart needed.

AppKit-only. Every function here must run on the main thread — the sole
entry point (on_playback_change) is only ever called by menubar_app.py via
AppHelper.callAfter, so nothing in this module does its own thread
dispatch. Never imported by main.py, which stays usable standalone in a
plain terminal with no GUI event loop at all.

A real quirk hit building this: a plain NSTimer (even matching rumps' own
alloc/init/addTimer_forMode_ pattern exactly) only fired once under
rumps.App.run()'s AppHelper.runEventLoop(), reproducibly, both sandboxed
and not — but a manually-pumped NSRunLoop fired it 16/16 times on
schedule. Root cause not fully chased down (likely something specific to
how AppHelper's loop wrapper services NSDefaultRunLoopMode timers outside
a real interactive WindowServer session, which automated test runs don't
have). Sidestepped by using rumps.Timer itself — rumps' own supported,
widely-used API for periodic callbacks — rather than hand-rolling NSTimer
scheduling a second time.
"""
import os
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import objc
import rumps
from AppKit import (
    NSBackingStoreBuffered, NSBezierPath, NSColor, NSCompositingOperationCopy,
    NSCompositingOperationSourceOver, NSFloatingWindowLevel, NSImage,
    NSRectFillUsingOperation, NSScreen, NSView, NSWindow,
    NSWindowStyleMaskBorderless,
)
from Foundation import NSMakePoint, NSMakeRect

import waveform_settings

# "cat" style assets: a chroma-keyed, transparent-background 2-frame Pop
# Cat flip (mouth_closed.png / mouth_open.png), both cropped to the same
# bounding box so swapping frames doesn't jitter or resize. See
# assets/cat/ — prepped once with NSBitmapImageRep + numpy (no Pillow
# dependency needed for the one-time chroma-key/crop pass).
_CAT_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "cat")
_cat_images: dict = {}


def _load_cat_image(name: str) -> Optional[NSImage]:
    if name not in _cat_images:
        path = os.path.join(_CAT_ASSETS_DIR, f"{name}.png")
        image = NSImage.alloc().initWithContentsOfFile_(path) if os.path.exists(path) else None
        _cat_images[name] = image
    return _cat_images[name]


# Hysteresis for the cat's mouth-open/closed toggle: needs to rise past
# OPEN to open, but fall below the lower CLOSE to close again, so audio
# hovering right at one single threshold doesn't flicker the mouth rapidly.
CAT_OPEN_THRESHOLD = 0.15
CAT_CLOSE_THRESHOLD = 0.08
# The cat's own attack/decay, faster than the shared ATTACK/DECAY below —
# a continuous bar/orb benefits from smoothing so it doesn't look jittery,
# but a 2-frame mouth flip needs the opposite: snapping back down quickly
# between syllables is what makes it flap rather than hold open through an
# entire phrase.
CAT_ATTACK = 0.85
CAT_DECAY = 0.85

FPS = 30
# Voice energy lives almost entirely below this — no point spending bars
# on frequencies speech barely touches. Not exposed as a setting like the
# purely visual knobs — this is about what the FFT actually measures.
MIN_FREQ_HZ = 60
MAX_FREQ_HZ = 8000

# VU-meter-style attack/decay: each tick, a displayed level moves only this
# fraction of the way toward that frame's target, rather than snapping
# straight to it. Rising fast (ATTACK) and falling slower (DECAY) is what
# makes the animation read as physical rather than stepped — the classic
# VU-meter needle behavior. Shared by all three styles. Tuned by eye for
# FPS=30; not exposed as a setting since it's about animation feel, not a
# look someone would want to reconfigure.
ATTACK = 0.6
DECAY = 0.15


def _window_origin(win_w: float, win_h: float, settings: dict) -> tuple:
    """Resolves settings["position"] + settings["margin"] into a
    bottom-left (x, y) origin on the main screen."""
    screen = NSScreen.mainScreen().frame()
    sw, sh = screen.size.width, screen.size.height
    margin = settings["margin"]
    positions = {
        "bottom-center": ((sw - win_w) / 2, margin),
        "bottom-left": (margin, margin),
        "bottom-right": (sw - win_w - margin, margin),
        "top-center": ((sw - win_w) / 2, sh - win_h - margin),
        "top-left": (margin, sh - win_h - margin),
        "top-right": (sw - win_w - margin, sh - win_h - margin),
        "center": ((sw - win_w) / 2, (sh - win_h) / 2),
    }
    # settings comes from waveform_settings.read(), which only ever returns
    # already-validated values (update()/_validate() reject a bad position
    # before it's ever written) — no fallback branch needed here.
    return positions[settings["position"]]


def compute_fft_envelope(wav_path: Path, num_bars: int, fps: int = FPS) -> np.ndarray:
    """Sweeps the whole clip into a [num_frames, num_bars] array of
    0..1-normalized magnitudes, log-spaced across frequency so voice
    (bass/mid-heavy) doesn't leave the upper bins looking dead. Shared by
    all three styles: "bars" uses a frame's bins directly as per-bar
    heights, "line"/"orb" collapse a frame to its mean as a single overall
    loudness value — same underlying data either way, no separate
    amplitude-only pass needed."""
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError(f"Expected 16-bit PCM wav, got {sampwidth * 8}-bit")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    hop = max(1, sr // fps)
    window_size = max(256, hop * 2)
    freqs = np.fft.rfftfreq(window_size, d=1 / sr)
    lo = max(MIN_FREQ_HZ, freqs[1] if len(freqs) > 1 else MIN_FREQ_HZ)
    hi = min(MAX_FREQ_HZ, freqs[-1])
    edges = np.geomspace(lo, hi, num_bars + 1)
    hann = np.hanning(window_size)

    frames = []
    for start in range(0, max(1, len(audio) - window_size), hop):
        chunk = audio[start:start + window_size]
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)))
        spectrum = np.abs(np.fft.rfft(chunk * hann))
        bars = np.zeros(num_bars, dtype=np.float32)
        for i in range(num_bars):
            mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
            if mask.any():
                bars[i] = spectrum[mask].mean()
        frames.append(bars)

    if not frames:
        return np.zeros((1, num_bars), dtype=np.float32)
    arr = np.array(frames)
    # Raw FFT magnitude for speech is mostly near-silent with occasional
    # spikes — looks "dead" as a linear bar height. Log compression spreads
    # that into something that actually moves.
    arr = np.log1p(arr)
    peak = arr.max()
    if peak > 0:
        arr = arr / peak
    return arr


class _VisualizerView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(_VisualizerView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._style = "bars"
        self.bar_heights = [0.0]   # "bars" style: one smoothed level per bar
        self.level = 0.0           # "line"/"orb"/"cat" style: one smoothed overall level
        self.history = []          # "line" style only: scrolling buffer of past levels
        self.cat_mouth_open = False  # "cat" style only: hysteresis-driven frame toggle
        self._color = (1.0, 1.0, 1.0)
        self._opacity = 0.85
        self._gap = 4.0
        self._bg_enabled = False
        self._bg_color = (0.0, 0.0, 0.0)
        self._bg_opacity = 0.5
        return self

    def apply_waveform_settings(self, settings: dict) -> None:
        """Applies a freshly-read settings dict — called once right after
        construction, before the view is ever shown."""
        self._style = settings["style"]
        self.bar_heights = [0.0] * settings["bars"]
        self.level = 0.0
        self.cat_mouth_open = False
        # One history sample roughly every 3px of window width, so the
        # scroll density scales with the window instead of a fixed sample
        # count looking sparse in a wide window or cramped in a narrow one.
        history_len = max(20, int(settings["width"] / 3))
        self.history = [0.0] * history_len
        self._color = waveform_settings.parse_hex_color(settings["color"])
        self._opacity = settings["opacity"]
        self._gap = settings["bar_gap"]
        self._bg_enabled = settings["bg_enabled"]
        self._bg_color = waveform_settings.parse_hex_color(settings["bg_color"])
        self._bg_opacity = settings["bg_opacity"]

    def drawRect_(self, rect):
        NSColor.clearColor().set()
        NSRectFillUsingOperation(rect, NSCompositingOperationCopy)

        bounds = self.bounds()
        w, h = bounds.size.width, bounds.size.height

        # Optional backing panel, drawn before the visualizer itself, so a
        # bright/dark foreground color doesn't disappear against a
        # similarly-colored desktop or window behind an otherwise fully
        # transparent visualizer. Rounded to match the bars/orb's own
        # rounded look rather than a hard-edged rectangle.
        if self._bg_enabled:
            bg_r, bg_g, bg_b = self._bg_color
            NSColor.colorWithCalibratedRed_green_blue_alpha_(bg_r, bg_g, bg_b, self._bg_opacity).set()
            radius = min(12.0, w / 2, h / 2)
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(rect, radius, radius).fill()

        r, g, b = self._color
        color = NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, self._opacity)

        if self._style == "line":
            self._draw_line(w, h, color)
        elif self._style == "orb":
            self._draw_orb(w, h, color)
        elif self._style == "cat":
            self._draw_cat(w, h)
        else:
            self._draw_bars(w, h, color)

    def _draw_bars(self, w: float, h: float, color) -> None:
        n = len(self.bar_heights)
        gap = self._gap
        bar_width = (w - gap * (n + 1)) / n
        color.set()
        for i, level in enumerate(self.bar_heights):
            bar_h = max(4.0, level * (h - 8))
            x = gap + i * (bar_width + gap)
            y = (h - bar_h) / 2
            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(x, y, bar_width, bar_h), bar_width / 2, bar_width / 2
            )
            path.fill()

    def _draw_line(self, w: float, h: float, color) -> None:
        """A filled amplitude ribbon: the level history (unsigned FFT
        magnitude, not true bipolar audio) mirrored above and below the
        vertical center, giving the familiar "sound wave" look voice-memo
        apps use for amplitude-only data rather than a literal signed
        oscilloscope trace."""
        n = len(self.history)
        if n < 2:
            return
        mid = h / 2
        amp = (h - 8) / 2
        step = w / (n - 1)
        path = NSBezierPath.bezierPath()
        path.moveToPoint_(NSMakePoint(0, mid + self.history[0] * amp))
        for i in range(1, n):
            path.lineToPoint_(NSMakePoint(i * step, mid + self.history[i] * amp))
        for i in range(n - 1, -1, -1):
            path.lineToPoint_(NSMakePoint(i * step, mid - self.history[i] * amp))
        path.closePath()
        color.set()
        path.fill()

    def _draw_orb(self, w: float, h: float, color) -> None:
        """A filled circle plus a larger, more transparent glow ring behind
        it, both sized off the same smoothed overall level — grows on
        louder audio, shrinks back on quiet stretches."""
        cx, cy = w / 2, h / 2
        base_r = min(w, h) * 0.12
        max_r = min(w, h) * 0.45
        radius = base_r + self.level * (max_r - base_r)
        r, g, b = self._color
        NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, b, self._opacity * 0.35).set()
        NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(cx - radius * 1.4, cy - radius * 1.4, radius * 2.8, radius * 2.8)
        ).fill()
        color.set()
        NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(cx - radius, cy - radius, radius * 2, radius * 2)
        ).fill()

    def _draw_cat(self, w: float, h: float) -> None:
        """Draws whichever Pop Cat frame cat_mouth_open (set in _tick, off
        the same smoothed level orb/line use, with hysteresis) currently
        selects — scaled to fit the window, aspect ratio preserved,
        centered. self._color is ignored here (recoloring photographic
        cat pixels doesn't make sense); self._opacity still applies, as
        the image's overall draw alpha."""
        image = _load_cat_image("mouth_open" if self.cat_mouth_open else "mouth_closed")
        if image is None:
            return
        size = image.size()
        if size.width <= 0 or size.height <= 0:
            return
        scale = min(w / size.width, h / size.height)
        draw_w, draw_h = size.width * scale, size.height * scale
        x, y = (w - draw_w) / 2, (h - draw_h) / 2
        image.drawInRect_fromRect_operation_fraction_(
            NSMakeRect(x, y, draw_w, draw_h),
            NSMakeRect(0, 0, 0, 0),  # empty source rect means "the whole image"
            NSCompositingOperationSourceOver,
            self._opacity,
        )

    def isFlipped(self):
        return False


_window: Optional[NSWindow] = None
_view: Optional[_VisualizerView] = None
_animation_timer: Optional[rumps.Timer] = None
_envelope: Optional[np.ndarray] = None
_playback_start: Optional[float] = None


def _teardown_window() -> None:
    global _window, _view
    if _window is not None:
        _window.orderOut_(None)
    _window = None
    _view = None


def _build_window(settings: dict) -> None:
    global _window, _view
    win_w, win_h = settings["width"], settings["height"]
    x, y = _window_origin(win_w, win_h, settings)
    frame = NSMakeRect(x, y, win_w, win_h)
    _window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
    )
    _window.setOpaque_(False)
    _window.setBackgroundColor_(NSColor.clearColor())
    _window.setLevel_(NSFloatingWindowLevel)
    _window.setIgnoresMouseEvents_(True)
    _window.setHasShadow_(False)
    _window.setCollectionBehavior_(1 << 8)  # NSWindowCollectionBehaviorCanJoinAllSpaces
    _view = _VisualizerView.alloc().initWithFrame_(NSMakeRect(0, 0, win_w, win_h))
    _view.apply_waveform_settings(settings)
    _window.setContentView_(_view)


def _tick(_timer) -> None:
    if _envelope is None or _playback_start is None or _view is None:
        return
    elapsed = time.time() - _playback_start
    idx = int(elapsed * FPS)
    frame = np.zeros(_envelope.shape[1]) if idx >= len(_envelope) else _envelope[idx]

    if _view._style == "bars":
        target = frame.tolist()
        _view.bar_heights = [
            current + (t - current) * (ATTACK if t > current else DECAY)
            for current, t in zip(_view.bar_heights, target)
        ]
    else:
        # "line", "orb", and "cat" all animate off one overall loudness
        # value per frame — the mean across the same frequency bins "bars"
        # uses, rather than a second amplitude computation.
        target = float(frame.mean()) if len(frame) else 0.0
        current = _view.level
        if _view._style == "cat":
            rate = CAT_ATTACK if target > current else CAT_DECAY
        else:
            rate = ATTACK if target > current else DECAY
        _view.level = current + (target - current) * rate
        if _view._style == "line":
            _view.history.append(_view.level)
            _view.history.pop(0)
        elif _view._style == "cat":
            if _view.cat_mouth_open and _view.level < CAT_CLOSE_THRESHOLD:
                _view.cat_mouth_open = False
            elif not _view.cat_mouth_open and _view.level > CAT_OPEN_THRESHOLD:
                _view.cat_mouth_open = True

    _view.setNeedsDisplay_(True)


def show_and_animate(wav_path: Path) -> None:
    """Precomputes the FFT envelope and starts the window animating in
    sync with playback. Rebuilds the window fresh from waveform_settings on
    every call — cheap (a plain NSWindow) — so any edit made since the last
    turn, whether from the menu bar or the CLI, shows up immediately rather
    than needing a restart or a background file-watcher. Must run on the
    main thread — see module docstring."""
    global _envelope, _playback_start, _animation_timer
    settings = waveform_settings.read()
    try:
        envelope = compute_fft_envelope(wav_path, num_bars=settings["bars"])
    except (FileNotFoundError, wave.Error, ValueError):
        return
    _teardown_window()
    _build_window(settings)
    _envelope = envelope
    _playback_start = time.time()
    _window.orderFront_(None)
    if _animation_timer is None:
        _animation_timer = rumps.Timer(_tick, 1.0 / FPS)
    if not _animation_timer.is_alive():
        _animation_timer.start()


def hide() -> None:
    """Stops the animation and tears down the window. Must run on the main
    thread — see module docstring."""
    global _envelope, _playback_start
    if _animation_timer is not None and _animation_timer.is_alive():
        _animation_timer.stop()
    _envelope = None
    _playback_start = None
    _teardown_window()


def on_playback_change(path: Optional[Path]) -> None:
    """Registered (via menubar_app.py) as main.py's playback listener.
    path is the audio file that just started playing, or None when
    playback stopped — naturally finishing or interrupted by barge-in."""
    if path is None:
        hide()
    else:
        show_and_animate(path)
