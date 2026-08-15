"""
Floating waveform visualizer — a small borderless, always-on-top window
showing a live-looking spectrum-bar animation while Computron talks.

Not truly live: main.py's actual playback (_play(), afplay) exposes no
access to audio samples as they're played, so instead the whole clip's
frequency envelope is precomputed upfront (a few milliseconds — benchmarked
at ~5.6ms for a 6-second clip, negligible next to the multi-second LLM+TTS
latency already present) and the window animates on a timer synced to
elapsed wall-clock time since playback started. Good enough for a
decorative visual; not attempting sample-exact sync with afplay.

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
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import objc
import rumps
from AppKit import (
    NSBackingStoreBuffered, NSBezierPath, NSColor, NSCompositingOperationCopy,
    NSFloatingWindowLevel, NSRectFillUsingOperation, NSScreen, NSView,
    NSWindow, NSWindowStyleMaskBorderless,
)
from Foundation import NSMakeRect

NUM_BARS = 9
FPS = 30
WINDOW_WIDTH = 220
WINDOW_HEIGHT = 60
BAR_GAP = 4.0
BOTTOM_MARGIN = 80  # px above the screen's bottom edge
# Voice energy lives almost entirely below this — no point spending bars
# on frequencies speech barely touches.
MIN_FREQ_HZ = 60
MAX_FREQ_HZ = 8000


def compute_fft_envelope(wav_path: Path, num_bars: int = NUM_BARS, fps: int = FPS) -> np.ndarray:
    """Sweeps the whole clip into a [num_frames, num_bars] array of
    0..1-normalized bar heights, log-spaced across frequency so voice
    (bass/mid-heavy) doesn't leave the upper bars looking dead."""
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


class _BarsView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(_BarsView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.bar_heights = [0.0] * NUM_BARS
        return self

    def drawRect_(self, rect):
        NSColor.clearColor().set()
        NSRectFillUsingOperation(rect, NSCompositingOperationCopy)

        bounds = self.bounds()
        w, h = bounds.size.width, bounds.size.height
        n = len(self.bar_heights)
        bar_width = (w - BAR_GAP * (n + 1)) / n
        NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.85).set()
        for i, level in enumerate(self.bar_heights):
            bar_h = max(4.0, level * (h - 8))
            x = BAR_GAP + i * (bar_width + BAR_GAP)
            y = (h - bar_h) / 2
            path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(x, y, bar_width, bar_h), bar_width / 2, bar_width / 2
            )
            path.fill()

    def isFlipped(self):
        return False


_window: Optional[NSWindow] = None
_view: Optional[_BarsView] = None
_animation_timer: Optional[rumps.Timer] = None
_envelope: Optional[np.ndarray] = None
_playback_start: Optional[float] = None


def _ensure_window() -> None:
    global _window, _view
    if _window is not None:
        return
    screen = NSScreen.mainScreen().frame()
    x = (screen.size.width - WINDOW_WIDTH) / 2
    y = BOTTOM_MARGIN
    frame = NSMakeRect(x, y, WINDOW_WIDTH, WINDOW_HEIGHT)
    _window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
    )
    _window.setOpaque_(False)
    _window.setBackgroundColor_(NSColor.clearColor())
    _window.setLevel_(NSFloatingWindowLevel)
    _window.setIgnoresMouseEvents_(True)
    _window.setHasShadow_(False)
    _window.setCollectionBehavior_(1 << 8)  # NSWindowCollectionBehaviorCanJoinAllSpaces
    _view = _BarsView.alloc().initWithFrame_(NSMakeRect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT))
    _window.setContentView_(_view)


def _tick(_timer) -> None:
    if _envelope is None or _playback_start is None or _view is None:
        return
    elapsed = time.time() - _playback_start
    idx = int(elapsed * FPS)
    if idx >= len(_envelope):
        _view.bar_heights = [0.0] * NUM_BARS
    else:
        _view.bar_heights = _envelope[idx].tolist()
    _view.setNeedsDisplay_(True)


def show_and_animate(wav_path: Path) -> None:
    """Precomputes the FFT envelope and starts the window animating in
    sync with playback. Must run on the main thread — see module docstring."""
    global _envelope, _playback_start, _animation_timer
    try:
        envelope = compute_fft_envelope(wav_path)
    except (FileNotFoundError, wave.Error, ValueError):
        return
    _ensure_window()
    _envelope = envelope
    _playback_start = time.time()
    _window.orderFront_(None)
    if _animation_timer is None:
        _animation_timer = rumps.Timer(_tick, 1.0 / FPS)
    if not _animation_timer.is_alive():
        _animation_timer.start()


def hide() -> None:
    """Stops the animation and hides the window. Must run on the main
    thread — see module docstring."""
    global _envelope, _playback_start
    if _animation_timer is not None and _animation_timer.is_alive():
        _animation_timer.stop()
    _envelope = None
    _playback_start = None
    if _window is not None:
        _window.orderOut_(None)


def on_playback_change(path: Optional[Path]) -> None:
    """Registered (via menubar_app.py) as main.py's playback listener.
    path is the audio file that just started playing, or None when
    playback stopped — naturally finishing or interrupted by barge-in."""
    if path is None:
        hide()
    else:
        show_and_animate(path)
