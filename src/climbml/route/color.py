"""Hold colour estimation: white balance, sampling, colour bins.

Routes are colour-coded, so grouping holds is mostly a colour question. A plain
mean over the bounding box measures the wall rather than the hold, so two
corrections do most of the work:

* Half-strength gray-world white balance. Gym lighting is tinted, and a full
  correction assumes the scene averages to gray. Wood walls do not, and the full
  correction pushes neutral holds into Blue. The square root halves it.
* Median over the hold's own pixels. The border band of the box is mostly wall,
  so it gives a local wall colour. The interior pixels furthest from that colour
  are the hold. Median rather than mean, so chalk and glare do not drag it.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .holds import Hold

# Saturation/value gates separating chromatic holds from neutral plastic.
SAT_MIN = 0.16
VAL_BLACK = 0.22
WHITE_V = 0.55

# Perceptual hue bins in degrees, [lo, hi); Red wraps through 360.
HUE_BINS = (
    ("Red", 345, 15), ("Orange", 15, 40), ("Yellow", 40, 68), ("Green", 68, 150),
    ("Teal", 150, 195), ("Blue", 195, 240), ("Purple", 240, 290), ("Pink", 290, 345),
)
NEUTRALS = frozenset({"White", "Gray", "Black"})


def rgb_to_hsv(rgb) -> tuple[float, float, float]:
    """(r, g, b) in 0-255 -> (hue degrees, saturation, value)."""
    r, g, b = (float(v) / 255 for v in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    hue = 0.0
    if d > 0:
        if mx == r:
            hue = 60 * (((g - b) / d) % 6)
        elif mx == g:
            hue = 60 * ((b - r) / d + 2)
        else:
            hue = 60 * ((r - g) / d + 4)
    return hue, 0.0 if mx == 0 else d / mx, mx


def bin_name(hue: float, sat: float, val: float) -> str:
    """Name the colour family a hold belongs to."""
    # Dark and desaturated is black rubber in shadow, not a dim chromatic hold.
    if val < VAL_BLACK or (val < 0.32 and sat < 0.40):
        return "Black"
    # Bright holds with a faint tint are white plastic under coloured light, so
    # the saturation cutoff ramps 0.16 -> 0.28 as value goes 0.70 -> 1.0.
    # Real pastel holds stay above it (s >= 0.3).
    sat_cut = SAT_MIN + 0.12 * max(0.0, val - 0.70) / 0.30
    if sat < sat_cut:
        return "White" if val >= WHITE_V else "Gray"
    for name, lo, hi in HUE_BINS:
        if lo > hi:                       # wraps through 360
            if hue >= lo or hue < hi:
                return name
        elif lo <= hue < hi:
            return name
    return "Gray"


class Sampler:
    """Samples a representative colour per hold from a white-balanced image."""

    #: pixels sampled per box edge, and the inset separating interior from border
    GRID = 17
    INSET = 0.26
    #: minimum RGB distance from the local wall estimate for a pixel to count
    WALL_DELTA = 26

    def __init__(self, img: Image.Image, max_side: int = 1024):
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        self.scale = scale
        small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        self.px = np.asarray(small.convert("RGB"), dtype=np.float64)
        means = self.px.reshape(-1, 3)[::7].mean(axis=0)
        self.gain = np.clip(np.sqrt(means.mean() / np.maximum(1, means)), 0.65, 1.5)

    def median_hsv(self, hold: Hold) -> tuple[float, float, float] | None:
        """Median HSV of the hold's own pixels, or None if the box is degenerate."""
        height, width = self.px.shape[:2]
        s = self.scale
        x0, x1 = max(0, int(hold.x1 * s)), min(width - 1, int(hold.x2 * s))
        y0, y1 = max(0, int(hold.y1 * s)), min(height - 1, int(hold.y2 * s))
        if x1 < x0 or y1 < y0:
            return None

        xs = np.unique(np.linspace(x0, x1, self.GRID).astype(int))
        ys = np.unique(np.linspace(y0, y1, self.GRID).astype(int))
        gx, gy = np.meshgrid(xs, ys)
        gx, gy = gx.ravel(), gy.ravel()
        samples = np.minimum(255, self.px[gy, gx] * self.gain)
        if len(samples) < 9:
            return None

        ix0, ix1 = x0 + (x1 - x0) * self.INSET, x1 - (x1 - x0) * self.INSET
        iy0, iy1 = y0 + (y1 - y0) * self.INSET, y1 - (y1 - y0) * self.INSET
        inside = (gx >= ix0) & (gx <= ix1) & (gy >= iy0) & (gy <= iy1)
        interior, border = samples[inside], samples[~inside]
        if len(interior) == 0:
            return rgb_to_hsv(np.median(samples, axis=0))

        wall = np.median(border if len(border) else samples, axis=0)
        dist = np.linalg.norm(interior - wall, axis=1)
        order = np.argsort(-dist)[: max(5, len(interior) // 2)]
        kept = interior[order[dist[order] > self.WALL_DELTA]]
        pixels = kept if len(kept) >= 5 else interior
        return rgb_to_hsv(np.median(pixels, axis=0))
