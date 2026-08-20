"""Set-of-Mark annotation: make a wall photo referable by a language model.

Every route hold is outlined and tagged with the id that also appears in the
JSON payload, so the model can name a hold without ambiguity. Everything
off-route is dimmed so the route stands out.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .holds import Hold

DIM = 0.42                      # brightness multiplier for off-route pixels
PAD = 0.15                      # box padding kept undimmed, as a fraction of size
START_COLOR = (255, 214, 10)
ROUTE_COLOR = (255, 255, 255)

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
    "Arial.ttf",
)


@lru_cache(maxsize=32)
def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """First available system font at `size`, falling back to PIL's bitmap font."""
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size, index=1 if bold and path.endswith(".ttc") else 0)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def draw_som(img: Image.Image, route: list[Hold], starts: set[int]) -> Image.Image:
    """Dim everything but the route, outline route holds, and tag them by id.

    Tag placement is collision-aware: each chip tries the corners and sides of
    its box and takes the position overlapping the fewest already-placed chips
    and hold boxes. A leader line keeps it clear which box a chip belongs to.
    """
    width, height = img.size
    base = np.asarray(img, dtype=np.float64)
    mask = np.zeros((height, width), dtype=bool)
    for h in route:
        pad_x, pad_y = h.w * PAD, h.h * PAD
        y0, y1 = max(0, int(h.y1 - pad_y)), min(height, int(h.y2 + pad_y))
        x0, x1 = max(0, int(h.x1 - pad_x)), min(width, int(h.x2 + pad_x))
        mask[y0:y1, x0:x1] = True
    som = Image.fromarray(np.where(mask[..., None], base, base * DIM).astype(np.uint8))

    draw = ImageDraw.Draw(som)
    # Denser walls get smaller tags so the chips do not cover the wall.
    density = max(0.68, min(1.1, (25 / max(1, len(route))) ** 0.5))
    tag_size = max(15, int(min(width, height) * 0.028 * density))
    font = load_font(tag_size)
    stroke = max(2, tag_size // 8)
    boxes = [(h.x1, h.y1, h.x2, h.y2) for h in route]
    placed: list[tuple[float, float, float, float]] = []

    for hold in route:
        is_start = hold.id in starts
        color = START_COLOR if is_start else ROUTE_COLOR
        box = (hold.x1, hold.y1, hold.x2, hold.y2)
        draw.rectangle(box, outline=color, width=stroke)

        label = f"S{hold.id}" if is_start else str(hold.id)
        pad = tag_size * 0.28
        cw = draw.textlength(label, font=font) + 2 * pad
        ch = tag_size + 2 * pad
        candidates = (
            (hold.x1 - 2, hold.y1 - ch - 2),          # above, left-aligned
            (hold.x1 - 2, hold.y2 + 2),               # below, left-aligned
            (hold.x2 + 2, hold.y1 - 2),               # right of the top edge
            (hold.x1 - cw - 2, hold.y1 - 2),          # left of the top edge
            (hold.x2 - cw + 2, hold.y1 - ch - 2),     # above, right-aligned
        )
        best, best_score = None, None
        for bx, by in candidates:
            bx = min(max(0.0, bx), width - cw)
            by = min(max(0.0, by), height - ch)
            chip = (bx, by, bx + cw, by + ch)
            score = (sum(3 for p in placed if _overlaps(chip, p))
                     + sum(1 for b in boxes if b != box and _overlaps(chip, b)))
            if best_score is None or score < best_score:
                best, best_score = chip, score
                if score == 0:
                    break
        placed.append(best)

        bx, by = best[0], best[1]
        corner = (hold.x1 if bx + cw / 2 < hold.cx else hold.x2,
                  hold.y1 if by + ch / 2 < hold.cy else hold.y2)
        draw.line([(bx + cw / 2, by + ch / 2), corner], fill=color, width=stroke)
        draw.rounded_rectangle(best, radius=pad, fill=color)
        draw.text((bx + pad, by + pad * 0.8), label, font=font, fill=(0, 0, 0))
    return som
