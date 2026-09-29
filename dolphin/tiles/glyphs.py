"""Printed letters in the top-down view: dark strokes on a quiet face.

The one thing every tile has on every desk is ink darker than the wood it is
printed on. Nothing here assumes the tile is brighter or darker than the desk,
or any absolute brightness at all:

- **Stroke depth is relative.** A black-hat of lightness, divided by the local
  closing, is how much darker a thin stroke is than whatever surrounds it -- a
  Weber contrast, so exposure and light level cancel.
- **The cut is the frame's own.** Otsu over the whole view splits "textured"
  from "smooth"; ink is well past it, so several multiples of it are tried and
  each place keeps the cut that isolates it best. A crack in the wood joins a
  letter at a low cut and falls away at a higher one.
- **A glyph sits on a quiet face.** The ring just outside it must be far
  quieter than its strokes. Wood grain, cables and the arm are texture beside
  texture and fail that; a letter on a tile is ink beside a plain face.
- **Its size is physical.** Capital letters on these tiles are 5-12 mm; the
  4-15 mm window is millimetres, which the calibrated view makes a pixel count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from topdown import TopDown

GLYPH_SIZE_MM = (4.0, 15.0)
# Wider than the thickest stroke, narrower than the tile face around it.
STROKE_KERNEL_MM = 4.0
# A 1.5 mm gap keeps a stroke's own blur out of the ring; 5 mm stays on the face.
RING_MM = (1.5, 5.0)
CUT_MULTIPLES = (1.0, 1.5, 2.0, 2.5, 3.0)
# The ring must be this much quieter than the strokes. Loose on purpose: the
# reader decides what is a letter. Lone tiles measured 0.006-0.07, but a
# neighbour's edge inside the ring puts touching tiles at 0.11, and wood grain
# starts near 0.1 -- the reader turns that away, and nothing false got through.
MAX_RING_TO_STROKE = 0.2
# Two candidates closer than this are the same letter found at two cuts.
SAME_GLYPH_MM = 6.0
# Strokes are about 1 mm wide; three pixels across them is enough to find them,
# and a quarter of the pixels of the 6 px/mm view.
SEARCH_PX_PER_MM = 3.0


@dataclass(frozen=True)
class Glyph:
    centre: Tuple[float, float]         # view pixels, the stroke pixels' centroid
    size_mm: Tuple[float, float]        # bounding box, view axes
    contour: np.ndarray                 # view pixels
    quietness: float                    # ring / stroke contrast; lower is cleaner


def stroke_contrast(image: np.ndarray, px_per_mm: float) -> np.ndarray:
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0]
    kernel = _disk(STROKE_KERNEL_MM, px_per_mm)
    closing = cv2.morphologyEx(lightness, cv2.MORPH_CLOSE, kernel).astype(np.float32)
    return (closing - lightness) / (closing + 1.0)


def find_glyphs(view: TopDown) -> List[Glyph]:
    """Every printed letter in the view. Positions and outlines are in view pixels."""
    shrink = min(1.0, SEARCH_PX_PER_MM / view.px_per_mm)
    image = cv2.resize(view.image, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
    per_mm = view.px_per_mm * shrink
    contrast = stroke_contrast(image, per_mm)
    valid = cv2.erode((image.sum(axis=2) > 0).astype(np.uint8), _disk(RING_MM[1] * 2, per_mm)) > 0
    otsu, _ = cv2.threshold((contrast[valid] * 255).astype(np.uint8).reshape(-1, 1), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    texture = cv2.blur(contrast, (round(2 * per_mm),) * 2)
    candidates = []
    for multiple in CUT_MULTIPLES:
        cut = otsu / 255.0 * multiple
        candidates += _candidates(contrast, texture, valid, cut, per_mm)
    candidates.sort(key=lambda glyph: glyph.quietness)
    kept: List[Glyph] = []
    same = SAME_GLYPH_MM * per_mm
    for glyph in candidates:
        if all(np.hypot(glyph.centre[0] - other.centre[0], glyph.centre[1] - other.centre[1]) > same
               for other in kept):
            kept.append(glyph)
    grow = 1.0 / shrink
    return [Glyph((g.centre[0] * grow, g.centre[1] * grow), g.size_mm,
                  np.round(g.contour * grow).astype(np.int32), g.quietness) for g in kept]


def _candidates(contrast: np.ndarray, texture: np.ndarray, valid: np.ndarray, cut: float,
                per_mm: float) -> List[Glyph]:
    mask = ((contrast > cut) & valid).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    smallest, largest = (size * per_mm for size in GLYPH_SIZE_MM)
    near, far = _disk(RING_MM[0] * 2, per_mm), _disk(RING_MM[1] * 2, per_mm)
    pad = far.shape[0]
    found = []
    for label in range(1, count):
        x, y, w, h, _ = stats[label]
        if not smallest <= max(w, h) <= largest:
            continue
        # Most blobs are wood grain with more grain around them. A dozen samples
        # of the blurred texture around the box turn those away before the exact
        # ring below, which is what costs; the bar is twice as loose as that one.
        stroke_mean = float(contrast[y:y + h, x:x + w][labels[y:y + h, x:x + w] == label].mean())
        if _box_ring(texture, x, y, w, h, (RING_MM[0] + RING_MM[1]) / 2 * per_mm) \
                > 2 * MAX_RING_TO_STROKE * stroke_mean:
            continue
        y0, x0 = max(0, y - pad), max(0, x - pad)
        y1, x1 = min(mask.shape[0], y + h + pad), min(mask.shape[1], x + w + pad)
        stroke = (labels[y0:y1, x0:x1] == label).astype(np.uint8)
        ring = (cv2.dilate(stroke, far) > 0) & (cv2.dilate(stroke, near) == 0)
        patch = contrast[y0:y1, x0:x1]
        quietness = float(patch[ring].mean() / patch[stroke > 0].mean())
        if quietness > MAX_RING_TO_STROKE:
            continue
        contour = max(cv2.findContours(stroke, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=len)
        found.append(Glyph((float(centroids[label][0]), float(centroids[label][1])),
                           (w / per_mm, h / per_mm), contour + (x0, y0), quietness))
    return found


def _box_ring(texture: np.ndarray, x: int, y: int, w: int, h: int, gap: float) -> float:
    """Mean texture at twelve points on a rectangle `gap` outside a blob's box."""
    left, right, top, bottom = x - gap, x + w + gap, y - gap, y + h + gap
    along = np.linspace(0.0, 1.0, 4)
    points = np.vstack([np.column_stack((left + (right - left) * along, np.full(4, top))),
                        np.column_stack((left + (right - left) * along, np.full(4, bottom))),
                        np.column_stack((np.full(2, left), top + (bottom - top) * along[1:3])),
                        np.column_stack((np.full(2, right), top + (bottom - top) * along[1:3]))])
    cols = np.clip(np.round(points[:, 0]).astype(int), 0, texture.shape[1] - 1)
    rows = np.clip(np.round(points[:, 1]).astype(int), 0, texture.shape[0] - 1)
    return float(texture[rows, cols].mean())


def _disk(diameter_mm: float, px_per_mm: float) -> np.ndarray:
    radius = max(1, round(diameter_mm * px_per_mm / 2))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
