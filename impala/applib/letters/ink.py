"""The ink on a tile face: whether there is any, and the dot that marks a letter's bottom right.

Ink is measured against the face round it, not against a level: each pixel's
darkness under the brightest of the face within `PATCH_MM` (a morphological
closing; Weber contrast), so a shadow or light falling off across the face
cancels. A pixel is ink at half the face's own strongest contrast.

On the owner's tiles the marker's strongest contrast is 0.56 to 0.77 and the
bare face's about 0 (STATUS-hyena); a face whose strongest is under
`MIN_INK_CONTRAST` holds no letter.

The dot is a small, round, separate blob of ink in a corner of the letter --
off its middle both ways -- and that corner is the letter's bottom right. A
faint stroke breaks into pieces at half the strongest ink, and a piece can look
like a dot: so the dot must be no longer than twice its width, and still apart
from the letter in the faint ink too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

WINDOW_MM = 23.0            # inside the face's rounded, darker rim
PX_PER_MM = 8.0
# Wider than a marker's stroke, so a closing fills the stroke with the face's own lightness.
PATCH_MM = 5.0
MIN_INK_CONTRAST = 0.25
# The strongest ink: a high percentile, not the maximum, so that a speck does not set it.
STRONGEST_PERCENTILE = 99.8
# Where a letter must have some ink: the face's middle.
MIDDLE_MM = 17.0
# A dot: this much ink, no larger across, round enough; this far off the letter's middle each way, as a share of
# the letter's half width and half height.
DOT_AREA_MM2 = (0.3, 9.0)
DOT_SIZE_MM = 3.5
DOT_MIN_FILL = 0.45
DOT_MAX_ELONGATION = 2.0
DOT_OFF_MIDDLE = 0.2
# Faint ink, a quarter of the face's strongest: what joins a piece of a stroke back to its letter.
FAINT_SHARE = 0.25


@dataclass(frozen=True)
class Ink:
    is_blank: bool
    contrast: float                             # the face's strongest ink, against the face round it
    dot: Optional[Tuple[float, float]]          # the dot, quarter turn 0's window, mm right and down of the centre


def size_px() -> int:
    return round(WINDOW_MM * PX_PER_MM)


def measure(grey: np.ndarray) -> Ink:
    """`grey`: the face at quarter turn 0, `WINDOW_MM` square at `PX_PER_MM`."""
    image = grey.astype(np.float32)
    patch = _odd(PATCH_MM * PX_PER_MM)
    lightest = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (patch, patch)))
    contrast = (lightest - image) / np.maximum(lightest, 1.0)
    strongest = float(np.percentile(contrast, STRONGEST_PERCENTILE))
    ink = contrast >= max(strongest / 2, MIN_INK_CONTRAST / 2)
    size = image.shape[0]
    margin = round((WINDOW_MM - MIDDLE_MM) / 2 * PX_PER_MM)
    if strongest < MIN_INK_CONTRAST or not ink[margin:size - margin, margin:size - margin].any():
        return Ink(True, strongest, None)
    return Ink(False, strongest, _dot(ink, contrast >= FAINT_SHARE * strongest))


def _dot(ink: np.ndarray, faint: np.ndarray) -> Optional[Tuple[float, float]]:
    count, labels, stats, centres = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    mm2 = PX_PER_MM ** 2
    size = ink.shape[0]
    dots = []
    for index in range(1, count):
        left, top, width, height, area = stats[index]
        touches_edge = left == 0 or top == 0 or left + width == size or top + height == size
        is_small = DOT_AREA_MM2[0] * mm2 <= area <= DOT_AREA_MM2[1] * mm2 \
            and max(width, height) <= DOT_SIZE_MM * PX_PER_MM
        is_round = area >= DOT_MIN_FILL * width * height \
            and max(width, height) <= DOT_MAX_ELONGATION * min(width, height)
        if not touches_edge and is_small and is_round:
            dots.append(index)
    if len(dots) != 1:
        return None
    letter = (labels > 0) & (labels != dots[0])
    if not letter.any():
        return None
    _, faint_labels = cv2.connectedComponents(faint.astype(np.uint8), connectivity=8)
    if (letter & (faint_labels == faint_labels[labels == dots[0]][0])).any():
        return None
    rows, columns = np.nonzero(letter)
    middle = np.array([(columns.min() + columns.max()) / 2, (rows.min() + rows.max()) / 2])
    half = np.array([max(columns.max() - columns.min(), 1) / 2, max(rows.max() - rows.min(), 1) / 2])
    off = (centres[dots[0]] - middle) / half
    if min(abs(off)) < DOT_OFF_MIDDLE:
        return None
    centre = (size - 1) / 2
    return (float((centres[dots[0]][0] - centre) / PX_PER_MM), float((centres[dots[0]][1] - centre) / PX_PER_MM))


def _odd(pixels: float) -> int:
    return int(pixels) // 2 * 2 + 1
