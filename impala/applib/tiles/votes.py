"""Where tiles may be: every colour edge votes for the centre of a square it could be a side of.

An edge point could be on the side of a tile whose centre is half a tile away
along the edge's normal, on either side: which side the face is on is not known,
nor whether it is lighter or darker than what lies beyond. A tile's four sides
all vote for its centre, and the seam between two touching tiles votes for both
of theirs.

Wood grain votes as well, but its edges all run one way, while a square's run two
ways at right angles. So each place counts its votes less the share that runs one
way only, measured on the doubled angle: two directions at right angles cancel
there, one direction does not. What is left peaks at the tiles' centres, and the
fit (`pose.py`) decides which peaks are tiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from .edges import EdgeField

# An edge votes if it is this many times the view's typical gradient: most of the desk's grain does not.
VOTE_FLOOR = 2.0
# Votes are pooled over this, mm.
POOL_MM = 1.0
# One peak per this much desk, mm across: a tile's own peak is not a point but a small cross, and the fit
# searches round each one (`pose.SEARCH_MM`).
PEAK_SPACING_MM = 18.0
# Peaks under this share of the view's best are not worth a fit: grain, stains, the arm's shadow.
PEAK_SHARE = 0.08


@dataclass(frozen=True)
class Candidate:
    x: float               # desk mm
    y: float
    votes: float


def candidates(field: EdgeField, half_mm: float, limit: int) -> List[Candidate]:
    """The places most voted for as the centre of a square `2 * half_mm` across, best first, at most `limit`."""
    view = field.view
    strength, direction = field.strongest()
    seen = strength[field.seen]
    if not seen.size:
        return []
    rows, columns = np.nonzero(strength > VOTE_FLOOR * float(np.median(seen)))
    weight = strength[rows, columns]
    angle = direction[rows, columns]
    reach = half_mm * view.px_per_mm
    height, width = strength.shape
    total = np.zeros(height * width)
    one_way = np.zeros((2, height * width))
    for sign in (1.0, -1.0):
        u = np.round(columns + sign * reach * np.cos(angle)).astype(int)
        v = np.round(rows + sign * reach * np.sin(angle)).astype(int)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        at = v[inside] * width + u[inside]
        total += np.bincount(at, weight[inside], height * width)
        one_way[0] += np.bincount(at, (weight * np.cos(2 * angle))[inside], height * width)
        one_way[1] += np.bincount(at, (weight * np.sin(2 * angle))[inside], height * width)
    sigma = POOL_MM * view.px_per_mm
    pooled = [cv2.GaussianBlur(layer.reshape(height, width).astype(np.float32), (0, 0), sigma)
              for layer in (total, *one_way)]
    score = pooled[0] - np.hypot(pooled[1], pooled[2])
    score[~field.seen] = 0.0
    spacing = max(3, round(PEAK_SPACING_MM * view.px_per_mm)) | 1
    is_peak = (score >= cv2.dilate(score, np.ones((spacing, spacing), np.uint8))) & (score > 0)
    peak_rows, peak_columns = np.nonzero(is_peak)
    values = score[peak_rows, peak_columns]
    if not values.size:
        return []
    order = np.argsort(-values)
    keep = order[values[order] >= PEAK_SHARE * values[order[0]]][:limit]
    xs, ys = view.to_desk(peak_columns[keep], peak_rows[keep])
    return [Candidate(float(x), float(y), float(value)) for x, y, value in zip(xs, ys, values[keep])]
