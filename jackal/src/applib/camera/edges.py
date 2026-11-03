"""Where a cube drawn through a camera meets the frame: the calibration's fit and the cube finder both measure so.

- `overlap()`: how much of a blob a drawn cube's silhouette covers.
- `distance_to_outline()`: each pixel's distance to a blob's outline, for fitting a silhouette onto it.
- `crop()` and `search()`: along each edge the cube shows, the strongest colour edge in the frame across it.
  An outline counts only where the cube's colour is on its inner side: a shadow's edge, or the desk's grain,
  has the desk on both sides.
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from . import cubes
from .cubes import Blob
from .model import Edge, sample

POINTS_PER_EDGE = 24


def overlap(corners: np.ndarray, mask: np.ndarray) -> float:
    """Intersection over union of a cube's silhouette, drawn, and a blob."""
    if not np.all(np.isfinite(corners)) or np.abs(corners).max() > 1e5:
        return 0.0
    drawn = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(drawn, cv2.convexHull(np.round(corners).astype(np.int32)), 1)
    drawn = drawn > 0
    union = np.count_nonzero(drawn | mask)
    return np.count_nonzero(drawn & mask) / union if union else 0.0


def distance_to_outline(blob: Blob) -> Tuple[np.ndarray, np.ndarray]:
    """Each pixel's distance to the blob's outline, in a box round it; and the box's corner."""
    x, y, w, h = cv2.boundingRect(blob.contour)
    margin = int(0.5 * max(w, h))
    corner = np.array([x - margin, y - margin])
    drawn = np.full((h + 2 * margin, w + 2 * margin), 255, np.uint8)
    cv2.drawContours(drawn, [blob.contour - corner.astype(np.int32)], -1, 0, 1)
    return corner, cv2.distanceTransform(drawn, cv2.DIST_L2, 5)


@dataclass
class Crop:
    """What the edge search reads round one cube: its colour gradients and its colourfulness."""
    corner: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    colourful: np.ndarray
    size: float


def crop(frame: np.ndarray, blob: Blob, desk: np.ndarray) -> Crop:
    x, y, w, h = cv2.boundingRect(blob.contour)
    margin = int(0.3 * max(w, h))
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
    lab = cv2.cvtColor(cv2.GaussianBlur(frame[y0:y1, x0:x1], (0, 0), 1.2), cv2.COLOR_BGR2LAB).astype(np.float32)
    weights = np.array([1.0, 2.0, 2.0], np.float32)       # colour, not just shade, sets a cube off
    gradient_x = np.stack([cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0, ksize=3) for c in range(3)], -1) * weights
    gradient_y = np.stack([cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1, ksize=3) for c in range(3)], -1) * weights
    chroma = cubes.chroma_from_desk(frame[y0:y1, x0:x1], desk)
    return Crop(np.array([x0, y0]), gradient_x, gradient_y, np.hypot(chroma[..., 0], chroma[..., 1]), blob.size)


def search(crop: Crop, corners: np.ndarray, edges: List[Edge],
           reach_share: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Along each edge, the strongest colour edge within reach across it: (starts, ends, points found)."""
    reach = max(3.0, reach_share * crop.size)
    offsets = np.arange(-reach, reach + 0.01, 0.5)
    middle = corners[:8].mean(axis=0)
    starts, ends, bases, normals, outline = [], [], [], [], []
    for edge in edges:
        start, end = corners[edge.start], corners[edge.end]
        length = np.linalg.norm(end - start)
        if length < 4 * reach:
            continue
        normal = np.array([start[1] - end[1], end[0] - start[0]]) / length
        if edge.is_outline and normal @ ((start + end) / 2 - middle) < 0:
            normal = -normal                                 # outward, for an outline
        for share in np.linspace(0.12, 0.88, POINTS_PER_EDGE):
            starts.append(edge.start)
            ends.append(edge.end)
            bases.append(start + share * (end - start))
            normals.append(normal)
            outline.append(edge.is_outline)
    if not bases:
        return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 2))
    bases, normals = np.array(bases), np.array(normals)
    across = bases[:, None, :] + offsets[None, :, None] * normals[:, None, :] - crop.corner
    flat = across.reshape(-1, 2)
    strength = np.abs(sample(crop.gradient_x, flat) * np.repeat(normals[:, 0], len(offsets))[:, None]
                      + sample(crop.gradient_y, flat) * np.repeat(normals[:, 1], len(offsets))[:, None])
    strength = np.linalg.norm(strength, axis=1).reshape(len(bases), len(offsets))
    best = np.argmax(strength, axis=1)
    rows = np.arange(len(bases))
    peak = strength[rows, best]
    inner = (0 < best) & (best < len(offsets) - 1)
    before, after = strength[rows, np.maximum(best - 1, 0)], strength[rows, np.minimum(best + 1, len(offsets) - 1)]
    bend = before - 2 * peak + after
    shift = np.where(inner & (bend < 0), 0.5 * (before - after) / np.where(bend < 0, bend, -1.0), 0.0)
    found = bases + ((offsets[best] + shift * 0.5)[:, None]) * normals
    keep = inner & (peak > 0.35 * np.median(peak))
    side = max(2.0, 0.02 * crop.size)
    inside = sample(crop.colourful, found - side * normals - crop.corner)
    outside = sample(crop.colourful, found + side * normals - crop.corner)
    keep &= ~np.array(outline) | (inside > outside)
    return np.array(starts)[keep], np.array(ends)[keep], found[keep]


def signed_distances(corners: np.ndarray, starts: np.ndarray, ends: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """How far each point found lies across the line of the edge it was found for, pixels."""
    side = corners[ends] - corners[starts]
    normal = np.column_stack((-side[:, 1], side[:, 0])) / np.linalg.norm(side, axis=1)[:, None]
    return np.sum((targets - corners[starts]) * normal, axis=1)
