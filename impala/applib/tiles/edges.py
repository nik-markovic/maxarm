"""Where the top-down view's colour changes, how sharply, and which way.

Stored as the structure tensor summed over L, a and b: for any direction n, the
colour gradient across an edge with that normal is sqrt(n' J n), so one lookup of
three numbers scores a point whichever way its edge runs, and whichever side is
lighter. Nothing here knows whether a tile is lighter or darker than the desk.
"""

from __future__ import annotations

import cv2
import numpy as np

from .topdown import TopDown

# Smoothing before the gradient, mm: below the tile's own rounded rim, above the camera's noise.
SMOOTHING_MM = 0.5
# Where the frame ends, the view goes black: an edge as strong as any tile's. Nothing this near it counts, mm.
FRAME_EDGE_MM = 2.5


class EdgeField:
    def __init__(self, view: TopDown, camera_xyz: np.ndarray) -> None:
        self.view = view
        self.camera_xyz = np.asarray(camera_xyz, float)     # where the camera is, desk mm
        lab = cv2.cvtColor(cv2.GaussianBlur(view.image, (0, 0), SMOOTHING_MM * view.px_per_mm),
                           cv2.COLOR_BGR2LAB).astype(np.float32)
        tensor = np.zeros(view.image.shape[:2] + (3,), np.float32)
        # A millimetre's colour change, wherever the view's scale: per pixel, times pixels per millimetre.
        scale = view.px_per_mm / 32.0
        for channel in range(3):
            gx = cv2.Scharr(lab[..., channel], cv2.CV_32F, 1, 0) * scale
            gy = cv2.Scharr(lab[..., channel], cv2.CV_32F, 0, 1) * scale
            tensor[..., 0] += gx * gx
            tensor[..., 1] += gx * gy
            tensor[..., 2] += gy * gy
        reach = max(1, round(FRAME_EDGE_MM * view.px_per_mm))
        self.seen = cv2.erode(view.is_seen.astype(np.uint8), np.ones((2 * reach + 1, 2 * reach + 1), np.uint8)) > 0
        tensor[~self.seen] = 0.0
        self.tensor = tensor

    def strongest(self):
        """Per pixel: the gradient's size the way it is greatest, and that direction (radians, 0 to pi)."""
        jxx, jxy, jyy = self.tensor[..., 0], self.tensor[..., 1], self.tensor[..., 2]
        spread = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
        strength = np.sqrt(np.maximum(0.5 * (jxx + jyy + spread), 0.0))
        return strength, 0.5 * np.arctan2(2.0 * jxy, jxx - jyy) % np.pi

    def across(self, points_mm: np.ndarray, normals: np.ndarray, is_exact: bool = True) -> np.ndarray:
        """The colour gradient across each point's edge, the edge's normal given: colour units per mm.

        Not exact: the nearest pixel's, for a coarse search."""
        u, v = self.view.to_view(points_mm[..., 0], points_mm[..., 1])
        tensor = _bilinear(self.tensor, u, v) if is_exact else _nearest(self.tensor, u, v)
        nx, ny = normals[..., 0], normals[..., 1]
        squared = nx * nx * tensor[..., 0] + 2.0 * nx * ny * tensor[..., 1] + ny * ny * tensor[..., 2]
        return np.sqrt(np.maximum(squared, 0.0))

    def is_seen(self, points_mm: np.ndarray) -> np.ndarray:
        """Whether the frame shows each point, clear of its edge."""
        u, v = self.view.to_view(points_mm[..., 0], points_mm[..., 1])
        rows = np.clip(np.round(v).astype(int), 0, self.seen.shape[0] - 1)
        columns = np.clip(np.round(u).astype(int), 0, self.seen.shape[1] - 1)
        inside = (u >= 0) & (u <= self.seen.shape[1] - 1) & (v >= 0) & (v <= self.seen.shape[0] - 1)
        return inside & self.seen[rows, columns]


def _nearest(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    rows = np.clip(np.round(v).astype(int), 0, image.shape[0] - 1)
    columns = np.clip(np.round(u).astype(int), 0, image.shape[1] - 1)
    return image[rows, columns]


def _bilinear(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    u = np.clip(u, 0, width - 1.001)
    v = np.clip(v, 0, height - 1.001)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = (u - u0)[..., None], (v - v0)[..., None]
    return ((1 - fu) * (1 - fv) * image[v0, u0] + fu * (1 - fv) * image[v0, u0 + 1]
            + (1 - fu) * fv * image[v0 + 1, u0] + fu * fv * image[v0 + 1, u0 + 1])
