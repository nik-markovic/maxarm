"""Where the top-down view's color changes, how sharply, and which way.

Stored as the structure tensor summed over L, a and b: for any direction n, the
color gradient across an edge with that normal is sqrt(n' J n), so one lookup of
three numbers scores a point whichever way its edge runs, and whichever side is
lighter. Nothing here knows whether a tile is lighter or darker than the desk.

A field can be kept in a folder for another process to take up (`keep()`, `kept()`): the tensor is 70 MB, and a
file in memory is shared by mapping it, where a pipe would copy it to each.
"""

from __future__ import annotations

import pickle
from pathlib import Path

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
        # A millimeter's color change, wherever the view's scale: per pixel, times pixels per millimeter.
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

    def keep(self, folder: Path) -> None:
        """Written into `folder`, for `kept()`."""
        np.save(folder / "tensor.npy", self.tensor)
        np.save(folder / "seen.npy", self.seen)
        view = self.view
        (folder / "view.pickle").write_bytes(pickle.dumps((view.height_mm, view.px_per_mm, view.origin,
                                                           self.camera_xyz)))

    @classmethod
    def kept(cls, folder: Path) -> "EdgeField":
        """The field `keep()` wrote into `folder`, its arrays mapped, not read. Its view has no picture: the edges
        are all a fit asks of it."""
        field = cls.__new__(cls)
        height_mm, px_per_mm, origin, field.camera_xyz = pickle.loads((folder / "view.pickle").read_bytes())
        field.view = TopDown(None, None, height_mm, px_per_mm, origin)
        field.tensor = np.asarray(np.load(folder / "tensor.npy", mmap_mode="r"))
        field.seen = np.asarray(np.load(folder / "seen.npy", mmap_mode="r"))
        return field

    def strongest(self):
        """Per pixel: the gradient's size the way it is greatest, and that direction (radians, 0 to pi)."""
        jxx, jxy, jyy = self.tensor[..., 0], self.tensor[..., 1], self.tensor[..., 2]
        spread = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
        strength = np.sqrt(np.maximum(0.5 * (jxx + jyy + spread), 0.0))
        return strength, 0.5 * np.arctan2(2.0 * jxy, jxx - jyy) % np.pi

    def across(self, points_mm: np.ndarray, normals: np.ndarray, is_exact: bool = True) -> np.ndarray:
        """The color gradient across each point's edge, the edge's normal given: color units per mm.

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
    return _pixels(image, rows * image.shape[1] + columns)


def _bilinear(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    u = np.clip(u, 0, width - 1.001)
    v = np.clip(v, 0, height - 1.001)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = (u - u0)[..., None], (v - v0)[..., None]
    at = v0 * width + u0
    return ((1 - fu) * (1 - fv) * _pixels(image, at) + fu * (1 - fv) * _pixels(image, at + 1)
            + (1 - fu) * fv * _pixels(image, at + width) + fu * fv * _pixels(image, at + width + 1))


def _pixels(image: np.ndarray, at: np.ndarray) -> np.ndarray:
    """`image`'s pixels at `at`, row * width + column: what `image[row, column]` gives, sooner. The fits read some
    tens of millions of them a frame."""
    return np.take(image.reshape(-1, image.shape[-1]), at, axis=0)
