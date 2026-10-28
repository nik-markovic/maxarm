"""The camera frame resampled onto a plane over the desk, as if seen from straight above.

A calibrated camera maps any horizontal plane to pixels by a homography, so the
plane the tiles' top faces lie on, 2 mm up, can be resampled into an image in
millimetres. In it every tile's top face is the same 25 mm square wherever it
lies, and perspective is gone.

The view is laid out like the pages' view from above: desk +x to the right,
+y down, so the arm reaches up the image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from maxarm import Camera

Box = Tuple[float, float, float, float]          # desk mm: x from, x to, y from, y to

# Close to what the camera itself resolves on the desk: it sees about 2.5 to 5 px a millimetre.
PX_PER_MM = 4.0
# Pixels at the frame's very edge are interpolated against nothing.
EDGE_MARGIN_PX = 2


@dataclass(frozen=True)
class TopDown:
    image: np.ndarray               # BGR
    is_seen: np.ndarray             # bool: the frame shows the desk there
    height_mm: float
    px_per_mm: float
    origin: Tuple[float, float]     # desk x, y at the centre of pixel (0, 0)

    def to_desk(self, u, v):
        """View pixels to desk mm; arrays or numbers."""
        return self.origin[0] + np.asarray(u) / self.px_per_mm, self.origin[1] + np.asarray(v) / self.px_per_mm

    def to_view(self, x, y):
        """Desk mm to view pixels; arrays or numbers."""
        return (np.asarray(x) - self.origin[0]) * self.px_per_mm, (np.asarray(y) - self.origin[1]) * self.px_per_mm


def render(frame: np.ndarray, camera: Camera, height_mm: float, box: Box,
           px_per_mm: float = PX_PER_MM) -> TopDown:
    """`frame` resampled onto the plane `height_mm` over the desk, over `box`, cut down to what the frame shows."""
    box = _seen_part(camera, height_mm, box, (frame.shape[1], frame.shape[0]))
    x0, x1, y0, y1 = box
    view_to_desk = np.array([[1 / px_per_mm, 0.0, x0], [0.0, 1 / px_per_mm, y0], [0.0, 0.0, 1.0]])
    view_to_frame = camera.plane_to_pixel(height_mm) @ view_to_desk
    size = (max(1, round((x1 - x0) * px_per_mm)), max(1, round((y1 - y0) * px_per_mm)))
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    image = cv2.warpPerspective(frame, view_to_frame, size, flags=flags)
    inside = np.zeros(frame.shape[:2], np.uint8)
    inside[EDGE_MARGIN_PX:-EDGE_MARGIN_PX, EDGE_MARGIN_PX:-EDGE_MARGIN_PX] = 255
    is_seen = cv2.warpPerspective(inside, view_to_frame, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
    rows, columns = np.mgrid[0:size[1], 0:size[0]]
    is_seen &= _is_in_front(camera.matrix, x0 + columns / px_per_mm, y0 + rows / px_per_mm, height_mm)
    return TopDown(image, is_seen, height_mm, px_per_mm, (x0, y0))


def _seen_part(camera: Camera, height_mm: float, box: Box, frame_size: Tuple[int, int]) -> Box:
    """`box`, cut down to the bounding box of the part of it the frame shows; `box` itself if none of it."""
    x0, x1, y0, y1 = box
    xs, ys = np.meshgrid(np.linspace(x0, x1, 67), np.linspace(y0, y1, 67))
    points = np.column_stack((xs.ravel(), ys.ravel(), np.full(xs.size, height_mm)))
    pixels = np.column_stack((points, np.ones(len(points)))) @ camera.matrix.T
    in_front = pixels[:, 2] > 0
    u, v = pixels[:, 0] / np.where(in_front, pixels[:, 2], 1), pixels[:, 1] / np.where(in_front, pixels[:, 2], 1)
    width, height = frame_size
    seen = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not seen.any():
        return box
    step_x, step_y = (x1 - x0) / 66, (y1 - y0) / 66
    return (max(x0, xs.ravel()[seen].min() - step_x), min(x1, xs.ravel()[seen].max() + step_x),
            max(y0, ys.ravel()[seen].min() - step_y), min(y1, ys.ravel()[seen].max() + step_y))


def _is_in_front(matrix: np.ndarray, x: np.ndarray, y: np.ndarray, height_mm: float) -> np.ndarray:
    """Whether each desk point is in front of the camera: behind it, a homography maps it into the frame too."""
    w = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2] * height_mm + matrix[2, 3]
    return w > 0
