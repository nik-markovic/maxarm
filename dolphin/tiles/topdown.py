"""The camera frame resampled onto a plane of the desk, as if seen from straight above.

A calibrated camera maps any horizontal plane to pixels by a homography, so the
plane a tile's top face lies on -- 4 mm up -- can be resampled into an image in
millimetres. In it every tile is the same 20 x 17.5 mm rectangle wherever it lies,
perspective and foreshortening are gone, and a letter is upright once the tile is.

The view is laid out like the camera's own: arm -y to the right, arm +x down,
which keeps letters the way round a person at the camera reads them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from mapping import DeskMapping

# The work area the calibration covers, in AACS millimetres, and a scale that
# keeps the nearest tiles at about their native resolution.
X_RANGE_MM = (-180.0, 140.0)
Y_RANGE_MM = (-340.0, -40.0)
PX_PER_MM = 6.0


@dataclass(frozen=True)
class TopDown:
    image: np.ndarray
    height_mm: float
    px_per_mm: float
    view_to_ground: np.ndarray      # 3x3: view pixel -> AACS (x, y)

    def to_ground(self, point: Tuple[float, float]) -> Tuple[float, float]:
        mapped = self.view_to_ground @ np.array([point[0], point[1], 1.0])
        return (float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2]))

    def to_view(self, ground: Tuple[float, float]) -> Tuple[float, float]:
        mapped = np.linalg.inv(self.view_to_ground) @ np.array([ground[0], ground[1], 1.0])
        return (float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2]))


def render(frame: np.ndarray, mapping: DeskMapping, height_mm: float,
           px_per_mm: float = PX_PER_MM) -> TopDown:
    """Resample `frame` onto the plane `height_mm` above the desk."""
    view_to_ground = np.array([
        [0.0, 1.0 / px_per_mm, X_RANGE_MM[0]],
        [-1.0 / px_per_mm, 0.0, Y_RANGE_MM[1]],
        [0.0, 0.0, 1.0]])
    view_to_frame = mapping.plane_to_pixel(height_mm) @ view_to_ground
    size = (round((Y_RANGE_MM[1] - Y_RANGE_MM[0]) * px_per_mm),
            round((X_RANGE_MM[1] - X_RANGE_MM[0]) * px_per_mm))
    image = cv2.warpPerspective(frame, view_to_frame, size,
                                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return TopDown(image, height_mm, px_per_mm, view_to_ground)
