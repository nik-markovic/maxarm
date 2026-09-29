"""A tile's top face as a flat, square-on image, taken straight from the camera frame.

The camera maps the plane 4 mm up to pixels by a homography, so a window laid
out in millimetres on that plane -- centred on the tile, turned to its axes --
can be sampled from the original frame in one interpolation. The window is a
little larger than the tile, so a fit a millimetre out still shows the whole
letter, and `turn` rotates it by quarter turns for a reader that expects the
letter upright.
"""

from __future__ import annotations

import cv2
import numpy as np

from mapping import DeskMapping

WINDOW_MM = 22.0
PX_PER_MM = 4.0


def tile_face(frame: np.ndarray, mapping: DeskMapping, centre_mm, angle_deg: float,
              height_mm: float, turn: int = 0, window_mm: float = WINDOW_MM,
              px_per_mm: float = PX_PER_MM) -> np.ndarray:
    """The top face with the baseline at `angle_deg` drawn horizontal, then turned
    `turn` quarter turns anticlockwise on the page."""
    size = round(window_mm * px_per_mm)
    angle = np.radians(angle_deg + 90.0 * turn)
    baseline = np.array([np.cos(angle), np.sin(angle)])
    # Page up is the letter's up. Which perpendicular that is on the desk is
    # a handedness question; the view's layout (-y right, +x down) settles it.
    up = np.array([baseline[1], -baseline[0]])
    scale = 1.0 / px_per_mm
    origin = np.array(centre_mm) - baseline * window_mm / 2 + up * window_mm / 2
    # page (col, row, 1) -> ground (x, y, 1)
    page_to_ground = np.array([
        [baseline[0] * scale, -up[0] * scale, origin[0]],
        [baseline[1] * scale, -up[1] * scale, origin[1]],
        [0.0, 0.0, 1.0]])
    page_to_frame = mapping.plane_to_pixel(height_mm) @ page_to_ground
    return cv2.warpPerspective(frame, page_to_frame, (size, size),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
