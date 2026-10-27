"""A tile's top face as a flat, square-on image, sampled straight from the camera frame.

The camera maps the plane of the tiles' top faces to pixels by a homography, so
a window laid out in desk millimetres on that plane -- centred on the tile,
turned to its sides -- is one resampling of the original frame. Quarter turn
`quarter` lays the window out along the tile's sides turned by `quarter` x 90
degrees, which shows the face turned a quarter turn each time.

In a window, image right is desk direction `side`, the tile's turn plus
`quarter` x 90 degrees, and image down is 90 degrees on from it; so a letter
upright in that window has its top towards `side` - 90 degrees on the desk.

Nothing here can mirror a letter, and nothing guesses: the window is laid out in
desk space, which is the arm's own -- +x to its right as it reaches out, straight
out -y -- so image right along +x and image down along +y is the desk seen from
above. `camera.json` only says where the frame shows each desk point, and the
window takes each of its pixels from there: wherever the camera stands, a
calibration that fits it gives the face as seen from above, never its mirror.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from maxarm import Camera

from ..tiles import TILE_HEIGHT_MM, Tile

# Sampled this much finer than asked, then averaged down: the frame has 2.5 to 5 px a millimetre.
SUPERSAMPLE = 4


def side_deg(tile: Tile, quarter: int) -> float:
    """The desk direction the window's image right runs along, degrees from desk +x."""
    return tile.turn + 90.0 * quarter


def upright_deg(tile: Tile, quarter: int) -> float:
    """Where the top of a letter upright in quarter turn `quarter` points, desk degrees, -180 to 180."""
    return wrap_deg(side_deg(tile, quarter) - 90.0)


def wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def window(frame: np.ndarray, camera: Camera, tile: Tile, quarter: int, window_mm: float, size: int,
           grey: bool = True) -> np.ndarray:
    """The face, `window_mm` square round its centre, `size` pixels square, at quarter turn `quarter`."""
    fine = size * SUPERSAMPLE
    mm_per_px = window_mm / fine
    side = math.radians(side_deg(tile, quarter))
    right = np.array([math.cos(side), math.sin(side)]) * mm_per_px
    down = np.array([-math.sin(side), math.cos(side)]) * mm_per_px
    corner = np.array([tile.x, tile.y]) - (right + down) * (fine - 1) / 2
    window_to_desk = np.array([[right[0], down[0], corner[0]], [right[1], down[1], corner[1]], [0.0, 0.0, 1.0]])
    window_to_frame = camera.plane_to_pixel(TILE_HEIGHT_MM) @ window_to_desk
    image = cv2.warpPerspective(frame, window_to_frame, (fine, fine), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE)
    if grey and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def quarter_with(sign_right: float, sign_down: float) -> int:
    """The quarter turn whose window shows quarter turn 0's (right, down) direction at its bottom right."""
    right, down = sign_right, sign_down
    for quarter in range(4):
        if right > 0 and down > 0:
            return quarter
        right, down = down, -right           # the next quarter turn's window, the same point
    raise ValueError("a direction on a window's diagonal has a sign each way")
