"""The check image: a frame with the calibration drawn on it, to see by eye whether it sits right.

- The cubes, where the calibration sees them: their visible edges, a pixel wide,
  yellow and half see-through, so the cube's own edge shows beside them; the edges
  their faces hide, and an X across each one's bottom under its centre, fainter. On a frame taken later, a
  cube standing anywhere else, or a camera nudged since, shows as the outline off
  its cube.
- The arm's base, where the calibration puts it, its sizes the owner's
  (work/PILOT-kiwi-idea.md): the four suction cups' rims, 42 mm across, flush on the
  desk, in cyan; the screw nuts at their centres on a 120 mm square round the arm's
  axis -- the hex, 7 mm across its flats, from the plate's top 10 mm up to 13, and
  the rounded tip, 5 mm across narrowing to 4, up to 18 -- in magenta; the base
  plate's four straight central edges, 124.5 mm apart, 10 mm up, in white. Away
  from the cubes, this is the one check of the calibration the arm always brings
  along.
- Desk coordinates: a dot every 50 mm within reach, labelled every 100 mm.
- The reach on the desk: 95 mm in orange, 270 mm in red, across the arm's fan.
- No text over the picture (the owner): the page says what the calibration is and
  how it was checked, so nothing covers an edge.

When no calibration comes of a frame, `draw_blobs()` shows what was found in it
instead: every blob that could have been a cube, warm or cool.
"""

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from maxarm.geometry import BASE_FAN_DEG, bearing_degrees
from maxarm.path import REACH_MAX_MM, REACH_MIN_MM

from applib.camera.cubes import COLOURS, Blob, is_warm
from applib.camera.fit import CameraFit
from applib.camera.model import project

YELLOW = (63, 210, 255)                 # BGR, as OpenCV draws: every cube's wireframe, as the overlay draws it
PINK = (255, 0, 255)                    # the blobs that could have been cubes, when no calibration comes of a frame
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
OUTLINE_OPACITY = 0.5                   # the cubes' outlines, a pixel wide: the cube's own edge shows beside them
HIDDEN_OPACITY = 0.25                   # the edges a cube hides, and the X under its centre: there, not in the way
DOT_STEP_MM = 50
LABEL_STEP_MM = 100
# The MaxArm's base: a suction cup at each corner of a square round the arm's axis, its sides along the arm's x and y.
# The arm's base (the owner, work/PILOT-kiwi-idea.md), mm, round the arm's axis, sides along its x and y.
BASE_CUP_SPACING_MM = 120.0       # the screw nuts' centres, and the cups' on them
BASE_CUP_RADIUS_MM = 21.0         # 162 mm across the outer rims, flush on the desk
PLATE_TOP_MM = 10.0
NUT_TOP_MM = 13.0                 # the hex, 3 mm tall on the plate
NUT_HALF_MM = 3.5                 # 7 mm across the flats
TIP_BOTTOM_HALF_MM, TIP_TOP_HALF_MM, TIP_TOP_MM, TIP_MM = 2.5, 2.0, 17.0, 18.0
PLATE_EDGE_MM = 124.5             # the plate's straight central edges apart
PLATE_EDGE_HALF_MM = 28.0         # each straight for this either side of its middle
MAGENTA = (255, 0, 255)
LEGEND = (f"yellow: the cubes as seen, fainter where hidden, X under the centre   cyan: the base's suction cups   "
          f"orange, red: the reach on the desk, {REACH_MIN_MM:.0f} and {REACH_MAX_MM:.0f} mm   "
          f"dots: every {DOT_STEP_MM} mm")


def draw(frame: np.ndarray, fit: CameraFit, caption: Sequence[str]) -> np.ndarray:
    """`frame` with the calibration drawn on it. `caption` is not drawn: kept for the callers, it says what the page says."""
    image = frame.copy()
    width = image.shape[1]
    line = max(1, round(width / 1300))
    matrix = fit.desk_to_pixel
    for radius, colour in ((REACH_MIN_MM, ORANGE), (REACH_MAX_MM, RED)):
        _polyline(image, matrix, _arc(radius), colour, 2 * line)
    _draw_base(image, matrix, line)
    wireframes = [fit.wireframe(cube) for cube in COLOURS]
    _desk_dots(image, matrix, line, [cv2.convexHull(cube.corners.astype(np.float32)) for cube in wireframes])
    for opacity, lines in ((HIDDEN_OPACITY, [cube.hidden_lines for cube in wireframes]),
                           (OUTLINE_OPACITY, [cube.seen_lines for cube in wireframes])):
        drawn = image.copy()
        for start, end in (pair for cube in lines for pair in cube):
            cv2.line(drawn, _pixel(start), _pixel(end), YELLOW, 1, cv2.LINE_AA)
        image = cv2.addWeighted(drawn, opacity, image, 1 - opacity, 0)
    return image


def draw_blobs(frame: np.ndarray, blobs: List[Blob], caption: Sequence[str]) -> np.ndarray:
    """`frame` with every blob that could have been a cube outlined and marked warm or cool."""
    image = frame.copy()
    line = max(1, round(image.shape[1] / 1300))
    for blob in blobs:
        cv2.drawContours(image, [blob.contour], -1, PINK, 2 * line, cv2.LINE_AA)
        _text(image, "warm" if is_warm(blob) else "cool", blob.centroid, 0.5 * line + 0.2)
    _caption(image, caption)
    return image


def frame_size_problem(frame: np.ndarray, fit: CameraFit) -> Optional[str]:
    """Why the calibration cannot be used on `frame`, if it cannot: it holds at one frame size only."""
    size = (frame.shape[1], frame.shape[0])
    if tuple(fit.frame_size) == size:
        return None
    return (f"this frame is {size[0]} x {size[1]}, the calibration was fitted at "
            f"{fit.frame_size[0]} x {fit.frame_size[1]}: CALIBRATE again")


def difference_mm(first: CameraFit, second: CameraFit) -> Tuple[float, float]:
    """How far apart two calibrations put the same pixels on the desk, within reach: median and worst, mm."""
    desk = np.array([(x, y, 0.0) for x in range(-280, 281, 20) for y in range(-280, 281, 20)
                     if REACH_MIN_MM <= math.hypot(x, y) <= REACH_MAX_MM and _in_fan(x, y)], float)
    pixels = project(first.desk_to_pixel, desk)
    width, height = first.frame_size
    seen = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    if not seen.any():
        return math.nan, math.nan
    there = np.array([_to_desk(second.desk_to_pixel, pixel) for pixel in pixels[seen]])
    apart = np.linalg.norm(there - desk[seen, :2], axis=1)
    return float(np.median(apart)), float(apart.max())


def _arc(radius: float) -> np.ndarray:
    """Desk points `radius` from the arm's axis, across its fan."""
    bearings = np.radians(np.linspace(-BASE_FAN_DEG, BASE_FAN_DEG, 241))
    return np.column_stack((radius * np.sin(bearings), -radius * np.cos(bearings), np.zeros(len(bearings))))


def _draw_base(image: np.ndarray, matrix: np.ndarray, line: int) -> None:
    """The arm's base: cups' rims, screw nuts and their tips, the plate's central edges (PLATE_*, NUT_*, TIP_*)."""
    half = BASE_CUP_SPACING_MM / 2
    for corner in ((half, half), (half, -half), (-half, -half), (-half, half)):
        _polyline(image, matrix, _circle(corner, BASE_CUP_RADIUS_MM), CYAN, line)
        for radius, height in ((NUT_HALF_MM, PLATE_TOP_MM), (NUT_HALF_MM, NUT_TOP_MM),
                               (TIP_BOTTOM_HALF_MM, NUT_TOP_MM), (TIP_TOP_HALF_MM, TIP_TOP_MM)):
            _polyline(image, matrix, _circle(corner, radius, height), MAGENTA, 1)
        _polyline(image, matrix, np.array([(*corner, NUT_TOP_MM), (*corner, TIP_MM)]), MAGENTA, 1)
    edge, along = PLATE_EDGE_MM / 2, PLATE_EDGE_HALF_MM
    for a, b in (((edge, -along), (edge, along)), ((-edge, -along), (-edge, along)),
                 ((-along, edge), (along, edge)), ((-along, -edge), (along, -edge))):
        _polyline(image, matrix, np.array([(*a, PLATE_TOP_MM), (*b, PLATE_TOP_MM)]), WHITE, 1)


def _circle(centre: Tuple[float, float], radius: float, height: float = 0.0) -> np.ndarray:
    """A circle `height` over the desk, closed."""
    turns = np.linspace(0.0, 2 * math.pi, 73)
    return np.column_stack((centre[0] + radius * np.cos(turns), centre[1] + radius * np.sin(turns),
                            np.full(len(turns), height)))


def _desk_dots(image: np.ndarray, matrix: np.ndarray, line: int, hidden: List[np.ndarray]) -> None:
    """A dot every 50 mm within reach, but where a cube hides the desk."""
    for x in range(-300, 301, DOT_STEP_MM):
        for y in range(-300, 301, DOT_STEP_MM):
            if not (REACH_MIN_MM <= math.hypot(x, y) <= REACH_MAX_MM and _in_fan(x, y)):
                continue
            pixel = project(matrix, np.array([[x, y, 0.0]]))[0]
            if not _is_inside(image, pixel) or any(
                    cv2.pointPolygonTest(outline, _pixel(pixel), False) >= 0 for outline in hidden):
                continue
            cv2.circle(image, _pixel(pixel), 2 * line, WHITE, -1, cv2.LINE_AA)


def _polyline(image: np.ndarray, matrix: np.ndarray, points: np.ndarray, colour, thickness: int) -> None:
    """Desk points joined up, in pieces, leaving out what is behind the camera."""
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    in_front = homogeneous[:, 2] > 0
    pixels = homogeneous[:, :2] / np.where(in_front, homogeneous[:, 2], 1.0)[:, None]
    for start in range(len(points) - 1):
        if in_front[start] and in_front[start + 1]:
            cv2.line(image, _pixel(pixels[start]), _pixel(pixels[start + 1]), colour, thickness, cv2.LINE_AA)


def _caption(image: np.ndarray, lines: Sequence[str], at_bottom: bool = False) -> None:
    """The lines across the top, or the bottom, on a dark band so they read over anything; a line too long for the
    image goes on in the next."""
    scale = image.shape[1] / 1800
    size = (0.75 if at_bottom else 0.9) * scale
    lines = [part for line in lines for part in _wrapped(line, size, image.shape[1] - 24 * scale)]
    height = int(34 * scale) + 4
    depth = height * len(lines) + height // 2
    top = image.shape[0] - depth if at_bottom else 0
    band = image[top:top + depth]
    band[:] = (band * 0.35).astype(np.uint8)
    for row, text in enumerate(lines):
        _text(image, text, np.array([12 * scale, top + (row + 1) * height]), size)


def _wrapped(line: str, scale: float, width: float) -> List[str]:
    """`line` as lines no wider than `width` pixels, written at `scale`, broken between words."""
    thickness = max(1, round(2 * scale))
    parts, current = [], ""
    for word in line.split(" "):
        trial = f"{current} {word}" if current else word
        if current and cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] > width:
            parts.append(current)
            trial = word
        current = trial
    return parts + [current]


def _text(image: np.ndarray, text: str, at: np.ndarray, scale: float) -> None:
    thickness = max(1, round(2 * scale))
    origin = _pixel(at)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, thickness, cv2.LINE_AA)


def _in_fan(x: float, y: float) -> bool:
    return abs(bearing_degrees(x, y)) <= BASE_FAN_DEG


def _is_inside(image: np.ndarray, pixel: np.ndarray) -> bool:
    return 0 <= pixel[0] < image.shape[1] and 0 <= pixel[1] < image.shape[0]


def _pixel(point) -> Tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def _to_desk(matrix: np.ndarray, pixel: np.ndarray) -> np.ndarray:
    """The desk point (height 0) a pixel sees."""
    plane = matrix[:, [0, 1, 3]]
    x, y, w = np.linalg.solve(plane, (pixel[0], pixel[1], 1.0))
    return np.array([x / w, y / w])
