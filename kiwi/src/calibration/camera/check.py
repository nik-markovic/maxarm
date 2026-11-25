"""The check image: a frame with the calibration drawn on it, to see by eye whether it sits right.

- The cubes, where the calibration sees them: their visible edges, a pixel wide,
  yellow and half see-through, so the cube's own edge shows beside them; the edges
  their faces hide, and an X across each one's bottom under its center, fainter. On a frame taken later, a
  cube standing anywhere else, or a camera nudged since, shows as the outline off
  its cube.
- The arm's base, where the calibration puts it -- round the arm's axis, turned as
  the arm's x and y are -- its sizes `applib/camera/base.py`'s: the four
  suction cups' rims, 41 mm across, flush on the desk, in cyan; the screw nuts at
  their centers on a 120 mm square round the arm's axis -- the hex, 7 mm across its
  flats, from the plate's top 10 mm up to 13, and the rounded tip, 5 mm across
  narrowing to 4, up to 18 -- in magenta; the base plate's four straight central
  edges, 124.5 mm apart, 10 mm up, in white. CALIBRATE sets the axis by the base's
  nuts and the turn by the points' bearings, so the drawing sits off the base by
  as much as those two disagree (`anchor.py`'s report). Away from the cubes, this
  is the one check of the calibration the arm always brings along.
- Desk coordinates: a dot every 50 mm within reach, labeled every 100 mm.
- The reach on the desk: 95 mm in orange, 270 mm in red, across the arm's fan.
- No text over the picture: the page says what the calibration is and
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

from applib.camera import base
from applib.camera.cubes import COLORS, Blob, is_warm
from applib.camera.fit import CameraFit
from applib.camera.model import CUBE_MM, project

YELLOW = (63, 210, 255)                 # BGR, as OpenCV draws: every cube's wireframe, as the overlay draws it
PINK = (255, 0, 255)                    # the blobs that could have been cubes, when no calibration comes of a frame
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
OUTLINE_OPACITY = 0.5                   # the cubes' outlines, a pixel wide: the cube's own edge shows beside them
HIDDEN_OPACITY = 0.25                   # the edges a cube hides, and the X under its center: there, not in the way
DOT_STEP_MM = 50
LABEL_STEP_MM = 100
# The MaxArm's base (applib/camera/base.py), mm, round the arm's axis.
BASE_CUP_SPACING_MM, BASE_CUP_RADIUS_MM = base.CUP_SPACING_MM, base.BASE_SUCTION_CUP_RADIUS_MM
PLATE_TOP_MM, NUT_TOP_MM, NUT_HALF_MM = base.PLATE_TOP_MM, base.NUT_TOP_MM, base.NUT_HALF_MM
TIP_BOTTOM_HALF_MM, TIP_TOP_HALF_MM, TIP_TOP_MM, TIP_MM = (base.TIP_BOTTOM_HALF_MM, base.TIP_TOP_HALF_MM,
                                                           base.TIP_TOP_MM, base.TIP_MM)
PLATE_EDGE_MM, PLATE_EDGE_HALF_MM = base.PLATE_EDGE_MM, base.PLATE_EDGE_HALF_MM
MAGENTA = (255, 0, 255)
LEGEND = (f"yellow: the cubes as seen, fainter where hidden, X under the center   cyan: the base's suction cups   "
          f"orange, red: the reach on the desk, {REACH_MIN_MM:.0f} and {REACH_MAX_MM:.0f} mm   "
          f"dots: every {DOT_STEP_MM} mm")


def draw(frame: np.ndarray, fit: CameraFit, caption: Sequence[str]) -> np.ndarray:
    """`frame` with the calibration drawn on it. `caption` is not drawn: kept for the callers, it says what the page says."""
    image = frame.copy()
    width = image.shape[1]
    line = max(1, round(width / 1300))
    matrix = fit.desk_to_pixel
    for radius, color in ((REACH_MIN_MM, ORANGE), (REACH_MAX_MM, RED)):
        _polyline(image, matrix, _arc(radius), color, 2 * line)
    _draw_base(image, matrix, line)
    wireframes = [fit.wireframe(cube) for cube in COLORS]
    _desk_dots(image, matrix, line, [cv2.convexHull(cube.corners.astype(np.float32)) for cube in wireframes])
    for opacity, lines in ((HIDDEN_OPACITY, [cube.hidden_lines for cube in wireframes]),
                           (OUTLINE_OPACITY, [cube.seen_lines for cube in wireframes])):
        drawn = image.copy()
        for start, end in (pair for cube in lines for pair in cube):
            cv2.line(drawn, _pixel(start), _pixel(end), YELLOW, 1, cv2.LINE_AA)
        image = cv2.addWeighted(drawn, opacity, image, 1 - opacity, 0)
    return image


def mark(image: np.ndarray, fit: CameraFit, color: str, label: str, recorded: Optional[np.ndarray] = None,
         caption: Sequence[str] = ()) -> np.ndarray:
    """The cube that does not fit, made plain: its outline in thick red, `label` beside it in large letters, a red
    cross where the arm's reading puts it (`recorded`, true x, y in `fit`, at the cube's top) with an arrow from the
    cube to it; and `caption` across the top."""
    image = image.copy()
    line = max(1, round(image.shape[1] / 1300))
    cube = fit.wireframe(color)
    for start, end in cube.seen_lines + cube.hidden_lines:
        cv2.line(image, _pixel(start), _pixel(end), RED, 3 * line, cv2.LINE_AA)
    top = project(fit.true_to_pixel, np.array([[*fit.seen_center(color), CUBE_MM]]))[0]
    if recorded is not None:
        there = project(fit.true_to_pixel, np.array([[*recorded, CUBE_MM]]))[0]
        cv2.arrowedLine(image, _pixel(top), _pixel(there), RED, 2 * line, cv2.LINE_AA, tipLength=0.3)
        cv2.drawMarker(image, _pixel(there), RED, cv2.MARKER_CROSS, 28 * line, 3 * line, cv2.LINE_AA)
    if caption:
        _caption(image, caption)
    corners = cube.corners
    scale = image.shape[1] / 1600
    thickness = max(2, round(2.5 * scale))
    (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = min(corners[:, 0].max() + 20 * line, image.shape[1] - width - 10)
    y = corners[:, 1].min() + height
    if y - height < 110 * scale:                     # under the caption's band: below the cube instead
        x, y = max(10, min(corners[:, 0].min(), image.shape[1] - width - 10)), corners[:, 1].max() + height + 20 * line
    origin = _pixel((x, y))
    cv2.rectangle(image, (origin[0] - 8, origin[1] - height - 10), (origin[0] + width + 8, origin[1] + 12),
                  (0, 0, 0), -1)
    cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, RED, thickness, cv2.LINE_AA)
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


def _circle(center: Tuple[float, float], radius: float, height: float = 0.0) -> np.ndarray:
    """A circle `height` over the desk, closed."""
    turns = np.linspace(0.0, 2 * math.pi, 73)
    return np.column_stack((center[0] + radius * np.cos(turns), center[1] + radius * np.sin(turns),
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


def _polyline(image: np.ndarray, matrix: np.ndarray, points: np.ndarray, color, thickness: int) -> None:
    """Desk points joined up, in pieces, leaving out what is behind the camera."""
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    in_front = homogeneous[:, 2] > 0
    pixels = homogeneous[:, :2] / np.where(in_front, homogeneous[:, 2], 1.0)[:, None]
    for start in range(len(points) - 1):
        if in_front[start] and in_front[start + 1]:
            cv2.line(image, _pixel(pixels[start]), _pixel(pixels[start + 1]), color, thickness, cv2.LINE_AA)


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
