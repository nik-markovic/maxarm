"""Locate the 40 mm calibration cubes and report where each one's base sits.

A cube is 40 mm tall, so the middle of its coloured blob floats about 20 mm above
the table. At the shallow angle this camera uses that projects to roughly 40 mm of
apparent displacement, so the blob centre is useless for calibration. What maps
onto the table plane is the centre of the cube's *base*, which is what this
computes, from the two base corners that sit on the silhouette.

Segmentation uses Lab chroma rather than HSV hue. Lit faces blow out to V=254,
which makes hue meaningless, but a* and b* barely move between a cube's lit and
shadowed faces.

This was `agenttools/find-cube.py`, the prototype detector, and the code is
unchanged. It moved here because `calibrate.py` is built on it and agenttools is
the part of the tree that gets thrown away between pilots; the tool is still
there, as the command-line front end that draws the overlay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

CUBE_MM = 40.0

# Windows in Lab chroma, as (a_min, a_max, b_min, b_max). Wood sits at about
# a*=132, b*=135, and every window needs bounding on both axes:
#   red   - the cube's own shadow on the wood reaches a*=147, nearly as red as
#           the lit face, but stays wood-coloured on b* while the cube is warmer.
#   green - the arm's anodised body matches the green cube on a* almost exactly
#           and is only separable by being markedly yellower.
#   blue  - the cube is closer to cyan than blue, so it sits low on b*.
COLOUR_WINDOWS = {
    # a* is deliberately well clear of the shadow's 147: loosening it to 145
    # moves the reported base centre by 20 px, because the extra pixels are the
    # soft shadow boundary rather than the cube.
    "red": (155, 255, 138, 255),
    "green": (0, 115, 115, 134),
    "blue": (0, 125, 0, 115),
}

OVERLAY_BGR = (255, 0, 255)  # magenta: distinct from all three cubes and the wood
OVERLAY_ALPHA = 0.55

# A cube showing three faces has a hexagonal silhouette. Sweep the polygon
# tolerance until exactly six vertices survive.
POLY_EPSILON_STEPS = (0.010, 0.015, 0.020, 0.025, 0.030, 0.040)
MIN_CONTOUR_AREA = 1500

Point = tuple[float, float]


@dataclass(frozen=True)
class CubeDetection:
    colour: str
    base_center: Point
    base_corners: tuple[Point, Point, Point]  # left, near, right
    top_corners: tuple[Point, Point, Point]   # left, far, right
    silhouette_center: Point
    hexagon: list[Point]

    @property
    def top_center(self) -> Point:
        """The centre of the cube's top face, one cube-height above `base_center`.

        Same construction as the base centre -- the midpoint of the two
        diagonally opposite corners -- on the other end of the two vertical
        edges. It is the same x and y as the base centre in the arm's world and a
        cube's height above it, which makes the pair of them the only thing in
        the scene that says how the image moves with height.
        """
        left, _, right = self.top_corners
        return ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)

    @property
    def base_edge_px(self) -> tuple[float, float]:
        left, near, right = self.base_corners
        return math.dist(left, near), math.dist(near, right)

    @property
    def elevation_deg(self) -> float:
        """Rough camera elevation, from how squashed the shorter base edge is.

        Only exact when the cube sits square-on to the camera. A cube turned
        about its vertical axis shortens both base edges, which inflates this
        figure, so treat it as an upper bound. Use the least of several cubes,
        or read it off a circular target instead, where there is no yaw to
        confound it.
        """
        across, receding = sorted(self.base_edge_px, reverse=True)
        return math.degrees(math.asin(min(1.0, receding / across)))

    @property
    def is_square_on(self) -> bool:
        """Whether the cube is turned far enough to spoil the derived figures."""
        across, receding = sorted(self.base_edge_px, reverse=True)
        return receding / across < 0.5

    @property
    def mm_per_px(self) -> float:
        """Millimetres per pixel across the view, where the table is least squashed."""
        return CUBE_MM / max(self.base_edge_px)

    @property
    def mm_per_px_receding(self) -> float:
        """Millimetres per pixel away from the camera, where a pixel costs most."""
        return CUBE_MM / min(self.base_edge_px)

    def ground_offset_mm(self, point: Point) -> float:
        """Distance on the table between the base centre and some image point.

        The two image axes carry very different amounts of table, so they cannot
        share one scale factor.
        """
        across = (point[0] - self.base_center[0]) * self.mm_per_px
        receding = (point[1] - self.base_center[1]) * self.mm_per_px_receding
        return math.hypot(across, receding)

    def wireframe(self) -> tuple[list[tuple[Point, Point]], list[tuple[Point, Point]]]:
        """Return (visible_edges, hidden_edges) for all twelve cube edges.

        Six corners are measured off the silhouette. The other two -- the near
        top corner where the three visible faces meet, and the back bottom
        corner -- are inferred by stepping along the cube's vertical direction,
        so treat them as indicative rather than measured.
        """
        top_l, far_top, top_r = self.top_corners
        bot_l, near_bot, bot_r = self.base_corners
        rise = (
            ((top_l[0] - bot_l[0]) + (top_r[0] - bot_r[0])) / 2,
            ((top_l[1] - bot_l[1]) + (top_r[1] - bot_r[1])) / 2,
        )
        near_top = (near_bot[0] + rise[0], near_bot[1] + rise[1])
        far_bot = (far_top[0] - rise[0], far_top[1] - rise[1])

        visible = [
            (top_l, far_top), (far_top, top_r), (top_r, near_top), (near_top, top_l),
            (bot_l, near_bot), (near_bot, bot_r),
            (top_l, bot_l), (top_r, bot_r), (near_top, near_bot),
        ]
        hidden = [
            (bot_r, far_bot), (far_bot, bot_l), (far_top, far_bot),
        ]
        return visible, hidden


def _segment(image_lab: np.ndarray, colour: str) -> np.ndarray:
    a_min, a_max, b_min, b_max = COLOUR_WINDOWS[colour]
    a, b = image_lab[:, :, 1], image_lab[:, :, 2]
    mask = ((a >= a_min) & (a <= a_max) & (b >= b_min) & (b <= b_max)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def _fit_edges(contour: np.ndarray, hexagon: list[Point]) -> list[Point] | None:
    """Re-derive the six corners by fitting lines to the silhouette edges.

    Corners taken straight from the convex hull are decided by a handful of
    extreme pixels, so they move several millimetres when the colour threshold
    shifts. Each edge instead gets a line fitted through hundreds of boundary
    pixels, and the corners come from intersecting adjacent lines.
    """
    points = contour.reshape(-1, 2).astype(np.float64)
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(6):
        start = np.array(hexagon[i])
        end = np.array(hexagon[(i + 1) % 6])
        span = end - start
        length = float(np.linalg.norm(span))
        if length < 12:
            return None
        along = span / length
        across = np.array([-along[1], along[0]])
        offsets = (points - start) @ along
        distances = np.abs((points - start) @ across)
        # Drop the corner regions: they curve, and they are what we are solving for.
        selected = (offsets > 0.2 * length) & (offsets < 0.8 * length)
        selected &= distances < max(3.0, 0.04 * length)
        if int(selected.sum()) < 10:
            return None
        vx, vy, x0, y0 = cv2.fitLine(
            points[selected].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
        ).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))

    corners: list[Point] = []
    for i in range(6):
        origin_a, direction_a = lines[(i - 1) % 6]
        origin_b, direction_b = lines[i]
        matrix = np.column_stack((direction_a, -direction_b))
        if abs(np.linalg.det(matrix)) < 1e-9:
            return None
        steps = np.linalg.solve(matrix, origin_b - origin_a)
        corner = origin_a + steps[0] * direction_a
        # A refined corner that runs away from the hull corner means the edge
        # selection went wrong; fall back rather than report nonsense.
        if math.dist(tuple(corner), hexagon[i]) > 0.3 * float(np.linalg.norm(np.array(hexagon[(i + 1) % 6]) - np.array(hexagon[i]))):
            return None
        corners.append((float(corner[0]), float(corner[1])))
    return corners


def _hexagon(contour: np.ndarray) -> list[Point] | None:
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for epsilon in POLY_EPSILON_STEPS:
        poly = cv2.approxPolyDP(hull, epsilon * perimeter, True)
        if len(poly) != 6:
            continue
        hexagon = [(float(p[0][0]), float(p[0][1])) for p in poly]
        return _fit_edges(contour, hexagon) or hexagon
    return None


def _corners(hexagon: list[Point]) -> tuple[tuple[Point, Point, Point], tuple[Point, Point, Point]]:
    """Split the hexagon into its three base corners and three top corners.

    The cube's vertical edges project to the steepest lines in the image, and two
    of them lie on the silhouette as an opposite pair. Their lower ends are two
    diagonally opposite base corners; the silhouette vertex between them is the
    corner nearest the camera, and the one opposite that is the far top corner.
    """
    def steepness(index: int) -> float:
        (x0, y0), (x1, y1) = hexagon[index], hexagon[(index + 1) % 6]
        return abs(y1 - y0) / (abs(x1 - x0) + 1e-6)

    # Opposite edges of a projected cube are parallel, so score them in pairs.
    pair = max(range(3), key=lambda i: steepness(i) + steepness(i + 3))
    lower: list[int] = []
    upper: list[int] = []
    for index in (pair, pair + 3):
        start, end = index, (index + 1) % 6
        if hexagon[start][1] > hexagon[end][1]:
            lower.append(start)
            upper.append(end)
        else:
            lower.append(end)
            upper.append(start)

    # The two remaining vertices: one sits between the base corners, one between
    # the top corners.
    remaining = [i for i in range(6) if i not in lower and i not in upper]
    near, far = sorted(remaining, key=lambda i: -hexagon[i][1])

    base = (hexagon[lower[0]], hexagon[near], hexagon[lower[1]])
    top = (hexagon[upper[0]], hexagon[far], hexagon[upper[1]])
    return base, top


def find_cube(image_lab: np.ndarray, image: np.ndarray, colour: str) -> CubeDetection | None:
    mask = _segment(image_lab, colour)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Largest first, but skip anything that is not cube-shaped -- that is what
    # rejects the arm's green body, which matches the green cube on colour.
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            break
        hexagon = _hexagon(contour)
        if hexagon is None:
            continue
        base, top = _corners(hexagon)
        left, _, right = base
        moments = cv2.moments(contour)
        return CubeDetection(
            colour=colour,
            base_center=((left[0] + right[0]) / 2, (left[1] + right[1]) / 2),
            base_corners=base,
            top_corners=top,
            silhouette_center=(moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
            hexagon=hexagon,
        )
    return None


def find_cubes(image: np.ndarray, colours: list[str]) -> list[CubeDetection]:
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    found = (find_cube(image_lab, image, colour) for colour in colours)
    return [det for det in found if det is not None]


def _dashed(layer: np.ndarray, start: Point, end: Point, colour, thickness: int) -> None:
    length = math.dist(start, end)
    if length < 1:
        return
    step = 10.0 / length
    position = 0.0
    while position < 1.0:
        tail = min(position + step, 1.0)
        p0 = (int(start[0] + (end[0] - start[0]) * position), int(start[1] + (end[1] - start[1]) * position))
        p1 = (int(start[0] + (end[0] - start[0]) * tail), int(start[1] + (end[1] - start[1]) * tail))
        cv2.line(layer, p0, p1, colour, thickness, cv2.LINE_AA)
        position += step * 2


def annotate(image: np.ndarray, detections: list[CubeDetection]) -> np.ndarray:
    layer = image.copy()
    for det in detections:
        visible, hidden = det.wireframe()
        for start, end in visible:
            cv2.line(layer, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])),
                     OVERLAY_BGR, 2, cv2.LINE_AA)
        for start, end in hidden:
            _dashed(layer, start, end, OVERLAY_BGR, 1)
        for corner in det.base_corners:
            cv2.drawMarker(layer, (int(corner[0]), int(corner[1])), OVERLAY_BGR,
                           cv2.MARKER_TILTED_CROSS, 14, 2)

    blended = cv2.addWeighted(layer, OVERLAY_ALPHA, image, 1 - OVERLAY_ALPHA, 0)

    # Labels and the base centre go on at full strength -- they are the readout,
    # not the overlay.
    for det in detections:
        bx, by = int(det.base_center[0]), int(det.base_center[1])
        cv2.drawMarker(blended, (bx, by), (255, 255, 255), cv2.MARKER_CROSS, 20, 3)
        cv2.drawMarker(blended, (bx, by), OVERLAY_BGR, cv2.MARKER_CROSS, 20, 1)
        label = f"{det.colour} base ({bx},{by})"
        cv2.putText(blended, label, (bx - 60, by + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(blended, label, (bx - 60, by + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return blended
