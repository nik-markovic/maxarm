"""Locate the 40 mm calibration cubes and report where each one's base sits.

A cube is 40 mm tall, so the middle of its coloured blob floats well above the
table and is useless for calibration. What maps onto the table plane is the
centre of the cube's *base*. It is never visible -- the cube stands on it -- so it
is constructed from the silhouette, in four steps:

1. **Silhouette.** A fixed Lab chroma window finds *which* blob is which cube. The
   outline itself is then re-cut per cube, splitting the pixels around it between
   that cube's own chroma and the desk's, both measured in this frame. A fixed
   window cannot do that job: on the owner's camera it cut through green's shaded
   face and took in the cyan light blue throws on the desk.
2. **Hexagon.** A cube showing three faces silhouettes as a hexagon. It is the
   smallest hexagon enclosing the outline, so a genuinely short edge survives,
   then each edge is re-fitted through the boundary pixels along it.
3. **Which reading.** The same hexagon is two cubes -- the Necker cube -- and at
   the 40-50 degrees the owner's camera looks down, a receding edge is as steep
   as a vertical one, so no rule on the outline can tell them apart. The image
   can: of the two, the right one puts the interior edges where the faces change
   shade. Each reading's three faces are cut out and the one whose faces are each
   more uniform wins; `reading_margin` is by how much, and a cube the shading
   cannot decide is not reported.
4. **Corners and centres.** Opposite hexagon edges are images of parallel cube
   edges, so each pair meets at a vanishing point, and from those the hidden far
   base corner and the interior near top corner are line intersections. The
   centres are where the diagonals cross. All of that is exact under full
   perspective and needs no camera model.

Segmentation uses Lab chroma rather than HSV hue. Lit faces blow out to V=254,
which makes hue meaningless, but a* and b* barely move between a cube's lit and
shadowed faces.

The history, and the failure that made step 3 necessary, is in
`work/STATUS-condor.md` §10.6 and `work/STATUS-condor-prototype.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

CUBE_MM = 40.0

# Windows in Lab chroma, as (a_min, a_max, b_min, b_max), that pick out each
# cube's blob. They only decide which blob is which: the outline that gets
# measured is re-cut from this frame's own colours (`_silhouette`). Wood sits at
# about a*=132, b*=135.
#   red   - the cube's own shadow on the wood reaches a*=147, nearly as red as
#           the lit face, but stays wood-coloured on b* while the cube is warmer.
#   green - the arm's anodised body matches the green cube on a* almost exactly
#           and is only separable by being markedly yellower.
#   blue  - the cube is closer to cyan than blue, so it sits low on b*.
COLOUR_WINDOWS = {
    "red": (155, 255, 138, 255),
    "green": (0, 115, 115, 134),
    "blue": (0, 125, 0, 115),
}

# The arm's own anodised body is the detector's oldest hazard: it matches the
# green cube on chroma closely enough to win on size, and it has taken a green
# detection over in practice. It is also *dark* -- L*=39 measured, against 153
# to 207 for the three cubes -- and that is a far wider margin than any chroma
# bound gives.
#
# Judged on the blob's average, not per pixel: a cube's own edges and shadowed
# corners run down to L*=20, so a pixel-level floor eats the silhouette.
MIN_BLOB_LIGHTNESS = 90.0

# How many times more shade variation the flipped reading leaves inside its faces
# than the chosen one. Under this the shading does not say which reading is
# right -- faces lit alike (1.00 rendered), a cube face-on (1.03) -- and the base
# centre would be a coin toss, so the cube is not reported. On the owner's 45
# degree camera real cubes read 4.9-8.3. Under flat light at a shallower angle
# (`agenttools/camtest/shots`, three cameras) many read 1.0-1.3, so there this
# refuses cubes whose faces are lit too alike to read -- deliberately, since
# missing is loud and a flipped reading is not.
#
# It is *not* a test of whether a blob is a cube: the arm and other blobs at the
# frame edge scored up to 2.7. As a gate at 2.0 it rejected real green cubes and
# fell through to a smaller green-window blob -- the blue cube's shaded
# underside -- which is why the detector no longer tries a second blob once one
# has been read.
MIN_READING_MARGIN = 1.3

MIN_CONTOUR_AREA = 1500

OVERLAY_BGR = (255, 0, 255)  # magenta: distinct from all three cubes and the wood
OVERLAY_ALPHA = 0.55

Point = tuple[float, float]


@dataclass(frozen=True)
class CubeDetection:
    colour: str
    base_corners: tuple[Point, Point, Point]  # left, near, right -- on the silhouette
    top_corners: tuple[Point, Point, Point]   # left, far, right -- on the silhouette
    far_base: Point                           # hidden behind the cube
    near_top: Point                           # where the three visible faces meet
    silhouette_center: Point
    hexagon: list[Point]
    reading_margin: float = math.inf         # see MIN_READING_MARGIN

    @property
    def base_center(self) -> Point:
        """Where the base's diagonals cross, which is its centre under any perspective."""
        left, near, right = self.base_corners
        return _crossing((left, right), (near, self.far_base))

    @property
    def top_center(self) -> Point:
        """The centre of the cube's top face, one cube-height above `base_center`.

        Same construction as the base centre, on the other end of the vertical
        edges. It is the same x and y as the base centre in the arm's world and a
        cube's height above it, which makes the pair of them the only thing in the
        scene that says how the image moves with height.
        """
        left, far, right = self.top_corners
        return _crossing((left, right), (self.near_top, far))

    @property
    def base_edge_px(self) -> tuple[float, float]:
        left, near, right = self.base_corners
        return math.dist(left, near), math.dist(near, right)

    @property
    def elevation_deg(self) -> float:
        """Rough camera elevation, from how squashed the shorter base edge is.

        Only exact when the cube sits square-on to the camera. A cube turned
        about its vertical axis shortens both base edges, which inflates this
        figure, so treat it as an upper bound.
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
        """Return (visible_edges, hidden_edges) for all twelve cube edges."""
        top_l, far_top, top_r = self.top_corners
        bot_l, near_bot, bot_r = self.base_corners
        near_top, far_bot = self.near_top, self.far_base
        visible = [
            (top_l, far_top), (far_top, top_r), (top_r, near_top), (near_top, top_l),
            (bot_l, near_bot), (near_bot, bot_r),
            (top_l, bot_l), (top_r, bot_r), (near_top, near_bot),
        ]
        hidden = [
            (bot_r, far_bot), (far_bot, bot_l), (far_top, far_bot),
        ]
        return visible, hidden


def _homogeneous(point: Point) -> np.ndarray:
    return np.array([point[0], point[1], 1.0])


def _meet(first: tuple, second: tuple) -> np.ndarray:
    """Where two lines, each through two points, cross -- in homogeneous form.

    Homogeneous so that parallel lines meet too, at a point at infinity: a
    vanishing point of two edges that are parallel in the image is exactly that,
    and every construction below goes through one unchanged.
    """
    line_a = np.cross(*(p if len(p) == 3 else _homogeneous(p) for p in first))
    line_b = np.cross(*(p if len(p) == 3 else _homogeneous(p) for p in second))
    return np.cross(line_a, line_b)


def _crossing(first: tuple, second: tuple) -> Point:
    x, y, w = _meet(first, second)
    return (float(x / w), float(y / w))


def _segment(image_lab: np.ndarray, colour: str) -> np.ndarray:
    a_min, a_max, b_min, b_max = COLOUR_WINDOWS[colour]
    a, b = image_lab[:, :, 1], image_lab[:, :, 2]
    mask = ((a >= a_min) & (a <= a_max) & (b >= b_min) & (b <= b_max)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def _silhouette(image_lab: np.ndarray, blob: np.ndarray) -> np.ndarray:
    """Re-cut a cube's outline from its own colour and the desk's, in this frame.

    Every pixel near the blob is put on a line in a*b* from the desk's median
    chroma (a ring well outside the blob) to the cube's (the blob's core), and
    goes to whichever end it is nearer. Both ends are measured, so a camera that
    shifts every colour moves them together.
    """
    x, y, width, height = cv2.boundingRect(blob)
    size = max(width, height)
    pad = int(0.3 * size)
    rows, cols = image_lab.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(cols, x + width + pad), min(rows, y + height + pad)
    chroma = image_lab[y0:y1, x0:x1, 1:].astype(np.float64)

    inside = np.zeros(chroma.shape[:2], np.uint8)
    cv2.drawContours(inside, [blob], -1, 255, cv2.FILLED, offset=(-x0, -y0))
    core = cv2.erode(inside, np.ones((max(3, size // 20),) * 2, np.uint8)) > 0
    ring = cv2.dilate(inside, np.ones((pad, pad), np.uint8)) == 0
    if not core.any() or not ring.any():
        return blob
    cube, desk = np.median(chroma[core], axis=0), np.median(chroma[ring], axis=0)
    towards_cube = cube - desk
    along = (chroma - desk) @ towards_cube / (towards_cube @ towards_cube)

    # Where to cut between desk and cube is read off this frame's own histogram
    # (Otsu), not fixed halfway: a face catching glare off its own surface is
    # pale -- 0.4 of the way to the cube's colour on the owner's desk -- while
    # the desk's reflection of the cube stays under 0.25.
    scaled = (np.clip(along, 0.0, 1.0) * 255).astype(np.uint8)
    cut, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (scaled > cut).astype(np.uint8) * 255
    # Scaled to the cube: this is what cuts off thin shadow spikes and fills the
    # speckle in a washed-out face, on a near cube and a far one alike.
    kernel = np.ones((max(3, size // 30) | 1,) * 2, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return blob

    def overlap(contour: np.ndarray) -> int:
        region = np.zeros_like(mask)
        cv2.drawContours(region, [contour], -1, 255, cv2.FILLED)
        return cv2.countNonZero(region & inside)

    return max(contours, key=overlap) + np.array([x0, y0])


def _hexagon(contour: np.ndarray) -> list[Point] | None:
    """The smallest hexagon enclosing the silhouette, edges re-fitted to it.

    Starts from the convex hull at pixel resolution and removes one edge at a
    time, extending its two neighbours to meet, always the edge whose removal adds
    least area. Unlike simplifying until six vertices survive, this cannot cut a
    corner off: a short edge of the real cube stays because removing it would add
    a lot of area, and a rounded corner goes because it adds almost none.
    """
    hull = cv2.approxPolyDP(cv2.convexHull(contour), 2.0, True)
    polygon = [(float(p[0][0]), float(p[0][1])) for p in hull]
    if len(polygon) < 6:
        return None
    while len(polygon) > 6:
        count = len(polygon)
        best, best_area, best_corner = -1, math.inf, None
        for i in range(count):
            before, start = polygon[i - 1], polygon[i]
            end, after = polygon[(i + 1) % count], polygon[(i + 2) % count]
            x, y, w = _meet((before, start), (end, after))
            if abs(w) < 1e-9:
                continue
            corner = (x / w, y / w)
            # The neighbours have to meet beyond this edge, not behind it.
            if (np.dot(np.subtract(corner, start), np.subtract(start, before)) < 0
                    or np.dot(np.subtract(corner, end), np.subtract(end, after)) < 0):
                continue
            (ax, ay), (bx, by) = np.subtract(start, corner), np.subtract(end, corner)
            area = abs(ax * by - ay * bx) / 2
            if area < best_area:
                best, best_area, best_corner = i, area, corner
        if best < 0:
            return None
        polygon[best] = best_corner
        del polygon[(best + 1) % count]
    return _fit_edges(contour, polygon) or polygon


def _fit_edges(contour: np.ndarray, hexagon: list[Point]) -> list[Point] | None:
    """Re-derive the six corners by fitting lines to the silhouette edges.

    The enclosing hexagon is decided by the outermost pixels, so it sits a pixel
    or two proud of the edge. Each edge instead gets a line fitted through the
    boundary pixels along it, and the corners come from intersecting adjacent
    lines.
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
        shorter = min(math.dist(hexagon[i], hexagon[i - 1]), math.dist(hexagon[i], hexagon[(i + 1) % 6]))
        if math.dist(tuple(corner), hexagon[i]) > 0.3 * shorter:
            return None
        corners.append((float(corner[0]), float(corner[1])))
    return corners


def _reading(hexagon: list[Point], near: int) -> dict[str, Point]:
    """Every corner of the cube, if hexagon vertex `near` is the near base corner.

    Around the hexagon from there: a base corner, the top corner above it, the far
    top corner, the top corner on the other side, and the base corner below it.
    Opposite hexagon edges are images of parallel cube edges and meet at their
    vanishing point; the two corners that are not on the silhouette follow from
    those, the far base one from the base edges and the near top one from the top
    edges.
    """
    near_bot, right, right_top, far_top, left_top, left = (
        hexagon[(near + step) % 6] for step in range(6))
    towards_right = _meet((near_bot, right), (far_top, left_top))
    towards_left = _meet((right_top, far_top), (left, near_bot))
    corners = {"near_bot": near_bot, "right": right, "right_top": right_top,
               "far_top": far_top, "left_top": left_top, "left": left}
    corners["far_bot"] = _crossing((left, towards_right), (right, towards_left))
    corners["near_top"] = _crossing((left_top, towards_left), (right_top, towards_right))
    return corners


def _face_spread(image: np.ndarray, corners: dict[str, Point]) -> float:
    """How much the shade varies inside each visible face, on average over all three.

    A cube's faces are each close to one flat shade and different from each
    other, so the right reading scores low. The flipped reading's faces straddle
    the real ones and score several times higher. Each face is shrunk a little
    first, so a blurred edge counts for neither.
    """
    faces = [
        ("near_top", "right_top", "far_top", "left_top"),
        ("near_top", "near_bot", "right", "right_top"),
        ("near_top", "left_top", "left", "near_bot"),
    ]
    squared, count = 0.0, 0
    for face in faces:
        quad = np.array([corners[name] for name in face])
        middle = quad.mean(axis=0)
        region = np.zeros(image.shape[:2], np.uint8)
        cv2.fillConvexPoly(region, (middle + 0.8 * (quad - middle)).astype(np.int32), 255)
        pixels = image[region > 0].astype(np.float64)
        if len(pixels):
            squared += float(((pixels - pixels.mean(axis=0)) ** 2).sum())
            count += len(pixels)
    return squared / count if count else math.inf


def _read_cube(image: np.ndarray, hexagon: list[Point]) -> tuple[dict[str, Point], float]:
    """The reading of the hexagon that the face shading supports, and by how much.

    The near base corner is the lowest of its alternate triple of vertices -- the
    other two are top corners, and a camera above the cube sees those higher up.
    That leaves one candidate per triple: the Necker pair.
    """
    candidates = []
    for first in (0, 1):
        near = max((first, first + 2, first + 4), key=lambda i: hexagon[i][1])
        corners = _reading(hexagon, near)
        candidates.append((_face_spread(image, corners), corners))
    (best, corners), (other, _) = sorted(candidates, key=lambda c: c[0])
    return corners, other / best if best > 0 else math.inf


def find_cube(image_lab: np.ndarray, image: np.ndarray, colour: str) -> CubeDetection | None:
    mask = _segment(image_lab, colour)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Largest first, but skip anything too dark -- that is what rejects the arm's
    # green body, which matches the green cube on colour.
    for blob in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(blob) < MIN_CONTOUR_AREA:
            break
        if _mean_lightness(image_lab, blob) < MIN_BLOB_LIGHTNESS:
            continue
        silhouette = _silhouette(image_lab, blob)
        hexagon = _hexagon(silhouette)
        if hexagon is None:
            continue
        corners, margin = _read_cube(image, hexagon)
        if margin < MIN_READING_MARGIN:
            return None
        left, right = sorted((corners["left"], corners["right"]))
        top_l, top_r = (corners["left_top"], corners["right_top"])
        if left != corners["left"]:
            top_l, top_r = top_r, top_l
        moments = cv2.moments(silhouette)
        return CubeDetection(
            colour=colour,
            base_corners=(left, corners["near_bot"], right),
            top_corners=(top_l, corners["far_top"], top_r),
            far_base=corners["far_bot"],
            near_top=corners["near_top"],
            silhouette_center=(moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
            hexagon=hexagon,
            reading_margin=margin,
        )
    return None


def _mean_lightness(image_lab: np.ndarray, contour: np.ndarray) -> float:
    region = np.zeros(image_lab.shape[:2], dtype=np.uint8)
    cv2.drawContours(region, [contour], -1, 255, cv2.FILLED)
    return float(cv2.mean(image_lab[:, :, 0], mask=region)[0])


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
        # The base's diagonals, dashed like the hidden edges: the base centre is
        # where they cross, and drawing them shows why it sits where it does --
        # at a steep camera that is well up the front face, not on its bottom edge.
        left, near, right = det.base_corners
        for start, end in hidden + [(left, right), (near, det.far_base)]:
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
