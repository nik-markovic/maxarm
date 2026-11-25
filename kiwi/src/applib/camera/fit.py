"""The camera, fitted to the three cubes: how the camera sees them, and where the arm says they stand.

    fit = fit_camera(frame, {"red": (x, y), "green": (x, y), "blue": (x, y)})   # desk mm
    fit.desk_to_pixel                                                           # 3x4, for camera.json

The cubes' shapes make the camera; the arm's readings only lay the arm on it. A
40 mm cube seen from above is a small frame of known edges, and three of them,
each turned however it was set down, fix a square-pixel camera: focal length,
principal point and pose, in true millimeters. A turn, a shift and a stretch
from the arm's points (six numbers, which three points fix exactly) leave each
cube wherever its shape puts it: which blob is which cube, and the camera, come
of the pictures alone. Then the arm is laid on that camera by a turn and a shift,
the board's mirror (it negates x) taken out, its millimeters as the library makes
them true (`maxarm/raw.py`): where the cubes stand against where the arm read
them is the arm's error there, not hidden in a bent camera.

1. **Start.** For each naming of the blobs (`cubes.py`), and focal lengths across
   any plausible lens, the camera that puts each cube's middle on its blob (P3P),
   scored by how much of each blob the cube drawn there covers.
2. **Outlines.** The cubes' silhouettes fitted to the blobs' outlines, whatever is
   ragged about a blob counting little.
3. **Edges.** Each visible edge of each cube moved onto the strongest color edge
   beside it (`edges.py`). An outline counts only where the cube's color is on
   its inner side: a shadow's edge, or the desk's grain, has the desk on both sides.
   The camera is fitted to the bottom and vertical edges only: the top's are found
   inside the cube, and a top looking small puts the camera further off than it is,
   reading the desk long. Where a cube stands, all its edges.
4. **The arm, roughly.** The camera stays as the cubes' edges made it -- reality, the
   40 mm cubes its only length (bending it onto the arm's readings tilted the
   cubes and moved the base) -- and the arm is laid on it by
   the turn and shift that best put the cubes' centers on the arm's readings, no
   stretch. That is only near the arm: CALIBRATE then lays it again by the base
   the frame shows and the readings' bearings, and keeps what is left as the arm's
   correction (`calibration/camera/anchor.py`); this lay stays as `cubes_alone`,
   the desk space the base and the points are found in. `desk_to_true` is the
   mirror, `true_to_pixel` the camera along the arm's x and y so laid, and
   `desk_to_pixel` names the desk point the arm must be sent to.
5. **The cubes as seen.** Each cube where its own edges put it through that camera:
   what the check image draws, and how far each arm reading is from its cube --
   the arm's own error there, since the camera is not bent to hide it. The
   stretch a turn, shift and stretch would take between the two is kept as the
   arm's measure against the camera (`arm_stretch`, `stretch_of()`): shown, not used.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import cubes, edges
from .cubes import COLORS, Blob, CubesNotFound
from .model import (BASE_DIAGONALS, CUBE_MM, Edge, Pinhole, around, camera_position, cube_corners, hidden_edges,
                    least_squares, project, sample, to_polygon, visible_edges)

Points = Dict[str, Tuple[float, float]]   # desk x, y under each cube's center

# The board negates x: the arm's x, y with z up are left-handed, the camera's world is not.
HANDEDNESS = np.diag([-1.0, 1.0])

# The parameters, in one vector: focal length and principal point in pixels; the
# camera's rotation and shift; each cube's turn; a stretch from the arm's x, y to
# where the cubes stand (symmetric: its turn is the camera's), which leaves each
# cube where its shape puts it, and is kept: the arm's measure against the camera's.
FOCAL, CENTER = 0, slice(1, 3)
ROTATION, SHIFT, TURNS, STRETCH = slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15)
STEPS = np.array([0.5, 0.25, 0.25] + [1e-5] * 3 + [0.01] * 3 + [1e-5] * 3 + [1e-6] * 3)
# A cube placed on its own: where it stands, true mm, and its turn, radians.
CUBE_STEPS = np.array([0.01, 0.01, 1e-5])

# Start: focal lengths from a long lens to a very wide one, as multiples of the
# frame's width; the overlap below which no naming is taken for the cubes.
FOCAL_RANGE = (0.3, 3.0)
FOCAL_TRIES = 48
TURN_TRIES_DEG = range(0, 90, 10)
MIN_OVERLAP = 0.6
# Pixels: how far a point counts fully, in the outline and edge fits.
OUTLINE_HUBER_PX = 3.0
EDGE_HUBER_PX = 1.0
# The outline fit is done twice: the second time without the outline points further from the
# cubes than this many times the typical one (a glossy desk's tint below a cube, say).
OUTLINE_KEEP = 3.0
OUTLINE_SAMPLES = 180
# The edge search: how far across each edge it looks, as shares of the cube's size
# in the frame, round by round.
REACH = (0.04, 0.03, 0.02, 0.02, 0.02)
# The principal point is held loosely to the frame's middle, by this share of its width.
CENTER_SPREAD = 0.1
# A stretch or a skew beyond these, between the arm's points and the cubes as the
# camera sees them, says a cube is not where the arm recorded it: moved since, or
# recorded off center.
MAX_STRETCH = 1.12
MAX_SKEW_DEG = 6.0
# Fewer edge points than this, all cubes together, and the cubes' edges were not followed.
MIN_EDGE_POINTS = 60


class CameraNotFound(Exception):
    """The cubes were found, but no camera explains them where the arm says they stand."""


class CubesDoNotFit(CameraNotFound):
    """The cubes' camera was found, but the arm's readings do not lay on the cubes: `fit` is the camera the cubes'
    edges make, `apart` each cube's distance from where the other two lay its reading (mm) and that place (true x, y
    in `fit`), `worst` the cube furthest."""

    def __init__(self, problem: str, fit: "CameraFit", apart: Dict[str, Tuple[float, np.ndarray]]) -> None:
        self.fit, self.apart = fit, apart
        self.worst = max(apart, key=lambda color: apart[color][0])
        super().__init__(f"the {self.worst} cube stands {apart[self.worst][0]:.0f} mm from where the arm recorded it, "
                         f"against the other two ({problem})")


@dataclass(frozen=True)
class Wireframe:
    """A cube as the camera sees it: its corners in pixels, the edges it shows, and those its faces hide."""
    corners: np.ndarray              # 8 x 2, as model.cube_corners() orders them
    seen: List[Edge]
    hidden: List[Tuple[int, int]]

    def lines(self, pairs) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [(self.corners[start], self.corners[end]) for start, end in pairs]

    @property
    def seen_lines(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return self.lines((edge.start, edge.end) for edge in self.seen)

    @property
    def hidden_lines(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """The hidden edges, and the bottom's diagonals: an X under the cube's center."""
        return self.lines([*self.hidden, *BASE_DIAGONALS])


@dataclass
class CameraFit:
    true_to_pixel: np.ndarray        # 3x4: true mm (x, y, z up) to pixels
    desk_to_true: np.ndarray         # 2x2: the arm's desk x, y to true x, y (z is the same): the board's mirror
                                     # and the arm's stretch
    cubes: Dict[str, dict]           # color: desk x, y, the cube's turn, degrees, and the pixel under its center
    frame_size: Tuple[int, int]
    edge_px: Tuple[float, float]     # how far the cubes' edges sit from the frame's: median, 90%
    edges_used: int
    # Refined by the red cube's points (refine.py): how many, and how far off they are, median and worst, mm;
    # and the calibration as the three cubes alone gave it: its two matrices and its cubes. None for the cubes' own.
    refined: Optional[dict] = None
    cubes_alone: Optional[dict] = None
    # As CALIBRATE fits it, not kept in camera.json: each cube's turn, degrees, as its edges' corners are numbered,
    # and the edges found round it in the frame -- for the refine points (store.cube_refine_points()).
    cube_edges: Optional[Dict[str, Tuple[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]]] = None
    # 2x2: the stretch and mirror a turn, shift and stretch would lay the arm on the cubes with: the arm's measure
    # against the camera's millimeters, shown, not used. None in a file written before the arm was laid without it.
    arm_stretch: Optional[np.ndarray] = None
    # How the arm was laid on the camera by its base and the points' bearings (calibration/camera/anchor.py): its axis
    # and turn in `cubes_alone`'s desk space, the base found, the report. None: laid by the readings alone.
    anchor: Optional[dict] = None

    @property
    def desk_to_pixel(self) -> np.ndarray:
        """3x4: desk (x, y, z) to pixels, the arm placed on the desk as fitted."""
        lift = np.eye(4)
        lift[:2, :2] = self.desk_to_true
        matrix = self.true_to_pixel @ lift
        return matrix / np.linalg.norm(matrix[2, :3])

    @property
    def desk_to_shares(self) -> np.ndarray:
        """3x4: desk (x, y, z) to shares of the frame's width and height, pixel centers as `desk_to_pixel` has them:
        what a page draws by, at any size of the same picture."""
        width, height = self.frame_size
        shares = np.array([[1 / width, 0.0, 0.5 / width], [0.0, 1 / height, 0.5 / height], [0.0, 0.0, 1.0]])
        return shares @ self.desk_to_pixel

    @property
    def measured_stretch(self) -> np.ndarray:
        """How the arm's x and y measure against the camera's millimeters: `arm_stretch`, or for a file written
        before it, the stretch the arm was laid with."""
        return self.desk_to_true if self.arm_stretch is None else self.arm_stretch

    @property
    def camera_position(self) -> np.ndarray:
        """Where the camera stands, in desk mm."""
        true = camera_position(self.true_to_pixel)
        return np.array([*np.linalg.solve(self.desk_to_true, true[:2]), true[2]])

    @property
    def focal(self) -> float:
        intrinsic = cv2.decomposeProjectionMatrix(self.true_to_pixel)[0]
        return float(abs(intrinsic[0, 0] / intrinsic[2, 2]))

    def seen_center(self, color: str) -> np.ndarray:
        """True x, y of the desk under the cube's center, where the camera saw it."""
        x, y, w = np.linalg.solve(self.true_to_pixel[:, [0, 1, 3]], (*self.cubes[color]["pixel"], 1.0))
        return np.array([x / w, y / w])

    def cube_off_mm(self, color: str) -> float:
        """How far from the arm's point for the cube the calibration puts the cube it saw, desk mm."""
        cube = self.cubes[color]
        return float(math.dist(np.linalg.solve(self.desk_to_true, self.seen_center(color)), (cube["x"], cube["y"])))

    def wireframe(self, color: str) -> "Wireframe":
        """A cube in the frame, standing where the camera saw it and turned as fitted."""
        center = self.seen_center(color)
        turn = math.radians(self.cubes[color]["turn_deg"])
        seen = visible_edges(center, turn, camera_position(self.true_to_pixel))
        return Wireframe(project(self.true_to_pixel, cube_corners(center, turn)), seen, hidden_edges(seen))

    def cube_fit(self) -> "CameraFit":
        """The calibration as the three cubes alone gave it: itself, unless refined."""
        if self.cubes_alone is None:
            return self
        alone = self.cubes_alone
        return CameraFit(np.array(alone["true_to_pixel"]), np.array(alone["desk_to_true"]), alone["cubes"],
                         self.frame_size, self.edge_px, self.edges_used,
                         arm_stretch=_matrix_or_none(alone.get("arm_stretch")))

    def to_json(self) -> dict:
        refinement = {"refined": self.refined} if self.refined else {}
        alone = {"cubes_alone": self.cubes_alone} if self.cubes_alone is not None else {}
        anchor = {"anchor": self.anchor} if self.anchor is not None else {}
        stretch = {} if self.arm_stretch is None else {"arm_stretch": self.arm_stretch.tolist()}
        return {"true_to_pixel": self.true_to_pixel.tolist(), "desk_to_true": self.desk_to_true.tolist(),
                **stretch, "cubes": self.cubes, "frame_size": list(self.frame_size),
                "edge_px": list(self.edge_px), "edges_used": self.edges_used, **refinement, **alone, **anchor}

    @classmethod
    def from_json(cls, stored: dict) -> "CameraFit":
        true_to_pixel, desk_to_true = np.array(stored["true_to_pixel"]), np.array(stored["desk_to_true"])
        cubes = {color: cube if "pixel" in cube else _with_pixel(cube, true_to_pixel, desk_to_true)
                 for color, cube in stored["cubes"].items()}
        return cls(true_to_pixel, desk_to_true, cubes, tuple(stored["frame_size"]), tuple(stored["edge_px"]),
                   stored["edges_used"], stored.get("refined"), stored.get("cubes_alone"),
                   arm_stretch=_matrix_or_none(stored.get("arm_stretch")), anchor=stored.get("anchor"))


def _matrix_or_none(stored: Optional[list]) -> Optional[np.ndarray]:
    return None if stored is None else np.array(stored)


def _with_pixel(cube: dict, true_to_pixel: np.ndarray, desk_to_true: np.ndarray) -> dict:
    """The cube's entry with the pixel under its center, on the desk, for a file written before it was kept: there
    the cubes' own fit put each cube exactly where the arm recorded it."""
    center = desk_to_true @ (cube["x"], cube["y"])
    pixel = project(true_to_pixel, np.array([[*center, 0.0]]))[0]
    return {**cube, "pixel": [round(float(pixel[0]), 3), round(float(pixel[1]), 3)]}


def measured_stretch(desk: np.ndarray, true: np.ndarray) -> np.ndarray:
    """The stretch (symmetric) and the board's mirror that a turn, shift and stretch would lay the arm's desk x, y
    rows on the `true` rows with, least squares: the arm's measure against the camera's millimeters, shown, not used
    to lay it. Three rows fit it exactly. Raises ValueError if no stretch does it without turning the mirror over:
    the rows in a line, or not the arm's."""
    desk, true = np.asarray(desk, float), np.asarray(true, float)
    to_desk = np.linalg.lstsq(np.column_stack((true, np.ones(len(true)))), desk, rcond=None)[0]
    linear = np.linalg.inv(to_desk[:2].T)                         # true = linear @ desk + shift
    left, scales, right = np.linalg.svd(linear @ HANDEDNESS)
    if np.linalg.det(left @ right) < 0:
        raise ValueError("the points lay the arm's x and y on the camera's mirrored")
    return right.T @ np.diag(scales) @ right @ HANDEDNESS


def stretch_of(desk_to_true: np.ndarray) -> Tuple[float, float, float]:
    """How the arm's x and y measure against the camera's millimeters: the least and most a desk distance is
    stretched, and how far from square the axes are laid, degrees."""
    scales = np.linalg.svd(desk_to_true, compute_uv=False)
    return float(scales.min()), float(scales.max()), abs(_skew_deg(desk_to_true))


def stretch_problem(desk_to_true: np.ndarray) -> Optional[str]:
    """What is wrong with a stretch beyond MAX_STRETCH or MAX_SKEW_DEG, or None."""
    low, high, skew = stretch_of(desk_to_true)
    if low < 1 / MAX_STRETCH or high > MAX_STRETCH or skew > MAX_SKEW_DEG:
        return (f"as the camera sees them the arm's points are {low:.2f} to {high:.2f} times as far apart, and "
                f"askew by {skew:.1f} degrees")
    return None


def arm_measure(desk: Sequence[Sequence[float]], seen: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """The arm's measure against the camera: the stretch a turn, shift and stretch would lay the arm's readings on
    the cubes as seen with (`measured_stretch()`), or None if none lays them without turning the mirror over."""
    try:
        return measured_stretch(desk, seen)
    except (ValueError, np.linalg.LinAlgError):
        return None


def fit_camera(frame: np.ndarray, points: Points) -> CameraFit:
    """The camera that sees the cubes in `frame` standing where `points` says. Raises if there is none."""
    blobs = cubes.find_blobs(frame)
    starts = [(*_start(frame, naming, points), naming) for naming in cubes.namings(blobs)]
    if not starts:
        raise CubesNotFound(f"the frame shows {cubes.describe(blobs)}; it needs a red, a green and a blue "
                            f"cube, whole and in view")
    overlap, params, naming = max(starts, key=lambda start: start[0])
    if overlap < MIN_OVERLAP:
        raise CameraNotFound(f"the cubes seen do not stand where the arm recorded them (best overlap "
                             f"{overlap:.2f}): was a cube moved after RECORD?")
    params = _fit_outlines(naming, points, params)
    params, _, found = _fit_edges(frame, naming, points, params)
    problem = stretch_problem(_desk_to_true(params))
    if problem:
        fit = _result(params, points, frame, found)
        raise CubesDoNotFit(problem, fit, _apart(fit, points))
    return _result(params, points, frame, found)


def _apart(fit: "CameraFit", points: Points) -> Dict[str, Tuple[float, np.ndarray]]:
    """Each cube's distance from where the other two cubes lay its reading -- a turn and shift putting their readings
    on them as seen -- and that place, true x, y: the cube recorded wrong is the furthest."""
    apart = {}
    for color in COLORS:
        others = [other for other in COLORS if other != color]
        arm = np.array([HANDEDNESS @ np.asarray(points[other], float) for other in others])
        seen = np.array([fit.seen_center(other) for other in others])
        left, _, right = np.linalg.svd((seen - seen.mean(axis=0)).T @ (arm - arm.mean(axis=0)))
        turn = left @ right
        if np.linalg.det(turn) < 0:
            turn = left @ np.diag([1.0, -1.0]) @ right
        laid = turn @ (HANDEDNESS @ np.asarray(points[color], float) - arm.mean(axis=0)) + seen.mean(axis=0)
        apart[color] = (float(np.linalg.norm(laid - fit.seen_center(color))), laid)
    return apart


# --- the parameters -------------------------------------------------------

def _stretch(params: np.ndarray) -> np.ndarray:
    xx, yy, xy = params[STRETCH]
    return np.array([[xx, xy], [xy, yy]])


def _desk_to_true(params: np.ndarray) -> np.ndarray:
    return _stretch(params) @ HANDEDNESS


def _skew_deg(matrix: np.ndarray) -> float:
    """How far from square a 2x2 lays the arm's x and y axes, degrees."""
    x_axis, y_axis = matrix[:, 0], matrix[:, 1]
    cosine = x_axis @ y_axis / (np.linalg.norm(x_axis) * np.linalg.norm(y_axis))
    return 90.0 - math.degrees(math.acos(cosine))


def _pinhole(params: np.ndarray) -> Pinhole:
    return Pinhole(params[FOCAL], tuple(params[CENTER]), params[ROTATION], params[SHIFT])


def _centers(params: np.ndarray, points: Points) -> List[np.ndarray]:
    to_true = _desk_to_true(params)
    return [to_true @ points[color] for color in COLORS]


def _corners(params: np.ndarray, points: Points) -> List[np.ndarray]:
    """Each cube's 8 corners in the frame, in COLORS order."""
    camera = _pinhole(params)
    return [camera.project(cube_corners(center, turn))
            for center, turn in zip(_centers(params, points), params[TURNS])]


def _result(params: np.ndarray, points: Points, frame: np.ndarray, found: list) -> CameraFit:
    """The cubes' camera with the arm laid on it (step 4), each cube where its own edges put it (step 5) -- the pixel
    under its center and its turn -- and the stretch between the arm's readings and those cubes kept as the arm's
    measure; with the edges `found` round each cube, numbered as the cube was turned when they were found."""
    turn, shift = _lay_arm(params, points)
    laid = np.eye(4)                                  # the arm's true x, y (as laid) to the cubes' own
    laid[:2, :2], laid[:2, 3] = turn, shift
    true_to_pixel = _pinhole(params).matrix @ laid
    turned = math.atan2(turn[1, 0], turn[0, 0])
    seen, distances = _seen_cubes(params, points, found)
    seen = [np.array([*(turn.T @ (cube[:2] - shift)), cube[2] - turned]) for cube in seen]
    pixels = project(true_to_pixel, np.array([(*cube[:2], 0.0) for cube in seen]))
    cubes_seen = {color: {"x": float(points[color][0]), "y": float(points[color][1]),
                           "turn_deg": round(math.degrees(cube[2]) % 90.0, 2),
                           "pixel": [round(float(pixel[0]), 3), round(float(pixel[1]), 3)]}
                  for color, cube, pixel in zip(COLORS, seen, pixels)}
    cube_edges = {color: (round(math.degrees(cube_turn - turned), 4), found_on)
                  for color, cube_turn, found_on in zip(COLORS, params[TURNS], found)}
    return CameraFit(true_to_pixel, HANDEDNESS.copy(), cubes_seen, (frame.shape[1], frame.shape[0]),
                     (float(np.median(np.abs(distances))), float(np.percentile(np.abs(distances), 90))),
                     len(distances), cube_edges=cube_edges,
                     arm_stretch=arm_measure([points[color] for color in COLORS], [cube[:2] for cube in seen]))


# --- 1. the start ---------------------------------------------------------

def _start(frame: np.ndarray, naming: Dict[str, Blob], points: Points) -> Tuple[float, np.ndarray]:
    """The best overlap the cubes get, and the camera that gives it, the arm's x and y taken as true."""
    height, width = frame.shape[:2]
    scale = min(1.0, 640.0 / width)
    masks = [cv2.resize(naming[color].mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) > 0
             for color in COLORS]
    params = np.zeros(15)
    params[CENTER], params[STRETCH] = (width / 2, height / 2), (1.0, 1.0, 0.0)
    middles = np.array([(*center, CUBE_MM / 2) for center in _centers(params, points)])
    blobs = np.array([naming[color].centroid for color in COLORS])
    best = (-1.0, params)
    for focal in np.geomspace(*FOCAL_RANGE, FOCAL_TRIES) * width:
        intrinsic = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
        _, rotations, shifts = cv2.solveP3P(middles, blobs, intrinsic, None, flags=cv2.SOLVEPNP_P3P)
        for rotation, shift in zip(rotations, shifts):
            trial = params.copy()
            trial[FOCAL], trial[ROTATION], trial[SHIFT] = focal, rotation.ravel(), shift.ravel()
            camera = _pinhole(trial)
            if camera.position[2] <= CUBE_MM:
                continue
            overlap = 0.0
            for index, (mask, center) in enumerate(zip(masks, _centers(trial, points))):
                turn_overlaps = [(edges.overlap(camera.project(cube_corners(center, math.radians(turn))) * scale,
                                                mask), math.radians(turn)) for turn in TURN_TRIES_DEG]
                cover, trial[TURNS.start + index] = max(turn_overlaps)
                overlap += cover / len(COLORS)
            if overlap > best[0]:
                best = (overlap, trial)
    return best


# --- 2. the outlines ------------------------------------------------------

def _fit_outlines(naming: Dict[str, Blob], points: Points, params: np.ndarray) -> np.ndarray:
    """The cubes' silhouettes onto the blobs' outlines, both ways: outline to silhouette and back.

    Twice: the second time without the outline points the first left far off.
    """
    outlines = []
    for color in COLORS:
        blob = naming[color]
        outline = blob.contour.reshape(-1, 2).astype(float)
        outline = outline[::max(1, len(outline) // 240)]
        box, distance = edges.distance_to_outline(blob)
        outlines.append([outline, box, distance])

    def residuals(trial: np.ndarray) -> np.ndarray:
        parts = []
        for corners, (outline, box, distance) in zip(_corners(trial, points), outlines):
            silhouette = cv2.convexHull(corners.astype(np.float32)).reshape(-1, 2).astype(float)
            parts.append(to_polygon(outline, silhouette))
            parts.append(sample(distance, around(silhouette, OUTLINE_SAMPLES) - box))
        return np.concatenate(parts)

    free = [index for index in range(len(params)) if index not in (CENTER.start, CENTER.stop - 1)]
    for _ in range(2):
        params = least_squares(residuals, params, STEPS, huber=OUTLINE_HUBER_PX, free=free, iterations=60)[0]
        for corners, entry in zip(_corners(params, points), outlines):
            silhouette = cv2.convexHull(corners.astype(np.float32)).reshape(-1, 2).astype(float)
            off = to_polygon(entry[0], silhouette)
            entry[0] = entry[0][off <= max(OUTLINE_HUBER_PX, OUTLINE_KEEP * np.median(off))]
    return params


# --- 3. the edges ---------------------------------------------------------

def _fit_edges(frame: np.ndarray, naming: Dict[str, Blob], points: Points,
               params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list]:
    """Every visible edge onto the frame's edges, round after round as the search narrows; and each cube's edges
    found in the last round."""
    desk = cubes.desk_chroma(frame)
    crops = [edges.crop(frame, naming[color], desk) for color in COLORS]
    center = np.array([frame.shape[1] / 2, frame.shape[0] / 2])
    distances = np.zeros(0)
    for reach in REACH:
        found = [edges.search(crop, corners, seen, reach) for crop, corners, seen in
                 zip(crops, _corners(params, points), _edges_seen(params, points))]
        if sum(len(targets) for _, _, targets in found) < MIN_EDGE_POINTS:
            raise CameraNotFound("the cubes' edges could not be followed in the frame: is the light even, and "
                                 "each cube clear of other things?")

        fitted = [without_top(found_on) for found_on in found]

        def residuals(trial: np.ndarray) -> np.ndarray:
            parts = [edges.signed_distances(corners, *found_on) for corners, found_on in
                     zip(_corners(trial, points), fitted)]
            parts.append((trial[CENTER] - center) / (CENTER_SPREAD * frame.shape[1]))
            return np.concatenate(parts)

        params, distances, _ = least_squares(residuals, params, STEPS, huber=EDGE_HUBER_PX, iterations=30)
        distances = distances[:-2]
    return params, distances, found


def without_top(found_on: tuple) -> tuple:
    """The edges found on a cube but those of its top face (corners 4 to 7, `cube_corners()`)."""
    starts, ends, targets = found_on
    keep = (starts < 4) | (ends < 4)
    return starts[keep], ends[keep], targets[keep]


def _edges_seen(params: np.ndarray, points: Points) -> List[List[Edge]]:
    camera = _pinhole(params).position
    return [visible_edges(center, turn, camera) for center, turn in zip(_centers(params, points), params[TURNS])]


# --- 4. the arm -------------------------------------------------------------

def _lay_arm(params: np.ndarray, points: Points) -> Tuple[np.ndarray, np.ndarray]:
    """The turn (2x2) and shift that best lay the arm's true x, y (its readings, mirrored) on the cubes' centers as
    the cubes' camera has them, least squares, no stretch: cubes' = turn @ arm's + shift."""
    arm = np.array([HANDEDNESS @ np.asarray(points[color], float) for color in COLORS])
    cubes_at = np.array(_centers(params, points))
    arm_mean, cubes_mean = arm.mean(axis=0), cubes_at.mean(axis=0)
    left, _, right = np.linalg.svd((cubes_at - cubes_mean).T @ (arm - arm_mean))
    turn = left @ right
    if np.linalg.det(turn) < 0:
        raise CameraNotFound("the arm's points lie on the cubes only mirrored: were two cubes swapped?")
    return turn, cubes_mean - turn @ arm_mean


# --- 5. the cubes as seen ---------------------------------------------------

def _seen_cubes(params: np.ndarray, points: Points, found: list) -> Tuple[List[np.ndarray], np.ndarray]:
    """Each cube where its own edges put it through the camera, the arm aside: x, y and turn, true mm; and how far
    the edges found lie from the cubes so placed, pixels."""
    camera = _pinhole(params)
    seen, distances = [], []
    for center, turn, found_on in zip(_centers(params, points), params[TURNS], found):
        def misses(trial: np.ndarray, found_on: tuple = found_on) -> np.ndarray:
            return edges.signed_distances(camera.project(cube_corners(trial[:2], trial[2])), *found_on)

        cube, missed, _ = least_squares(misses, np.array([*center, turn]), CUBE_STEPS, huber=EDGE_HUBER_PX,
                                        iterations=30)
        seen.append(cube)
        distances.append(missed)
    return seen, np.concatenate(distances)
