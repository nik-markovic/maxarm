"""The camera, fitted to the three cubes: how the camera sees them, and where the arm says they stand.

    fit = fit_camera(frame, {"red": (x, y), "green": (x, y), "blue": (x, y)})   # desk mm
    fit.desk_to_pixel                                                           # 3x4, for camera.json

The cubes' shapes make the camera. A 40 mm cube seen from above is a small
frame of known edges, and three of them, each turned however it was set down,
fix a square-pixel camera: focal length, principal point and pose, in true
millimetres. Where the arm says the cubes stand then fixes how the arm's own x
and y lie on the desk: a turn, a mirror (the board negates x) and a slight
stretch -- the arm's x/y error. `desk_to_pixel` takes it in, so a pixel names the
desk point the arm must be sent to.

1. **Start.** For each naming of the blobs (`cubes.py`), and focal lengths across
   any plausible lens, the camera that puts each cube's middle on its blob (P3P),
   scored by how much of each blob the cube drawn there covers.
2. **Outlines.** The cubes' silhouettes fitted to the blobs' outlines, whatever is
   ragged about a blob counting little.
3. **Edges.** Each visible edge of each cube moved onto the strongest colour edge
   beside it (`edges.py`). An outline counts only where the cube's colour is on
   its inner side: a shadow's edge, or the desk's grain, has the desk on both sides.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import cubes, edges
from .cubes import COLOURS, Blob, CubesNotFound
from .model import (BASE_DIAGONALS, CUBE_MM, Edge, Pinhole, around, camera_position, cube_corners, hidden_edges,
                    least_squares, project, sample, to_polygon, visible_edges)

Points = Dict[str, Tuple[float, float]]   # desk x, y under each cube's centre

# The board negates x: the arm's x, y with z up are left-handed, the camera's world is not.
HANDEDNESS = np.diag([-1.0, 1.0])

# The parameters, in one vector: focal length and principal point in pixels; the
# camera's rotation and shift; each cube's turn; the stretch from the arm's x, y
# to true mm (symmetric: its turn is the camera's).
FOCAL, CENTRE = 0, slice(1, 3)
ROTATION, SHIFT, TURNS, STRETCH = slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15)
STEPS = np.array([0.5, 0.25, 0.25] + [1e-5] * 3 + [0.01] * 3 + [1e-5] * 3 + [1e-6] * 3)

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
CENTRE_SPREAD = 0.1
# A stretch or a skew from the arm's x, y to true mm beyond these says a cube is
# not where the arm recorded it: moved since, or recorded off centre.
MAX_STRETCH = 1.12
MAX_SKEW_DEG = 6.0
# Fewer edge points than this, all cubes together, and the cubes' edges were not followed.
MIN_EDGE_POINTS = 60


class CameraNotFound(Exception):
    """The cubes were found, but no camera explains them where the arm says they stand."""


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
        """The hidden edges, and the bottom's diagonals: an X under the cube's centre."""
        return self.lines([*self.hidden, *BASE_DIAGONALS])


@dataclass
class CameraFit:
    true_to_pixel: np.ndarray        # 3x4: true mm (x, y, z up) to pixels
    desk_to_true: np.ndarray         # 2x2: the arm's desk x, y to true x, y (z is the same)
    cubes: Dict[str, dict]           # colour: desk x, y, the cube's turn, degrees, and the pixel under its centre
    frame_size: Tuple[int, int]
    edge_px: Tuple[float, float]     # how far the cubes' edges sit from the frame's: median, 90%
    edges_used: int
    # Refined by the red cube's points (refine.py): how many, and how far off they are, median and worst, mm;
    # and the calibration as the three cubes alone gave it: its two matrices and its cubes. None for the cubes' own.
    refined: Optional[dict] = None
    cubes_alone: Optional[dict] = None

    @property
    def desk_to_pixel(self) -> np.ndarray:
        """3x4: desk (x, y, z) to pixels, the arm's x/y error taken in."""
        lift = np.eye(4)
        lift[:2, :2] = self.desk_to_true
        matrix = self.true_to_pixel @ lift
        return matrix / np.linalg.norm(matrix[2, :3])

    @property
    def desk_to_shares(self) -> np.ndarray:
        """3x4: desk (x, y, z) to shares of the frame's width and height, pixel centres as `desk_to_pixel` has them:
        what a page draws by, at any size of the same picture."""
        width, height = self.frame_size
        shares = np.array([[1 / width, 0.0, 0.5 / width], [0.0, 1 / height, 0.5 / height], [0.0, 0.0, 1.0]])
        return shares @ self.desk_to_pixel

    @property
    def camera_position(self) -> np.ndarray:
        """Where the camera stands, in desk mm."""
        true = camera_position(self.true_to_pixel)
        return np.array([*np.linalg.solve(self.desk_to_true, true[:2]), true[2]])

    @property
    def focal(self) -> float:
        intrinsic = cv2.decomposeProjectionMatrix(self.true_to_pixel)[0]
        return float(abs(intrinsic[0, 0] / intrinsic[2, 2]))

    def arm_scale(self) -> Tuple[float, float, float]:
        """True mm per desk mm along the arm's x and y, and how far from square their axes are, degrees."""
        x_axis, y_axis = self.desk_to_true[:, 0], self.desk_to_true[:, 1]
        cosine = x_axis @ y_axis / (np.linalg.norm(x_axis) * np.linalg.norm(y_axis))
        return float(np.linalg.norm(x_axis)), float(np.linalg.norm(y_axis)), 90.0 - math.degrees(math.acos(cosine))

    def wireframe(self, colour: str) -> "Wireframe":
        """A cube in the frame, standing where it was recorded and turned as fitted."""
        cube = self.cubes[colour]
        centre = self.desk_to_true @ (cube["x"], cube["y"])
        turn = math.radians(cube["turn_deg"])
        seen = visible_edges(centre, turn, camera_position(self.true_to_pixel))
        return Wireframe(project(self.true_to_pixel, cube_corners(centre, turn)), seen, hidden_edges(seen))

    def cube_fit(self) -> "CameraFit":
        """The calibration as the three cubes alone gave it: itself, unless refined."""
        if self.cubes_alone is None:
            return self
        alone = self.cubes_alone
        return CameraFit(np.array(alone["true_to_pixel"]), np.array(alone["desk_to_true"]), alone["cubes"],
                         self.frame_size, self.edge_px, self.edges_used)

    def to_json(self) -> dict:
        refinement = {"refined": self.refined, "cubes_alone": self.cubes_alone} if self.refined else {}
        return {"true_to_pixel": self.true_to_pixel.tolist(), "desk_to_true": self.desk_to_true.tolist(),
                "cubes": self.cubes, "frame_size": list(self.frame_size),
                "edge_px": list(self.edge_px), "edges_used": self.edges_used, **refinement}

    @classmethod
    def from_json(cls, stored: dict) -> "CameraFit":
        true_to_pixel, desk_to_true = np.array(stored["true_to_pixel"]), np.array(stored["desk_to_true"])
        cubes = {colour: cube if "pixel" in cube else _with_pixel(cube, true_to_pixel, desk_to_true)
                 for colour, cube in stored["cubes"].items()}
        return cls(true_to_pixel, desk_to_true, cubes, tuple(stored["frame_size"]), tuple(stored["edge_px"]),
                   stored["edges_used"], stored.get("refined"), stored.get("cubes_alone"))


def _with_pixel(cube: dict, true_to_pixel: np.ndarray, desk_to_true: np.ndarray) -> dict:
    """The cube's entry with the pixel under its centre, on the desk, as the cubes' fit puts it."""
    centre = desk_to_true @ (cube["x"], cube["y"])
    pixel = project(true_to_pixel, np.array([[*centre, 0.0]]))[0]
    return {**cube, "pixel": [round(float(pixel[0]), 3), round(float(pixel[1]), 3)]}


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
    params, distances = _fit_edges(frame, naming, points, params)
    fit = _result(params, points, frame, distances)
    scales, skew = np.linalg.eigvalsh(_stretch(params)), abs(fit.arm_scale()[2])
    if scales.min() < 1 / MAX_STRETCH or scales.max() > MAX_STRETCH or skew > MAX_SKEW_DEG:
        raise CameraNotFound(f"the cubes do not stand where the arm recorded them: as the camera sees them they "
                             f"are {scales.min():.2f} to {scales.max():.2f} times as far apart, and askew by "
                             f"{skew:.1f} degrees. Was a cube moved after RECORD, or recorded off centre?")
    return fit


# --- the parameters -------------------------------------------------------

def _stretch(params: np.ndarray) -> np.ndarray:
    xx, yy, xy = params[STRETCH]
    return np.array([[xx, xy], [xy, yy]])


def _desk_to_true(params: np.ndarray) -> np.ndarray:
    return _stretch(params) @ HANDEDNESS


def _pinhole(params: np.ndarray) -> Pinhole:
    return Pinhole(params[FOCAL], tuple(params[CENTRE]), params[ROTATION], params[SHIFT])


def _centres(params: np.ndarray, points: Points) -> List[np.ndarray]:
    to_true = _desk_to_true(params)
    return [to_true @ points[colour] for colour in COLOURS]


def _corners(params: np.ndarray, points: Points) -> List[np.ndarray]:
    """Each cube's 8 corners in the frame, in COLOURS order."""
    camera = _pinhole(params)
    return [camera.project(cube_corners(centre, turn))
            for centre, turn in zip(_centres(params, points), params[TURNS])]


def _result(params: np.ndarray, points: Points, frame: np.ndarray, distances: np.ndarray) -> CameraFit:
    true_to_pixel, desk_to_true = _pinhole(params).matrix, _desk_to_true(params)
    cubes_seen = {colour: _with_pixel({"x": float(points[colour][0]), "y": float(points[colour][1]),
                                       "turn_deg": round(math.degrees(turn) % 90.0, 2)}, true_to_pixel, desk_to_true)
                  for colour, turn in zip(COLOURS, params[TURNS])}
    return CameraFit(true_to_pixel, desk_to_true, cubes_seen,
                     (frame.shape[1], frame.shape[0]),
                     (float(np.median(np.abs(distances))), float(np.percentile(np.abs(distances), 90))),
                     len(distances))


# --- 1. the start ---------------------------------------------------------

def _start(frame: np.ndarray, naming: Dict[str, Blob], points: Points) -> Tuple[float, np.ndarray]:
    """The best overlap the cubes get, and the camera that gives it, the arm's x and y taken as true."""
    height, width = frame.shape[:2]
    scale = min(1.0, 640.0 / width)
    masks = [cv2.resize(naming[colour].mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) > 0
             for colour in COLOURS]
    params = np.zeros(15)
    params[CENTRE], params[STRETCH] = (width / 2, height / 2), (1.0, 1.0, 0.0)
    middles = np.array([(*centre, CUBE_MM / 2) for centre in _centres(params, points)])
    blobs = np.array([naming[colour].centroid for colour in COLOURS])
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
            for index, (mask, centre) in enumerate(zip(masks, _centres(trial, points))):
                turn_overlaps = [(edges.overlap(camera.project(cube_corners(centre, math.radians(turn))) * scale,
                                                mask), math.radians(turn)) for turn in TURN_TRIES_DEG]
                cover, trial[TURNS.start + index] = max(turn_overlaps)
                overlap += cover / len(COLOURS)
            if overlap > best[0]:
                best = (overlap, trial)
    return best


# --- 2. the outlines ------------------------------------------------------

def _fit_outlines(naming: Dict[str, Blob], points: Points, params: np.ndarray) -> np.ndarray:
    """The cubes' silhouettes onto the blobs' outlines, both ways: outline to silhouette and back.

    Twice: the second time without the outline points the first left far off.
    """
    outlines = []
    for colour in COLOURS:
        blob = naming[colour]
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

    free = [index for index in range(len(params)) if index not in (CENTRE.start, CENTRE.stop - 1)]
    for _ in range(2):
        params = least_squares(residuals, params, STEPS, huber=OUTLINE_HUBER_PX, free=free, iterations=60)[0]
        for corners, entry in zip(_corners(params, points), outlines):
            silhouette = cv2.convexHull(corners.astype(np.float32)).reshape(-1, 2).astype(float)
            off = to_polygon(entry[0], silhouette)
            entry[0] = entry[0][off <= max(OUTLINE_HUBER_PX, OUTLINE_KEEP * np.median(off))]
    return params


# --- 3. the edges ---------------------------------------------------------

def _fit_edges(frame: np.ndarray, naming: Dict[str, Blob], points: Points,
               params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Every visible edge onto the frame's edges, round after round as the search narrows."""
    desk = cubes.desk_chroma(frame)
    crops = [edges.crop(frame, naming[colour], desk) for colour in COLOURS]
    centre = np.array([frame.shape[1] / 2, frame.shape[0] / 2])
    distances = np.zeros(0)
    for reach in REACH:
        found = [edges.search(crop, corners, seen, reach) for crop, corners, seen in
                 zip(crops, _corners(params, points), _edges_seen(params, points))]
        if sum(len(targets) for _, _, targets in found) < MIN_EDGE_POINTS:
            raise CameraNotFound("the cubes' edges could not be followed in the frame: is the light even, and "
                                 "each cube clear of other things?")

        def residuals(trial: np.ndarray) -> np.ndarray:
            parts = [edges.signed_distances(corners, *found_on) for corners, found_on in
                     zip(_corners(trial, points), found)]
            parts.append((trial[CENTRE] - centre) / (CENTRE_SPREAD * frame.shape[1]))
            return np.concatenate(parts)

        params, distances, _ = least_squares(residuals, params, STEPS, huber=EDGE_HUBER_PX, iterations=30)
        distances = distances[:-2]
    return params, distances


def _edges_seen(params: np.ndarray, points: Points) -> List[List[Edge]]:
    camera = _pinhole(params).position
    return [visible_edges(centre, turn, camera) for centre, turn in zip(_centres(params, points), params[TURNS])]
