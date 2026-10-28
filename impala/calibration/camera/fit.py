"""The camera, fitted to the three cubes: how the camera sees them, and where the arm says they stand.

    fit = fit_camera(frame, {"red": (x, y), "green": (x, y), "blue": (x, y)})   # desk mm
    fit.desk_to_pixel                                                           # 3x4, for camera.json

The cubes' shapes make the camera. A 40 mm cube seen from above is a small
frame of known edges, and three of them, each turned however it was set down,
fix a square-pixel camera: focal length, principal point and pose, in true
millimetres. While the camera is fitted, the cubes stand wherever their shapes
put them. Where the arm says they stand then places the arm on the desk: a turn
and a shift, with the board's mirror (it negates x), the best for the three
together (`place_arm()`). The arm's x and y are true millimetres, as the camera's
are -- the library corrects the firmware's (`maxarm/raw.py`) -- so nothing is
stretched: three points would make any stretch fit them exactly, whatever error
each was recorded with, and carry that error across the desk.

1. **Start.** For each naming of the blobs (`cubes.py`), and focal lengths across
   any plausible lens, the camera that puts each cube's middle on its blob (P3P),
   scored by how much of each blob the cube drawn there covers.
2. **Outlines.** The cubes' silhouettes fitted to the blobs' outlines, whatever is
   ragged about a blob counting little.
3. **Edges.** Each visible edge of each cube moved onto the strongest colour edge
   beside it (`edges.py`). An outline counts only where the cube's colour is on
   its inner side: a shadow's edge, or the desk's grain, has the desk on both sides.
4. **The arm.** Its three points turned and shifted onto the cubes as seen; the
   camera's true millimetres then laid along the arm's, so that `desk_to_true` is
   the mirror alone and `desk_to_pixel` names the desk point the arm must be sent to.
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
# camera's rotation and shift; each cube's turn; a stretch from the arm's x, y to
# where the cubes stand (symmetric: its turn is the camera's), which leaves each
# cube where its shape puts it while the camera is fitted.
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
# A stretch or a skew beyond these, between the arm's points and the cubes as the
# camera sees them, says a cube is not where the arm recorded it: moved since, or
# recorded off centre.
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
        """3x4: desk (x, y, z) to pixels, the arm placed on the desk as fitted."""
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

    def seen_centre(self, colour: str) -> np.ndarray:
        """True x, y of the desk under the cube's centre, where the camera saw it."""
        x, y, w = np.linalg.solve(self.true_to_pixel[:, [0, 1, 3]], (*self.cubes[colour]["pixel"], 1.0))
        return np.array([x / w, y / w])

    def cube_off_mm(self, colour: str) -> float:
        """How far from the arm's point for the cube the calibration puts the cube it saw, desk mm."""
        cube = self.cubes[colour]
        return float(math.dist(np.linalg.solve(self.desk_to_true, self.seen_centre(colour)), (cube["x"], cube["y"])))

    def wireframe(self, colour: str) -> "Wireframe":
        """A cube in the frame, standing where the camera saw it and turned as fitted."""
        centre = self.seen_centre(colour)
        turn = math.radians(self.cubes[colour]["turn_deg"])
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
    """The cube's entry with the pixel under its centre, on the desk, for a file written before it was kept: there
    the cubes' own fit put each cube exactly where the arm recorded it."""
    centre = desk_to_true @ (cube["x"], cube["y"])
    pixel = project(true_to_pixel, np.array([[*centre, 0.0]]))[0]
    return {**cube, "pixel": [round(float(pixel[0]), 3), round(float(pixel[1]), 3)]}


def place_arm(desk: np.ndarray, true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The turn and shift that lay the arm's desk x, y rows, through the board's mirror, best onto the `true` rows:
    true = turn @ HANDEDNESS @ desk + shift, least squares."""
    mirrored = np.asarray(desk, float) @ HANDEDNESS
    true = np.asarray(true, float)
    desk_mean, true_mean = mirrored.mean(axis=0), true.mean(axis=0)
    left, _, right = np.linalg.svd((true - true_mean).T @ (mirrored - desk_mean))
    turn = left @ np.diag([1.0, np.linalg.det(left @ right)]) @ right
    return turn, true_mean - turn @ desk_mean


def laid_along_arm(true_to_pixel: np.ndarray, cubes_seen: Dict[str, dict], turn: np.ndarray,
                   shift: np.ndarray) -> Tuple[np.ndarray, Dict[str, dict]]:
    """The camera with its true millimetres moved onto the arm's (true = turn @ new + shift), and each cube's turn as
    the new millimetres have it. The pixels stay: they are where the camera saw things."""
    lift = np.eye(4)
    lift[:2, :2], lift[:2, 3] = turn, shift
    turned_deg = math.degrees(math.atan2(turn[1, 0], turn[0, 0]))
    return true_to_pixel @ lift, {colour: {**cube, "turn_deg": round((cube["turn_deg"] - turned_deg) % 90.0, 2)}
                                  for colour, cube in cubes_seen.items()}


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
    scales, skew = np.linalg.eigvalsh(_stretch(params)), abs(_skew_deg(_desk_to_true(params)))
    if scales.min() < 1 / MAX_STRETCH or scales.max() > MAX_STRETCH or skew > MAX_SKEW_DEG:
        raise CameraNotFound(f"the cubes do not stand where the arm recorded them: as the camera sees them they "
                             f"are {scales.min():.2f} to {scales.max():.2f} times as far apart, and askew by "
                             f"{skew:.1f} degrees. Was a cube moved after RECORD, or recorded off centre?")
    return _result(params, points, frame, distances)


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
    """The camera as fitted, the arm laid onto the cubes it saw (step 4)."""
    true_to_pixel, centres = _pinhole(params).matrix, _centres(params, points)
    pixels = project(true_to_pixel, np.array([(*centre, 0.0) for centre in centres]))
    cubes_seen = {colour: {"x": float(points[colour][0]), "y": float(points[colour][1]),
                           "turn_deg": round(math.degrees(turn) % 90.0, 2),
                           "pixel": [round(float(pixel[0]), 3), round(float(pixel[1]), 3)]}
                  for colour, turn, pixel in zip(COLOURS, params[TURNS], pixels)}
    turn, shift = place_arm(np.array([points[colour] for colour in COLOURS]), np.array(centres))
    true_to_pixel, cubes_seen = laid_along_arm(true_to_pixel, cubes_seen, turn, shift)
    return CameraFit(true_to_pixel, HANDEDNESS.copy(), cubes_seen,
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
