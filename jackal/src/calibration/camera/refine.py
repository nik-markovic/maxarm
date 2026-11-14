"""The camera fitted again to the points: a cube set down in place after place, centred under the cup, and seen.

    points = [Sighting(x, y, pixel, frame_size, turn_deg, edges), ...]   # sighting.py
    refined = refine(fit, points)            # fit: the saved calibration, store.load_camera()
    off = off_by(refined, point)             # how far its reading is from its cube as seen; how tall it looks

A point pairs where the arm says the cup is, over the cube's centre, with the
edges found round the cube in the snapshot (`sighting.py`). SAVE CALIBRATION keeps
the camera CALIBRATE made from the cubes' edges -- reality, not bent onto the arm
(the owner, 13 Nov) -- and lays the arm on it again, by the turn and shift that
best put every point's cube, as the camera sees it, on the arm's reading for it:
more points, spread out, place the arm's axis and turn better than three. Each
point's miss is then the arm's own error there.

A point's miss is how far its reading is from its cube as the camera sees it,
placed where its own edges put it, through the camera the other points make
(`left_out()`): the cup off the cube's centre, the reading off, or the cube's
edges not found right. The stretch a turn, shift and stretch would take between
the readings and the cubes as seen is kept as the arm's measure (`arm_stretch`):
shown, and checked.

A point's height (Z) is the camera's alone: how far from 40 mm tall its cube
looks, its base on the desk. One far off while the others are not says its
edges were not all the cube's: a bad snapshot.

The three cubes are points themselves, the first three, until forgotten; their
own calibration (`cubes_alone`) is only where the camera's fit starts, and what
a focal length far from it is refused against. A point from before the edges
were kept has only the pixel under the cube's centre: it does not shape the
camera, and has no height.

`camera.json` keeps its form: `desk_to_true` is the board's mirror,
`true_to_pixel` the camera along the arm's x and y, `arm_stretch` the measure. Points are the
frame's pixels and edges, which no refinement changes, so a calibration refined
again is refined from the same points.
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from applib.camera.fit import (CENTRE_SPREAD, EDGE_HUBER_PX, HANDEDNESS, STEPS, CameraFit, arm_measure,
                               stretch_problem)
from applib.camera.model import CUBE_MM, Pinhole, least_squares
from applib.camera.sighting import STEPS as CUBE_STEPS
from applib.camera.sighting import Sighting, in_frame, measure, misses, on_desk, place, start

# The camera's numbers: focal length, principal point, rotation, shift; fit.py's steps for them.
CAMERA_STEPS = STEPS[:9]
ITERATIONS = 100
# A point left out: the camera without it starts from the one all the points make and moves little, so it is fitted
# this many steps, not to the last digit: the page works the points out again whenever one changes.
LEFT_OUT_ITERATIONS = 12
# Gauss-Newton steps a cube is turned by through each camera tried.
TURN_STEPS = 2
# The points' camera may be this much longer or shorter in focal length than the cubes' before it is refused: points
# all in one corner of the desk, or along one line, do not fix a camera.
MAX_FOCAL_CHANGE = 0.15
# Points with edges the camera is fitted again from, at the least: three cubes fix it, as CALIBRATE's do. With fewer,
# the cubes' own camera is kept.
MIN_CUBES = 3


class PointsDoNotFit(Exception):
    """The points do not make a camera: too few, or too close together, to fix it."""


@dataclass(frozen=True)
class Off:
    seen: np.ndarray                 # desk x, y of the cube's centre, as the calibration sees it
    across: float                    # mm from where the arm says the cup was, across the desk
    height: Optional[float]          # mm the cube's top is from 40: + taller; None for a point without edges


def cube_sightings(fit: CameraFit) -> List[Sighting]:
    """The three cubes the camera was fitted to, as points: where they stand, not how tall."""
    return [Sighting(cube["x"], cube["y"], tuple(cube["pixel"]), tuple(fit.frame_size))
            for cube in fit.cubes.values()]


def off_by(fit: CameraFit, point: Sighting) -> Off:
    """Where the calibration sees the point's cube, a 40 mm one, how far that is from where the arm says the cup was,
    and how tall the cube looks."""
    matrix = in_frame(fit.true_to_pixel, fit.frame_size, point)
    centre, height = _seen(matrix, point), None
    if point.edges is not None:
        height = float(measure(matrix, point, start(matrix, point))[3] - CUBE_MM)
    seen = np.linalg.solve(fit.desk_to_true, centre)
    return Off(seen, float(math.dist(seen, (point.x, point.y))), height)


def off_mm(fit: CameraFit, point: Sighting) -> float:
    """How far from where the arm says the cup was the calibration puts the cube's centre, desk mm."""
    return off_by(fit, point).across


def seen_desk(fit: CameraFit, point: Sighting) -> np.ndarray:
    """Desk x, y of the cube's centre as the calibration has it: where it would send the arm."""
    return off_by(fit, point).seen


def seen_true(fit: CameraFit, point: Sighting) -> np.ndarray:
    """True x, y of the desk at the point's pixel, as the calibration's camera sees it."""
    return on_desk(in_frame(fit.true_to_pixel, fit.frame_size, point), point)


def refine(fit: CameraFit, points: Sequence[Sighting]) -> CameraFit:
    """The camera fitted with every point's cube where the arm read it, started from `fit`'s three cubes'
    calibration. With no points, the three cubes' own calibration. Raises PointsDoNotFit if the points do not fix a
    camera, or the cubes as it sees them do not stand where the arm read them."""
    return refine_and_leave_out(fit, points)[0]


def refine_and_leave_out(fit: CameraFit,
                         points: Sequence[Sighting]) -> Tuple[CameraFit, List[Optional[np.ndarray]]]:
    """`refine()`, and each point's reading as the camera the other points make has it (`left_out()`)."""
    alone = fit.cube_fit()
    if not points:
        return alone, []
    seen = [_seen(in_frame(alone.true_to_pixel, alone.frame_size, point), point) for point in points]
    readings = [HANDEDNESS @ np.array([point.x, point.y]) for point in points]
    turn, shift = _lay(readings, seen)
    laid = np.eye(4)
    laid[:2, :2], laid[:2, 3] = turn, shift
    refined = _calibration(alone, alone.true_to_pixel @ laid, points)
    put = []                                          # each reading laid by the others, in the refined frame
    for index, reading in enumerate(readings):
        others = [k for k in range(len(points)) if k != index]
        if len(others) < MIN_CUBES:
            put.append(None)
            continue
        turn_o, shift_o = _lay([readings[k] for k in others], [seen[k] for k in others])
        put.append(turn.T @ (turn_o @ reading + shift_o - shift))
    offs = [off_by(refined, point) for point in points]
    across = [off.across if others is None else float(math.dist(others, refined.desk_to_true @ off.seen))
              for off, others in zip(offs, put)]                     # each through the others' camera
    heights = [abs(off.height) for off in offs if off.height is not None]
    summary = {"points": len(points), "off_mm": [round(float(np.median(across)), 2), round(max(across), 2)]}
    if heights:
        summary["height_mm"] = [round(float(np.median(heights)), 2), round(max(heights), 2)]
    stretch = {} if alone.arm_stretch is None else {"arm_stretch": alone.arm_stretch.tolist()}
    return dataclasses.replace(refined, refined=summary,
                               cubes_alone={"true_to_pixel": alone.true_to_pixel.tolist(),
                                            "desk_to_true": alone.desk_to_true.tolist(), **stretch,
                                            "cubes": alone.cubes}), put


def _lay(readings: Sequence[np.ndarray], seen: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """The turn and shift that best lay the readings (true x, y) on the cubes as seen: seen = turn @ reading + shift."""
    readings, seen = np.asarray(readings, float), np.asarray(seen, float)
    r_mean, s_mean = readings.mean(axis=0), seen.mean(axis=0)
    left, _, right = np.linalg.svd((seen - s_mean).T @ (readings - r_mean))
    turn = left @ right
    if np.linalg.det(turn) < 0:
        raise PointsDoNotFit("the points lie on their cubes only mirrored: a point recorded on the wrong cube?")
    return turn, s_mean - turn @ r_mean


def left_out(fit: CameraFit, points: Sequence[Sighting]) -> List[Optional[np.ndarray]]:
    """For each point, its reading as the camera the other points make has it against its cube: true x, y, so placed
    by the cube as `fit`'s camera sees it that its distance from there is the point's miss -- its cube, placed where
    its own edges put it through the others' camera, fitted again from `fit`'s, against where the arm read it. A point
    that does not shape the camera has its reading itself. None each with fewer than three points."""
    if len(points) < 3:
        return [None] * len(points)
    readings = [HANDEDNESS @ np.array([point.x, point.y]) for point in points]
    shaping = [index for index, point in enumerate(points) if point.edges is not None]
    if len(shaping) <= MIN_CUBES:                     # no camera made by the points, or none without one of them
        return readings
    numbers = _numbers(fit.true_to_pixel)
    put = list(readings)
    for index in shaping:
        point = points[index]
        on_arm = _OnArm(fit.frame_size, [points[other] for other in shaping if other != index], fit.true_to_pixel)
        try:
            others = least_squares(on_arm.residuals, numbers, CAMERA_STEPS, huber=EDGE_HUBER_PX,
                                   iterations=LEFT_OUT_ITERATIONS, derivatives=on_arm.derivatives)[0]
        except np.linalg.LinAlgError:
            continue                                  # the others alone do not fix a camera: its reading itself
        apart = _seen(in_frame(_pinhole(others).matrix, fit.frame_size, point), point)
        seen = _seen(in_frame(fit.true_to_pixel, fit.frame_size, point), point)
        put[index] = seen - (apart - readings[index])
    return put


def _seen(matrix: np.ndarray, point: Sighting) -> np.ndarray:
    """True x, y under the point's cube through a camera for its frame: the 40 mm cube on its edges, or the pixel."""
    if point.edges is None:
        return on_desk(matrix, point)
    return place(matrix, point, start(matrix, point)[:3])[:2]


class _OnArm:
    """The points' cubes where the arm read them, each turned as its edges say, through a camera's numbers: their
    edges' misses, and how those move with the numbers. Each cube is turned afresh through each camera tried,
    starting from its last turn."""

    def __init__(self, frame_size: Tuple[int, int], points: Sequence[Sighting], first: np.ndarray) -> None:
        self.frame_size = tuple(frame_size)
        self.points = list(points)
        self.centres = [HANDEDNESS @ np.array([point.x, point.y]) for point in self.points]
        self.turns = [float(start(in_frame(first, self.frame_size, point), point)[2]) for point in self.points]
        self.middle = np.array(self.frame_size, float) / 2
        self.spread = CENTRE_SPREAD * self.frame_size[0]

    def residuals(self, numbers: np.ndarray) -> np.ndarray:
        return np.concatenate([*self.blocks(numbers), self.held(numbers)])

    def derivatives(self, numbers: np.ndarray) -> np.ndarray:
        return np.vstack([*self.rows(numbers), self.held_rows(len(numbers))])

    def blocks(self, numbers: np.ndarray) -> List[np.ndarray]:
        """Each point's edge misses, its cube turned afresh."""
        self._turn(numbers)
        return [misses(self._view(numbers, point), point, self._cube(index))
                for index, point in enumerate(self.points)]

    def rows(self, numbers: np.ndarray) -> List[np.ndarray]:
        """How each point's misses move with the camera's numbers, its cube turned afresh: a block each."""
        self._turn(numbers)
        return [_edge_moves(lambda trial, point=point: self._view(trial, point), numbers, point, self._cube(index))
                for index, point in enumerate(self.points)]

    def held(self, numbers: np.ndarray) -> np.ndarray:
        """The principal point, held loosely to the frame's middle."""
        return (numbers[1:3] - self.middle) / self.spread

    def held_rows(self, count: int) -> np.ndarray:
        rows = np.zeros((2, count))
        rows[0, 1] = rows[1, 2] = 1 / self.spread
        return rows

    def _view(self, numbers: np.ndarray, point: Sighting) -> np.ndarray:
        return in_frame(_pinhole(numbers).matrix, self.frame_size, point)

    def _cube(self, index: int, turn: Optional[float] = None) -> np.ndarray:
        """The cube's numbers as `misses()` takes them: where it stands, its turn, its height."""
        return np.array([*self.centres[index], self.turns[index] if turn is None else turn, CUBE_MM])

    def _turn(self, numbers: np.ndarray) -> None:
        """Each cube turned through the camera: Gauss-Newton steps from its last turn, its edges weighted as the
        fit weights them. A camera tried moves little from the last, and the turn with it."""
        step = CUBE_STEPS[2]
        for index, point in enumerate(self.points):
            matrix = self._view(numbers, point)
            turn = self.turns[index]
            for _ in range(TURN_STEPS):
                now = misses(matrix, point, self._cube(index, turn))
                slope = (misses(matrix, point, self._cube(index, turn + step)) - now) / step
                weights = np.where(np.abs(now) <= EDGE_HUBER_PX, 1.0, EDGE_HUBER_PX / np.maximum(np.abs(now), 1e-12))
                turn -= float((weights * slope) @ now / max(float((weights * slope) @ slope), 1e-12))
            self.turns[index] = turn


def _camera(alone: CameraFit, points: Sequence[Sighting]) -> np.ndarray:
    """The camera, true mm, through which every point's cube, 40 mm, standing where the arm read it, lies best along
    the edges found round it. Fewer than MIN_CUBES points with edges: the cubes' own."""
    seen = [point for point in points if point.edges is not None]
    if len(seen) < MIN_CUBES:
        return alone.true_to_pixel
    first = _numbers(alone.true_to_pixel)
    on_arm = _OnArm(alone.frame_size, seen, alone.true_to_pixel)
    try:
        numbers = least_squares(on_arm.residuals, first, CAMERA_STEPS, huber=EDGE_HUBER_PX, iterations=ITERATIONS,
                                derivatives=on_arm.derivatives)[0]
    except np.linalg.LinAlgError:
        numbers = np.full(len(first), np.nan)                     # refused below
    camera = _pinhole(numbers)
    if not abs(numbers[0] / first[0] - 1) <= MAX_FOCAL_CHANGE or not camera.position[2] > CUBE_MM:
        raise PointsDoNotFit(f"the points do not fix the camera (its focal length {numbers[0]:.0f} px against the "
                             f"cubes' {first[0]:.0f}): spread them over the desk, near and far, left and right")
    return camera.matrix


def _calibration(alone: CameraFit, true_to_pixel: np.ndarray, points: Sequence[Sighting]) -> CameraFit:
    """The camera along the arm's x and y, as fitted; the stretch a turn, shift and stretch would take between the
    readings and the cubes as the camera sees them kept as the arm's measure, and checked."""
    seen = [_seen(in_frame(true_to_pixel, alone.frame_size, point), point) for point in points]
    stretch = arm_measure([(point.x, point.y) for point in points], seen)
    problem = ("as the camera sees them the arm's points are laid mirrored, or along a line" if stretch is None
               else stretch_problem(stretch))
    if problem:
        raise PointsDoNotFit(f"the points do not stand where the arm read them: {problem}. Forget the points far off "
                             f"and record them again")
    return CameraFit(true_to_pixel, HANDEDNESS.copy(), alone.cubes, alone.frame_size, alone.edge_px, alone.edges_used,
                     arm_stretch=stretch)


def _edge_moves(camera: Callable[[np.ndarray], np.ndarray], numbers: np.ndarray, point: Sighting,
                cube: np.ndarray) -> np.ndarray:
    """How the point's edge misses move with each of the camera's numbers, the cube turned afresh: a column each.

    The cube stands where the arm read it, turned as its edges miss those found least. Nudge a number of the camera
    and the misses move (`by_camera`); the cube turned then undoes what it can of that (`by_turn`), each edge weighted
    as the turning weights it; what is left is the camera's.
    """
    def edge_misses(camera_numbers: np.ndarray, turn: float) -> np.ndarray:
        return misses(camera(camera_numbers), point, np.array([cube[0], cube[1], turn, CUBE_MM]))

    now = edge_misses(numbers, cube[2])
    weights = np.where(np.abs(now) <= EDGE_HUBER_PX, 1.0, EDGE_HUBER_PX / np.maximum(np.abs(now), 1e-12))[:, None]
    by_turn = ((edge_misses(numbers, cube[2] + CUBE_STEPS[2]) - now) / CUBE_STEPS[2])[:, None]
    by_camera = _differences(lambda trial: edge_misses(trial, cube[2]), numbers, CAMERA_STEPS)
    moves = -np.linalg.lstsq(by_turn.T @ (weights * by_turn), by_turn.T @ (weights * by_camera), rcond=None)[0]
    return by_camera + by_turn @ moves


def _differences(function: Callable[[np.ndarray], np.ndarray], at: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """How `function` changes with each number of `at`: a column each, by finite differences."""
    now = function(at)
    columns = []
    for index, step in enumerate(steps):
        nudged = at.copy()
        nudged[index] += step
        columns.append((function(nudged) - now) / step)
    return np.column_stack(columns)


def _numbers(true_to_pixel: np.ndarray) -> np.ndarray:
    """A square-pixel 3x4 camera as its numbers: focal length, principal point, rotation, shift."""
    if np.linalg.det(true_to_pixel[:, :3]) < 0:
        true_to_pixel = -true_to_pixel                            # the same camera, its rotation a proper one
    intrinsic, rotation, position = cv2.decomposeProjectionMatrix(true_to_pixel)[:3]
    signs = np.diag(np.sign(np.diag(intrinsic)))                 # the decomposition's sign choice, undone
    intrinsic, rotation = intrinsic @ signs, signs @ rotation
    intrinsic = intrinsic / intrinsic[2, 2]
    position = position[:3, 0] / position[3, 0]
    return np.concatenate(([intrinsic[0, 0]], intrinsic[:2, 2], cv2.Rodrigues(rotation)[0].ravel(),
                           -rotation @ position))


def _pinhole(numbers: np.ndarray) -> Pinhole:
    return Pinhole(numbers[0], tuple(numbers[1:3]), numbers[3:6], numbers[6:9])
