"""The camera fitted again to the points: a cube set down in place after place, centred under the cup, and seen.

    points = [Sighting(x, y, pixel, frame_size, turn_deg, edges), ...]   # sighting.py
    refined = refine(fit, points)            # fit: the saved calibration, store.load_camera()
    off = off_by(refined, point)             # how far its reading is from where the points put it; how tall it looks

A point pairs where the arm says the cup is, over the cube's centre, with the
edges found round the cube in the snapshot (`sighting.py`). SAVE CALIBRATION
makes two things of the points, apart:

1. **The camera**, from the pictures alone. Every point's cube is a 40 mm cube,
   and the camera -- focal length, principal point (held loosely to the frame's
   middle, as `fit.py` holds it), where it stands and which way it looks -- is
   the one through which all their edges lie best along those found, each cube
   wherever its edges put it: as CALIBRATE fits three cubes, all the points'.
   The arm has no say in it; the camera's millimetres are the cubes'.
2. **The arm**, laid on what the camera sees: a turn, a shift and a stretch
   (`fit.link_arm()`), the least squares over the points of how far each arm
   reading is from its cube as the camera sees it. Six numbers, which three
   points fit exactly: a point's miss across the desk (XY) says something from
   the fourth point on -- how far its reading is from where the others put it:
   the cup off the cube's centre, or the cube's edges not found right. The
   stretch is the arm's measure against the camera's, shown, not counted
   against any point.

A point's height (Z) is the camera's alone: how far from 40 mm tall its cube
looks, its base on the desk. One far off while the others are not says its
edges were not all the cube's: a bad snapshot.

The three cubes are points themselves, the first three, until forgotten; their
own calibration (`cubes_alone`) is only where the camera's fit starts, and what
a focal length far from it is refused against. A point from before the edges
were kept has only the pixel under the cube's centre, as the camera of its day
saw it: it counts for the arm, not the camera, and has no height.

`camera.json` keeps its form: `desk_to_true` is the stretch and the board's
mirror, `true_to_pixel` the camera along the arm's x and y. Points are the
frame's pixels and edges, which no refinement changes, so a calibration refined
again is refined from the same points.
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

from .fit import CENTRE_SPREAD, EDGE_HUBER_PX, STEPS, CameraFit, laid_along_arm, link_arm, stretch_problem
from .model import CUBE_MM, Pinhole, least_squares
from .sighting import STEPS as CUBE_STEPS
from .sighting import Sighting, in_frame, measure, misses, on_desk, place, start

# The camera's numbers: focal length, principal point, rotation, shift; fit.py's steps for them.
CAMERA_STEPS = STEPS[:9]
ITERATIONS = 100
# The points' camera may be this much longer or shorter in focal length than the cubes' before it is refused: points
# all in one corner of the desk, or along one line, do not fix a camera.
MAX_FOCAL_CHANGE = 0.15
# Points with edges the camera is fitted again from, at the least: three cubes fix it, as CALIBRATE's do. With fewer,
# the cubes' own camera is kept and only the arm laid again.
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
    """The camera fitted to `points`' pictures, started from `fit`'s three cubes' calibration, and the arm laid on
    all the points. With no points, the three cubes' own calibration. Raises PointsDoNotFit if the points do not
    fix a camera, or do not lay the arm."""
    alone = fit.cube_fit()
    if not points:
        return alone
    refined = _laid(alone, _camera(alone, points), points)
    offs = [off_by(refined, point) for point in points]
    across = [off.across for off in offs]
    heights = [abs(off.height) for off in offs if off.height is not None]
    summary = {"points": len(points), "off_mm": [round(float(np.median(across)), 2), round(max(across), 2)]}
    if heights:
        summary["height_mm"] = [round(float(np.median(heights)), 2), round(max(heights), 2)]
    return dataclasses.replace(refined, refined=summary,
                               cubes_alone={"true_to_pixel": alone.true_to_pixel.tolist(),
                                            "desk_to_true": alone.desk_to_true.tolist(), "cubes": alone.cubes})


def _seen(matrix: np.ndarray, point: Sighting) -> np.ndarray:
    """True x, y under the point's cube through a camera for its frame: the 40 mm cube on its edges, or the pixel."""
    if point.edges is None:
        return on_desk(matrix, point)
    return place(matrix, point, start(matrix, point)[:3])[:2]


def _camera(alone: CameraFit, points: Sequence[Sighting]) -> np.ndarray:
    """The camera, true mm, through which every point's cube, 40 mm, lies best along the edges found round it, each
    cube wherever its edges put it: the pictures alone. Fewer than MIN_CUBES points with edges: the cubes' own.

    The fit steps through the camera's numbers alone: each cube is placed afresh through each camera tried, starting
    from where it was last, and its edges' misses move with the camera as far as its own move cannot undo
    (`_edge_moves()`).
    """
    seen = [point for point in points if point.edges is not None]
    if len(seen) < MIN_CUBES:
        return alone.true_to_pixel
    first = _numbers(alone.true_to_pixel)
    width, height = alone.frame_size
    middle = np.array([width / 2, height / 2])
    cubes = [start(in_frame(alone.true_to_pixel, alone.frame_size, point), point)[:3] for point in seen]

    def view(numbers: np.ndarray, point: Sighting) -> np.ndarray:
        return in_frame(_pinhole(numbers).matrix, alone.frame_size, point)

    def placed(numbers: np.ndarray) -> None:
        for index, point in enumerate(seen):
            cubes[index] = place(view(numbers, point), point, cubes[index])

    def residuals(numbers: np.ndarray) -> np.ndarray:
        placed(numbers)
        parts = [misses(view(numbers, point), point, np.append(cube, CUBE_MM)) for point, cube in zip(seen, cubes)]
        parts.append((numbers[1:3] - middle) / (CENTRE_SPREAD * width))
        return np.concatenate(parts)

    def derivatives(numbers: np.ndarray) -> np.ndarray:
        placed(numbers)
        rows = [_edge_moves(lambda trial: view(trial, point), numbers, point, cube) for point, cube in zip(seen, cubes)]
        held = np.zeros((2, len(numbers)))
        held[0, 1] = held[1, 2] = 1 / (CENTRE_SPREAD * width)
        return np.vstack([*rows, held])

    try:
        numbers = least_squares(residuals, first, CAMERA_STEPS, huber=EDGE_HUBER_PX, iterations=ITERATIONS,
                                derivatives=derivatives)[0]
    except np.linalg.LinAlgError:
        numbers = np.full(len(first), np.nan)                     # refused below
    camera = _pinhole(numbers)
    if not abs(numbers[0] / first[0] - 1) <= MAX_FOCAL_CHANGE or not camera.position[2] > CUBE_MM:
        raise PointsDoNotFit(f"the points do not fix the camera (its focal length {numbers[0]:.0f} px against the "
                             f"cubes' {first[0]:.0f}): spread them over the desk, near and far, left and right")
    return camera.matrix


def _laid(alone: CameraFit, true_to_pixel: np.ndarray, points: Sequence[Sighting]) -> CameraFit:
    """The camera with the arm laid on it: the turn, shift and stretch that put each point's arm reading nearest its
    cube as the camera sees it, least squares over the points (`fit.link_arm()`)."""
    seen = [_seen(in_frame(true_to_pixel, alone.frame_size, point), point) for point in points]
    try:
        turn, shift, desk_to_true = link_arm([(point.x, point.y) for point in points], seen)
    except (ValueError, np.linalg.LinAlgError):
        raise PointsDoNotFit("the points do not lay the arm on the camera: spread them over the desk, not along a "
                             "line") from None
    problem = stretch_problem(desk_to_true)
    if problem:
        raise PointsDoNotFit(f"the points do not lay the arm on the camera: {problem}. Forget the points far off "
                             f"and record them again")
    camera, cubes = laid_along_arm(true_to_pixel, alone.cubes, turn, shift)
    return CameraFit(camera, desk_to_true, cubes, alone.frame_size, alone.edge_px, alone.edges_used)


def _edge_moves(camera: Callable[[np.ndarray], np.ndarray], numbers: np.ndarray, point: Sighting,
                cube: np.ndarray) -> np.ndarray:
    """How the point's edge misses move with each of the camera's numbers, the cube placed afresh: a column each.

    The cube placed is where its edges miss those found least. Nudge a number of the camera and the misses move
    (`by_camera`); the cube placed then moves as far as undoes what it can of that (`by_cube`), each edge weighted
    as the placing weights it; what is left is the camera's.
    """
    def edge_misses(camera_numbers: np.ndarray, cube_numbers: np.ndarray) -> np.ndarray:
        return misses(camera(camera_numbers), point, np.append(cube_numbers, CUBE_MM))

    now = edge_misses(numbers, cube)
    weights = np.where(np.abs(now) <= EDGE_HUBER_PX, 1.0, EDGE_HUBER_PX / np.maximum(np.abs(now), 1e-12))[:, None]
    by_cube = _differences(lambda trial: edge_misses(numbers, trial), cube, CUBE_STEPS[:3])
    by_camera = _differences(lambda trial: edge_misses(trial, cube), numbers, CAMERA_STEPS)
    moves = -np.linalg.lstsq(by_cube.T @ (weights * by_cube), by_cube.T @ (weights * by_camera), rcond=None)[0]
    return by_camera + by_cube @ moves


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
