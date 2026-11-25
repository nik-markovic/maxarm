"""SAVE CALIBRATION: the camera fitted to every point's cube, the arm laid on it again, and its correction.

    refined, tie, put = refine_and_leave_out(fit, points, frame)   # points: RefinePoints; frame: CALIBRATE's
    off = off_by(refined, point.sighting)    # how far its reading is from its cube as seen; how tall it looks

A point is a cube set down in place after place, centered under the cup, and seen:
the firmware's position over it, and the edges found round it in the snapshot
(`sighting.py`). Neither depends on a camera. SAVE fits one camera to all the
points' cubes at once, as CALIBRATE does to its three: the 40 mm cubes its only
length, the arm aside -- reality, not bent onto the arm -- from their bottom and
vertical edges, the tops' being found inside the cube. Through it the base is found
again in CALIBRATE's frame and the arm laid as CALIBRATE lays it (`anchor.fit_tie()`):
its axis by the base, its turn by the points' bearings. What is left at each point is a row of the arm's correction, the
mesh (`anchor.py`): SAVE writes `arm-true.json`, which the library reads on the fly
(`maxarm/converter.py`), and camera.json with its report. With fewer than
MIN_CAMERA_POINTS points with edges, or without CALIBRATE's frame, the camera and
the arm's frame stay CALIBRATE's.

A point's miss without it is its reading through the mesh of the other points,
against its cube as the camera sees it: how far the arm would land from the cube,
had the point not been taken. A point its neighbors put further than
`anchor.DOES_NOT_FIT_MM` from its cube does not fit: it is left out of the mesh and
named. At the points in the mesh the correction is exact.

A point's height (Z) is the camera's alone: how far from 40 mm tall its cube
looks, its base on the desk. One far off while the others are not says its edges
were not all the cube's: a bad snapshot.

With no points, SAVE puts back the three cubes' own calibration: the arm laid on
them by their readings, and no correction.
"""

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

import cv2

from applib.camera.base import find_base
from applib.camera.fit import CENTER_SPREAD, EDGE_HUBER_PX, CameraFit, arm_measure, without_top
from applib.camera.model import CUBE_MM, Pinhole, least_squares
from applib.camera.sighting import Sighting, in_frame, measure, misses, on_desk, place, start
from maxarm import raw

from . import anchor

MIN_CAMERA_POINTS = 3       # points with edges the camera is fitted to; fewer: CALIBRATE's camera
# The joint fit's finite-difference steps: focal length and principal point, px; turn, Rodrigues; shift, mm; then
# each cube's x, y, mm, and turn, radians.
CAMERA_STEPS = [0.05, 0.05, 0.05, 1e-5, 1e-5, 1e-5, 0.01, 0.01, 0.01]
CUBE_STEPS = [0.01, 0.01, 1e-5]


class PointsDoNotFit(Exception):
    """The points do not lay the arm on the camera."""


@dataclass(frozen=True)
class Off:
    seen: np.ndarray                 # desk x, y of the cube's center, as the calibration sees it
    across: float                    # mm from where the arm says the cup was, across the desk
    height: Optional[float]          # mm the cube's top is from 40: + taller; None for a point without edges


def off_by(fit: CameraFit, point: Sighting) -> Off:
    """Where the calibration sees the point's cube, a 40 mm one, how far that is from where the arm says the cup was,
    and how tall the cube looks."""
    matrix = in_frame(fit.true_to_pixel, fit.frame_size, point)
    center, height = _seen(matrix, point), None
    if point.edges is not None:
        height = float(measure(matrix, point, start(matrix, point))[3] - CUBE_MM)
    seen = np.linalg.solve(fit.desk_to_true, center)
    return Off(seen, float(math.dist(seen, (point.x, point.y))), height)


def refine(fit: CameraFit, points, frame: Optional[np.ndarray]) -> Tuple[CameraFit, Optional[anchor.Tie]]:
    """camera.json fitted and laid by the points, and the tie whose mesh is the correction; with no points, the three
    cubes' own calibration and no tie. Raises PointsDoNotFit if the points do not lay the arm."""
    refined, tie, _ = refine_and_leave_out(fit, points, frame)
    return refined, tie


def refine_and_leave_out(fit: CameraFit, points, frame: Optional[np.ndarray]
                         ) -> Tuple[CameraFit, Optional[anchor.Tie], List[Optional[np.ndarray]]]:
    """`refine()`, and each point's reading through the mesh of the other points: true x, y as the refined
    calibration has them (the board's mirror in them), None with no others."""
    alone = fit.cube_fit()
    if not points:
        return alone, None, []
    camera = _points_camera(alone, points) if frame is not None else None
    try:
        if camera is not None:
            found = find_base(frame, camera.desk_to_pixel)
            alone, tie = camera, anchor.fit_tie(found, anchor.tie_points(camera, points))
        else:
            found, tie_points = anchor.base_of(fit, frame), anchor.tie_points(alone, points)
            tie = (anchor.kept_tie(fit.anchor, found, tie_points) if fit.anchor
                   else anchor.fit_tie(found, tie_points))
    except (ValueError, np.linalg.LinAlgError) as reason:
        raise PointsDoNotFit(f"the points do not lay the arm on the camera: {reason}") from None
    refined = anchor.anchored(alone, tie)
    put = [None if tie.left.get(point.id) is None else anchor.to_anchored(tie.left[point.id]) for point in points]
    offs = [off_by(refined, point.sighting) for point in points]
    across = [float(math.dist(others, refined.desk_to_true @ off.seen))
              for point, off, others in zip(points, offs, put) if others is not None and point.id not in tie.misfits]
    heights = [abs(off.height) for off in offs if off.height is not None]
    summary = {"points": len(points)}
    if across:
        summary["off_mm"] = [round(float(np.median(across)), 2), round(max(across), 2)]
    if heights:
        summary["height_mm"] = [round(float(np.median(heights)), 2), round(max(heights), 2)]
    if tie.misfits:
        summary["does_not_fit_mm"] = {str(key): value for key, value in tie.misfits.items()}
    return replace(refined, refined=summary), tie, put


def _points_camera(alone: CameraFit, points) -> Optional[CameraFit]:
    """One camera fitted to every point's cube on its bottom and vertical edges, from `alone`, and the arm's measure
    against it; None with too few points seen in the calibration's frame size."""
    seen = [point for point in points if point.edges and tuple(point.frame_size) == tuple(alone.frame_size)]
    if len(seen) < MIN_CAMERA_POINTS:
        return None
    sightings = [point.sighting for point in seen]
    fitted = [replace(sighting, edges=without_top(sighting.edges)) for sighting in sightings]
    matrix = alone.true_to_pixel
    numbers = np.concatenate([_pinhole_numbers(matrix),
                              *(place(matrix, sighting, start(matrix, sighting)[:3]) for sighting in sightings)])
    middle = np.array(alone.frame_size, float) / 2

    def residuals(trial: np.ndarray) -> np.ndarray:
        camera = _camera(trial)
        parts = [misses(camera, sighting, np.append(trial[9 + 3 * index:12 + 3 * index], CUBE_MM))
                 for index, sighting in enumerate(fitted)]
        parts.append((trial[1:3] - middle) / (CENTER_SPREAD * alone.frame_size[0]))     # as CALIBRATE holds it
        return np.concatenate(parts)

    steps = np.array(CAMERA_STEPS + CUBE_STEPS * len(fitted))
    numbers, distances, _ = least_squares(residuals, numbers, steps, huber=EDGE_HUBER_PX, iterations=60)
    distances = np.abs(distances[:-2])
    camera = _camera(numbers)
    readings = [raw.formula_true(tuple(point.raw))[:2] if point.raw else (point.x, point.y) for point in seen]
    cubes = [numbers[9 + 3 * index:11 + 3 * index] for index in range(len(seen))]
    return replace(alone, true_to_pixel=camera, edge_px=(float(np.median(distances)), float(np.percentile(distances, 90))),
                   edges_used=len(distances), arm_stretch=arm_measure(readings, cubes), refined=None,
                   cubes_alone=None, anchor=None)


def _pinhole_numbers(matrix: np.ndarray) -> np.ndarray:
    """A 3x4 camera as focal length, principal point, Rodrigues turn and shift."""
    intrinsic = cv2.RQDecomp3x3(matrix[:, :3])[1]
    intrinsic, matrix = intrinsic / intrinsic[2, 2], matrix / intrinsic[2, 2]
    rotation, shift = np.linalg.solve(intrinsic, matrix[:, :3]), np.linalg.solve(intrinsic, matrix[:, 3])
    if np.linalg.det(rotation) < 0:
        rotation, shift = -rotation, -shift
    return np.array([(intrinsic[0, 0] + intrinsic[1, 1]) / 2, intrinsic[0, 2], intrinsic[1, 2],
                     *cv2.Rodrigues(rotation)[0].ravel(), *shift])


def _camera(numbers: np.ndarray) -> np.ndarray:
    return Pinhole(numbers[0], (numbers[1], numbers[2]), numbers[3:6], numbers[6:9]).matrix


def _seen(matrix: np.ndarray, point: Sighting) -> np.ndarray:
    """True x, y under the point's cube through a camera for its frame: the 40 mm cube on its edges, or the pixel."""
    if point.edges is None:
        return on_desk(matrix, point)
    return place(matrix, point, start(matrix, point)[:3])[:2]
