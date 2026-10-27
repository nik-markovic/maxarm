"""The camera calibration refined by the red cube, set down in place after place, centred under the cup, and seen.

    points = [Sighting(x, y, pixel, frame_size), ...]   # desk x, y: where the arm says the cup is, over the cube
    refined = refine(fit, points)                         # fit: the saved calibration, store.load_camera()
    off_mm(refined, point)                                # how far from the arm's point the camera puts the cube

A point pairs where the arm says the cup is, over the cube's centre, with where
the camera sees that centre: the pixel under it, on the desk (`locate.py`). The
camera is the calibration's, held as it is: through it the pixel is a place on
the desk in true millimetres. How the arm's x and y lie on the desk -- the arm's
x/y error -- is then fitted to every point at once, the three cubes the camera
was fitted to among them: true = A desk + b, any turn, mirror, stretch and
shift, least squares. The three cubes alone fix it exactly, each with whatever
error it was recorded with; more points, spread over the desk, share it out.

`camera.json` keeps its form: the camera's true millimetres are turned and
shifted on the desk so that the arm's error is again a stretch and the board's
mirror (`fit.HANDEDNESS`), as the cubes gave it. Points are pixels, which no
refinement changes, so a calibration refined again is refined from the same
points.
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .fit import HANDEDNESS, MAX_SKEW_DEG, MAX_STRETCH, CameraFit
from .model import rescaled


@dataclass(frozen=True)
class Sighting:
    x: float                         # desk mm: where the arm says the cup is, over the cube's centre
    y: float
    pixel: Tuple[float, float]       # where the camera sees the desk under the cube's centre
    frame_size: Tuple[int, int]      # the frame's: the calibration's, or the same picture smaller


class RefineRefused(Exception):
    """The points do not make a calibration. The message says why."""


def cube_sightings(fit: CameraFit) -> List[Sighting]:
    """The three cubes the camera was fitted to, as points."""
    return [Sighting(cube["x"], cube["y"], tuple(cube["pixel"]), tuple(fit.frame_size))
            for cube in fit.cubes.values()]


def seen_true(fit: CameraFit, point: Sighting) -> np.ndarray:
    """True x, y of the desk under the cube's centre, as the calibration's camera sees it."""
    matrix = fit.true_to_pixel
    if tuple(point.frame_size) != tuple(fit.frame_size):
        matrix = rescaled(matrix, fit.frame_size, point.frame_size)
    x, y, w = np.linalg.solve(matrix[:, [0, 1, 3]], (*point.pixel, 1.0))
    return np.array([x / w, y / w])


def seen_desk(fit: CameraFit, point: Sighting) -> np.ndarray:
    """Desk x, y of the cube's centre as the calibration has it: where it would send the arm."""
    return np.linalg.solve(fit.desk_to_true, seen_true(fit, point))


def off_mm(fit: CameraFit, point: Sighting) -> float:
    """How far from where the arm says the cup was the calibration puts the cube's centre, desk mm."""
    return float(math.dist(seen_desk(fit, point), (point.x, point.y)))


def refine(fit: CameraFit, points: Sequence[Sighting]) -> CameraFit:
    """`fit` with the arm's x/y error fitted to its three cubes and `points`, the camera as it is. Raises if none."""
    every = cube_sightings(fit) + list(points)
    desk = np.array([(point.x, point.y, 1.0) for point in every])
    true = np.array([seen_true(fit, point) for point in every])
    solution = np.linalg.lstsq(desk, true, rcond=None)[0]
    turn, stretch = _turn_and_stretch(solution[:2].T @ HANDEDNESS)
    lift = np.eye(4)
    lift[:2, :2], lift[:2, 3] = turn, solution[2]
    turned_deg = math.degrees(math.atan2(turn[1, 0], turn[0, 0]))
    cubes = {colour: {**cube, "turn_deg": round((cube["turn_deg"] - turned_deg) % 90.0, 2)}
             for colour, cube in fit.cubes.items()}
    alone = fit.cubes_alone or {"true_to_pixel": fit.true_to_pixel.tolist(), "desk_to_true": fit.desk_to_true.tolist(),
                                "cubes": fit.cubes}
    refined = CameraFit(fit.true_to_pixel @ lift, stretch @ HANDEDNESS, cubes, fit.frame_size, fit.edge_px,
                        fit.edges_used, cubes_alone=alone)
    _check_stretch(refined, stretch)
    offs = [off_mm(refined, point) for point in every]
    return dataclasses.replace(refined, refined={"points": len(points),
                                                 "off_mm": [round(float(np.median(offs)), 2), round(max(offs), 2)]})


def _turn_and_stretch(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """`matrix` as a turn after a symmetric stretch; raises if it mirrors."""
    if np.linalg.det(matrix) <= 0:
        raise RefineRefused("the points lie mirrored to the cubes' calibration: is the camera the same one, "
                            "unmoved since CALIBRATE?")
    left, scales, right = np.linalg.svd(matrix)
    return left @ right, right.T @ np.diag(scales) @ right


def _check_stretch(refined: CameraFit, stretch: np.ndarray) -> None:
    """Refuse a stretch or a skew the arm cannot have, as the cubes' fit does: a point far off, most likely."""
    scales, skew = np.linalg.eigvalsh(stretch), abs(refined.arm_scale()[2])
    if scales.min() < 1 / MAX_STRETCH or scales.max() > MAX_STRETCH or skew > MAX_SKEW_DEG:
        raise RefineRefused(f"the points make the arm's x and y {scales.min():.2f} to {scales.max():.2f} times as "
                            f"long as the camera sees them, and askew by {skew:.1f} degrees: delete the points "
                            f"furthest off")
