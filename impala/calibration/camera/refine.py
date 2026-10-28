"""The camera calibration refined by the red cube, set down in place after place, centred under the cup, and seen.

    points = [Sighting(x, y, pixel, frame_size), ...]   # desk x, y: where the arm says the cup is, over the cube
    refined = refine(fit, points)                         # fit: the saved calibration, store.load_camera()
    off_mm(refined, point)                                # how far from the arm's point the camera puts the cube

A point pairs where the arm says the cup is, over the cube's centre, with where
the camera sees that centre: the pixel under it, on the desk (`locate.py`). The
camera is the calibration's, held as it is: through it the pixel is a place on
the desk in true millimetres. Where the arm stands on the desk, and which way it
faces, is then fitted to every point at once, the three cubes the camera was
fitted to among them: a turn and a shift, with the board's mirror, least squares
(`fit.place_arm()`), as CALIBRATE places it on the three cubes alone. The arm's x
and y are true millimetres, as the camera's are, so nothing is stretched; each
point's own error stays its own instead of bending the desk round it.

`camera.json` keeps its form: the camera's true millimetres are turned and
shifted onto the arm's, so that `desk_to_true` is the board's mirror alone.
Points are pixels, which no refinement changes, so a calibration refined again
is refined from the same points.
"""

import dataclasses
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .fit import HANDEDNESS, CameraFit, laid_along_arm, place_arm
from .model import rescaled


@dataclass(frozen=True)
class Sighting:
    x: float                         # desk mm: where the arm says the cup is, over the cube's centre
    y: float
    pixel: Tuple[float, float]       # where the camera sees the desk under the cube's centre
    frame_size: Tuple[int, int]      # the frame's: the calibration's, or the same picture smaller


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
    """`fit` with the arm placed again, on its three cubes and `points` together; the camera as it is. With no
    points, the three cubes' own calibration: the arm placed on them alone, as CALIBRATE places it."""
    alone = _placed(fit, cube_sightings(fit))
    if not points:
        return alone
    every = cube_sightings(fit) + list(points)
    refined = _placed(fit, every)
    offs = [off_mm(refined, point) for point in every]
    return dataclasses.replace(refined, refined={"points": len(points),
                                                 "off_mm": [round(float(np.median(offs)), 2), round(max(offs), 2)]},
                               cubes_alone={"true_to_pixel": alone.true_to_pixel.tolist(),
                                            "desk_to_true": alone.desk_to_true.tolist(), "cubes": alone.cubes})


def _placed(fit: CameraFit, sightings: Sequence[Sighting]) -> CameraFit:
    """`fit`'s camera with the arm placed on `sightings`: a turn and a shift, the best for all of them."""
    turn, shift = place_arm(np.array([(point.x, point.y) for point in sightings]),
                            np.array([seen_true(fit, point) for point in sightings]))
    true_to_pixel, cubes = laid_along_arm(fit.true_to_pixel, fit.cubes, turn, shift)
    return CameraFit(true_to_pixel, HANDEDNESS.copy(), cubes, fit.frame_size, fit.edge_px, fit.edges_used)
