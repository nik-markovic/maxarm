#!/usr/bin/env python3
"""How far the red cube's points sit from the camera calibration, and how much of it the base's turning explains.

    ../.venv/bin/python agenttools/check-refine-points.py              # config/ as it is; no arm, no camera

Reads `camera.json` (the three cubes' own camera, `cubes_alone` if refined) and
`camera-refine-points.json`. Each point and cube is a desk point the arm reported
and a place the camera saw; the arm's x/y error is fitted to them as a turn,
stretch and shift alone (what `camera.json` holds), and again with the base's
angle off by a wave that repeats every half turn (a sin 2b + c cos 2b). Each
point is also predicted from a fit to all the others, so that a model is judged
on points it was not fitted to.
"""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera import store                                      # noqa: E402
from camera.fit import HANDEDNESS                             # noqa: E402
from camera.refine import cube_sightings, seen_true           # noqa: E402


def fields(desk: np.ndarray, is_wave: bool) -> np.ndarray:
    """The columns of the linear fit: a turn, stretch and shift; and the angle wave's two, as mm at 200 mm out."""
    n = len(desk)
    rows = np.zeros((2 * n, 8 if is_wave else 6))
    rows[0::2, 0:2], rows[0::2, 2] = desk, 1.0
    rows[1::2, 3:5], rows[1::2, 5] = desk, 1.0
    if is_wave:
        reach, bearing = np.hypot(*desk.T), np.arctan2(desk[:, 0], -desk[:, 1])
        across = np.column_stack((np.cos(bearing), np.sin(bearing))) @ HANDEDNESS * (reach / 200.0)[:, None]
        for column, wave in ((6, np.sin(2 * bearing)), (7, np.cos(2 * bearing))):
            rows[0::2, column], rows[1::2, column] = across[:, 0] * wave, across[:, 1] * wave
    return rows


def offs(desk: np.ndarray, true: np.ndarray, is_wave: bool):
    """How far off each point is, fitted with all and predicted from the others; and the fit's numbers."""
    rows, target = fields(desk, is_wave), true.ravel()
    solution = np.linalg.lstsq(rows, target, rcond=None)[0]
    fitted = np.linalg.norm((rows @ solution - target).reshape(-1, 2), axis=1)
    predicted = []
    for index in range(len(desk)):
        keep = np.repeat(np.arange(len(desk)) != index, 2)
        alone = np.linalg.lstsq(rows[keep], target[keep], rcond=None)[0]
        predicted.append(np.linalg.norm((rows[~keep] @ alone - target[~keep])))
    return fitted, np.array(predicted), solution


def main() -> int:
    config = ROOT / "config"
    saved = store.load_camera(config)
    if saved is None:
        print("no camera.json")
        return 1
    alone = saved.cube_fit()
    sightings = [point.sighting for point in store.load_refine_points(config)] + cube_sightings(alone)
    desk = np.array([(sighting.x, sighting.y) for sighting in sightings])
    true = np.array([seen_true(alone, sighting) for sighting in sightings])
    print(f"{len(sightings) - 3} red cube points and the 3 cubes")
    for name, is_wave in (("turn, stretch, shift", False), ("and the base's angle wave", True)):
        fitted, predicted, solution = offs(desk, true, is_wave)
        line = (f"{name:26s}: fitted {np.median(fitted):.2f} mm typically, {np.percentile(fitted, 90):.2f} for 90%, "
                f"{fitted.max():.2f} worst; each from the others {np.median(predicted):.2f}, "
                f"{np.percentile(predicted, 90):.2f}, {predicted.max():.2f}")
        if is_wave:
            size = math.degrees(math.hypot(solution[6], solution[7]) / 200.0)
            peak = math.degrees(0.5 * math.atan2(solution[6], solution[7]))
            line += f"\n{'':28s}the wave: {size:.2f} deg, largest at bearing {peak:.0f} and every 90 deg on"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
