#!/usr/bin/env python3
"""How well each camera puts a 40 mm cube on every point's edges, and where it stands; then the focal length held.

    ../.venv/bin/python agenttools/systemic/edges-by-camera.py [config_dir]
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "calibration")]

from applib.camera import store                                                     # noqa: E402
from applib.camera.model import CUBE_MM, least_squares                              # noqa: E402
from applib.camera.sighting import in_frame, measure, misses, place, start          # noqa: E402
from camera import refine                                                           # noqa: E402

config = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "config"
fit = store.load_camera(config)
alone = fit.cube_fit()
points = [point.sighting for point in store.load_refine_points(config, fit)]
seen = [point for point in points if point.edges is not None]


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def describe(name, true_to_pixel):
    numbers = refine._numbers(true_to_pixel)
    position = refine._pinhole(numbers).position
    print(f"{name}: focal {numbers[0]:.0f} px, camera at true ({position[0]:.1f}, {position[1]:.1f}), "
          f"{position[2]:.1f} mm over the desk, {np.hypot(*position[:2]):.1f} from the base axis")
    for point in seen:
        matrix = in_frame(true_to_pixel, fit.frame_size, point)
        cube = place(matrix, point, start(matrix, point)[:3])
        at_pixel = misses(matrix, point, start(matrix, point))           # the check image's: the stored pixel
        free = measure(matrix, point, start(matrix, point))
        print(f"   ({point.x:7.1f}, {point.y:7.1f})  edges {rms(misses(matrix, point, np.append(cube, CUBE_MM))):.2f} px"
              f"  drawn at its pixel {rms(at_pixel):.2f} px  height free {free[3]:.2f} mm")


describe("CALIBRATE, three cubes", alone.true_to_pixel)
describe("SAVE", fit.true_to_pixel)

print("\nThe camera fitted to all the points' edges with its focal length held (refine._camera, focal fixed):")
first = refine._numbers(alone.true_to_pixel)
width, height = fit.frame_size
middle = np.array([width / 2, height / 2])
for share in (0.80, 0.85, 0.90, 0.95, 1.00, 1.05):
    numbers = first.copy()
    numbers[0] *= share
    cubes = [start(in_frame(alone.true_to_pixel, fit.frame_size, point), point)[:3] for point in seen]

    def residuals(trial):
        parts = []
        for index, point in enumerate(seen):
            matrix = in_frame(refine._pinhole(trial).matrix, fit.frame_size, point)
            cubes[index] = place(matrix, point, cubes[index])
            parts.append(misses(matrix, point, np.append(cubes[index], CUBE_MM)))
        parts.append((trial[1:3] - middle) / (0.1 * width))
        return np.concatenate(parts)

    numbers = least_squares(residuals, numbers, refine.CAMERA_STEPS, huber=1.0, iterations=40,
                            free=range(1, 9))[0]
    found = residuals(numbers)[:-2]
    position = refine._pinhole(numbers).position
    print(f"   focal {numbers[0]:6.0f} px ({share:.2f}): edges {rms(found):.3f} px, camera {position[2]:.1f} mm "
          f"over the desk, {np.hypot(*position[:2]):.1f} from the base axis, true ({position[0]:.1f}, {position[1]:.1f})")
