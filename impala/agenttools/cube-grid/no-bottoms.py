#!/usr/bin/env python3
"""SAVE's camera fitted from every point's cube at once, as SAVE fits it, but without the cubes' bottom edges (the
tops and the upright edges only); then the cubes placed through it and the arm laid on them. No arm.

    .venv/bin/python impala/agenttools/cube-grid/no-bottoms.py
"""

import dataclasses
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera import store                                                   # noqa: E402
from camera.fit import link_arm                                            # noqa: E402
from camera.refine import _numbers, refine                                 # noqa: E402
from camera.sighting import in_frame, place, start                         # noqa: E402

spec = importlib.util.spec_from_file_location("faces", HERE / "faces.py")
faces = importlib.util.module_from_spec(spec)
spec.loader.exec_module(faces)
DAYS = {"today": ROOT / "config", "noon": ROOT.parent / "work" / "config-impala-1030"}


def without_bottoms(sighting):
    starts, ends, found = sighting.edges
    keep = ~((starts < 4) & (ends < 4))
    return dataclasses.replace(sighting, edges=(starts[keep], ends[keep], found[keep]))


def misses(points, camera, frame_size, way):
    seen = []
    for point in points:
        matrix = in_frame(camera, frame_size, point.sighting)
        first = place(matrix, point.sighting, start(matrix, point.sighting)[:3])
        seen.append(first[:2] if way == "box" else faces.fits(matrix, point.sighting.edges, first)[way][0][:2])
    arm = np.array([(point.x, point.y) for point in points])
    turn, shift, desk_to_true = link_arm(arm, np.array(seen))
    back = np.linalg.solve(desk_to_true, np.linalg.solve(turn, (np.array(seen) - shift).T)).T
    return np.hypot(*(back - arm).T)


def main():
    for day, config in DAYS.items():
        fit = store.load_camera(config)
        points = store.load_refine_points(config)
        size = tuple(fit.frame_size)
        cameras = {"all edges (SAVE)": refine(fit, [p.sighting for p in points]).true_to_pixel,
                   "no bottom edges": refine(fit, [without_bottoms(p.sighting) for p in points]).true_to_pixel}
        for name, camera in cameras.items():
            numbers = _numbers(camera)
            cells = []
            for way in ("box", "top"):
                off = misses(points, camera, size, way)
                cells.append(f"placed by {way}: {math.sqrt(np.mean(off ** 2)):.2f} mm rms, green {np.sum(off <= 1)}")
            print(f"{day:5s} camera from {name:17s} (focal {numbers[0]:.0f} px): " + "; ".join(cells))


if __name__ == "__main__":
    main()
