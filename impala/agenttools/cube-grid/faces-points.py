#!/usr/bin/env python3
"""Fine Tune's points placed each way faces.py fits a cube (box, top alone, lifted, tall), through each day's SAVE
camera, and the arm laid on them: which way makes the arm and the camera agree best? No arm.

    .venv/bin/python impala/agenttools/cube-grid/faces-points.py
"""

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
from camera.refine import refine                                           # noqa: E402
from camera.sighting import in_frame, place, start                         # noqa: E402

spec = importlib.util.spec_from_file_location("faces", HERE / "faces.py")
faces = importlib.util.module_from_spec(spec)
spec.loader.exec_module(faces)
DAYS = {"today": ROOT / "config", "noon": ROOT.parent / "work" / "config-impala-1030"}


def main():
    for day, config in DAYS.items():
        fit = store.load_camera(config)
        points = store.load_refine_points(config)
        camera = refine(fit, [point.sighting for point in points]).true_to_pixel
        placed, extra = {}, {"tall": [], "top s": [], "lifted": []}
        for point in points:
            sighting = point.sighting
            matrix = in_frame(camera, tuple(fit.frame_size), sighting)
            first = place(matrix, sighting, start(matrix, sighting)[:3])
            result = faces.fits(matrix, sighting.edges, first)
            for way, (numbers, *_rest) in result.items():
                placed.setdefault(way, []).append(numbers[:2])
            for way in extra:
                extra[way].append(result[way][0][3])
        arm = np.array([(point.x, point.y) for point in points])
        print(f"{day}: {len(points)} points; tall {np.min(extra['tall']):.2f}-{np.max(extra['tall']):.2f} mm, top width "
              f"{np.min(extra['top s']):.2f}-{np.max(extra['top s']):.2f}, lifted bottom "
              f"{np.min(extra['lifted']):+.2f} to {np.max(extra['lifted']):+.2f}")
        for way, seen in placed.items():
            seen = np.array(seen)
            turn, shift, desk_to_true = link_arm(arm, seen)
            back = np.linalg.solve(desk_to_true, np.linalg.solve(turn, (seen - shift).T)).T
            off = np.hypot(*(back - arm).T)
            print(f"   {way:7s} misses rms {math.sqrt(np.mean(off ** 2)):.2f} mm, median {np.median(off):.2f}, worst "
                  f"{off.max():.2f}, green {np.sum(off <= 1)}/{len(off)}")


if __name__ == "__main__":
    main()
