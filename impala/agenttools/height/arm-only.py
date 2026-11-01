#!/usr/bin/env python3
"""CALIBRATE's three cubes: the camera as the cubes' shapes give it, against cameras forced onto the arm's readings.

    .venv/bin/python impala/agenttools/height/arm-only.py [config-dir]

1. CALIBRATE's camera: fitted to the cubes' edges (40 mm cubes), the arm turned and shifted onto it.
2. SAVE's fit on the three points alone, its focal-length guard lifted: each cube under the cup and 40 mm tall.
3. The same camera's lens and centre held, its place and aim fitted so each cube is under the cup (height free).

For each: where each cube lands from its arm reading (XY), its height from 40 (Z), how far the cube's edges found in
the picture lie from the cube drawn through that camera (px, rms), and the camera's focal length and height above the
desk. Nothing is written.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera import check, refine, store                                    # noqa: E402
from camera.fit import HANDEDNESS, CameraFit                                # noqa: E402
from camera.model import least_squares                                      # noqa: E402
from camera.sighting import in_frame, measure, misses, start               # noqa: E402


def report(name, fit, points):
    camera = refine._pinhole(refine._numbers(fit.true_to_pixel))
    print(f"{name}: focal {refine._numbers(fit.true_to_pixel)[0]:.0f} px, camera {camera.position[2]:.0f} mm above "
          f"the desk")
    for point in points:
        matrix = in_frame(fit.true_to_pixel, fit.frame_size, point)
        cube = measure(matrix, point, start(matrix, point))
        edges = misses(matrix, point, cube)
        off = refine.off_by(fit, point)
        print(f"  {point.cube:5s} XY {off.across:5.2f} mm  Z {off.height:+6.2f} mm  edges {math.sqrt(np.mean(edges ** 2)):.2f} px")


def pose_only(fit, points):
    """Lens and centre held; place and aim fitted so each cube's base is under the cup, its height free."""
    numbers = refine._numbers(fit.true_to_pixel)
    arm = [HANDEDNESS @ (point.x, point.y) for point in points]
    cubes = [start(in_frame(fit.true_to_pixel, fit.frame_size, point), point) for point in points]

    def residuals(trial):
        parts = []
        for index, point in enumerate(points):
            matrix = in_frame(refine._pinhole(trial).matrix, fit.frame_size, point)
            cubes[index] = measure(matrix, point, cubes[index])
            parts.append(cubes[index][:2] - arm[index])
        return np.concatenate(parts)

    numbers = least_squares(residuals, numbers, refine.CAMERA_STEPS, huber=math.inf, free=range(3, 9),
                            iterations=60)[0]
    return CameraFit(refine._pinhole(numbers).matrix, HANDEDNESS.copy(), fit.cubes, fit.frame_size, fit.edge_px,
                     fit.edges_used)


def main():
    config = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "config"
    fit = CameraFit.from_json(json.loads((config / "camera.json").read_text())).cube_fit()
    points = [store.RefinePoint(**point).sighting for point in
              json.loads((config / "camera-refine-points.json").read_text())["points"]]
    for point, stored in zip(points, json.loads((config / "camera-refine-points.json").read_text())["points"]):
        object.__setattr__(point, "cube", stored["cube"])
    report("1. CALIBRATE (cubes' shapes, arm turned and shifted)", fit, points)
    refine.MAX_FOCAL_CHANGE = math.inf
    forced = refine.refine(fit, points)
    report("2. SAVE's fit on the three, unguarded (under the cup, 40 mm tall)", forced, points)
    print(f"   desk in reach moves from 1.: {check.difference_mm(fit, forced)} mm (median, worst)")
    held = pose_only(fit, points)
    report("3. lens held, place and aim fitted (under the cup, height free)", held, points)
    print(f"   desk in reach moves from 1.: {check.difference_mm(fit, held)} mm (median, worst)")


if __name__ == "__main__":
    main()
