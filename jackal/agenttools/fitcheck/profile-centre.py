#!/usr/bin/env python3
"""As profile.py, the principal point held instead of the focal length: a grid of (cx, cy) round the fit's.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/profile-centre.py camera.json points.json 1,2,3
"""
import json
import sys
from pathlib import Path

import numpy as np

from applib.camera.fit import EDGE_HUBER_PX, HANDEDNESS, CameraFit
from applib.camera.model import least_squares, _huber_cost
from applib.camera.store import RefinePoint
from camera import refine

fit = CameraFit.from_json(json.loads(Path(sys.argv[1]).read_text()))
stored = json.loads(Path(sys.argv[2]).read_text())["points"]
chosen = [int(k) for k in sys.argv[3].split(",")]
points = {p["id"]: RefinePoint(**p).sighting for p in stored}
used = [points[k] for k in chosen]
first = fit.cube_fit().true_to_pixel


class Unheld(refine._OnArm):
    """The principal point not held to the frame's middle: what the edges alone say."""
    def held(self, numbers):
        return np.zeros(2)

    def held_rows(self, count):
        return np.zeros((2, count))


on_arm = Unheld(fit.frame_size, used, first)
best = least_squares(on_arm.residuals, refine._numbers(first), refine.CAMERA_STEPS, huber=EDGE_HUBER_PX,
                     iterations=200, derivatives=on_arm.derivatives)[0]
_, res0, _ = least_squares(on_arm.residuals, best, refine.CAMERA_STEPS, huber=EDGE_HUBER_PX, iterations=1,
                           derivatives=on_arm.derivatives)
base = _huber_cost(res0, EDGE_HUBER_PX)
print(f"points {chosen}, centre free: f {best[0]:.0f}, centre ({best[1]:.0f}, {best[2]:.0f}), cost {base:.1f}")
for dx in (-120, -60, 0, 60, 120):
    for dy in (-120, -60, 0, 60, 120):
        trial = best.copy()
        trial[1] += dx
        trial[2] += dy
        on_arm = Unheld(fit.frame_size, used, refine._pinhole(best).matrix)
        n, res, _ = least_squares(on_arm.residuals, trial, refine.CAMERA_STEPS, huber=EDGE_HUBER_PX, iterations=200,
                                  derivatives=on_arm.derivatives, free=[0, 3, 4, 5, 6, 7, 8])
        matrix = refine._pinhole(n).matrix
        row = []
        for k, p in points.items():
            d = HANDEDNESS @ refine._seen(matrix, p) - (p.x, p.y)
            row.append(f"#{k}({d[0]:+5.2f},{d[1]:+5.2f})")
        print(f"  centre ({trial[1]:5.0f},{trial[2]:5.0f}) f {n[0]:5.0f}: cost +{_huber_cost(res, EDGE_HUBER_PX) - base:6.1f}  "
              + " ".join(row))
