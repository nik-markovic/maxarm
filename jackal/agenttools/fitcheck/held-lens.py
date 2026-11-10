#!/usr/bin/env python3
"""The lens held (focal length, principal point from another fit), only where the camera stands and points fitted
to the points given; each other point predicted. No arm.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/held-lens.py camera.json points.json fit-ids f cx cy
"""
import json, math, sys
from pathlib import Path
import numpy as np
from applib.camera.fit import EDGE_HUBER_PX, HANDEDNESS, CameraFit
from applib.camera.model import least_squares
from applib.camera.store import RefinePoint
from camera import refine

fit = CameraFit.from_json(json.loads(Path(sys.argv[1]).read_text()))
points = {p["id"]: RefinePoint(**p).sighting for p in json.loads(Path(sys.argv[2]).read_text())["points"]}
used = [points[int(k)] for k in sys.argv[3].split(",")]
lens = [float(v) for v in sys.argv[4:7]]
first = fit.cube_fit().true_to_pixel
start = refine._numbers(first)
on_arm = refine._OnArm(fit.frame_size, used, first)
free = None
if lens[0] > 0:
    start[0:3] = lens
    free = [3, 4, 5, 6, 7, 8]
n = least_squares(on_arm.residuals, start, refine.CAMERA_STEPS, huber=EDGE_HUBER_PX, iterations=300,
                  derivatives=on_arm.derivatives, free=free)[0]
m = refine._pinhole(n).matrix
print(f"lens f {n[0]:.0f} centre ({n[1]:.0f}, {n[2]:.0f}), fitted to {sys.argv[3]}")
for k, p in points.items():
    d = HANDEDNESS @ refine._seen(m, p) - (p.x, p.y)
    print(f"  #{k} ({p.x:7.1f},{p.y:7.1f}): {np.linalg.norm(d):5.2f} mm")
