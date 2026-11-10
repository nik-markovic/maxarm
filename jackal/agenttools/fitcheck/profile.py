#!/usr/bin/env python3
"""How loosely the cubes fix the camera: the focal length held at values round the fit's, the rest of the camera
refitted to the same points (each cube at its reading, edges), and how the fit's cost and where each point's cube is
seen change with it. No arm, no app.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/profile.py camera.json refine-points.json 1,2,3
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
on_arm = refine._OnArm(fit.frame_size, used, first)
best = least_squares(on_arm.residuals, refine._numbers(first), refine.CAMERA_STEPS, huber=EDGE_HUBER_PX,
                     iterations=200, derivatives=on_arm.derivatives)[0]
edge_count = sum(len(p.edges[2]) for p in used)
print(f"points {chosen}: {edge_count} edge points; best f {best[0]:.0f}, centre ({best[1]:.0f}, {best[2]:.0f}), "
      f"camera at {np.round(refine._pinhole(best).position * [-1, 1, 1], 1).tolist()} (desk x)")
header = "  f      cost+  rms px  centre        height  " + "  ".join(f"#{k} seen-read (dx, dy)" for k in points)
print(header)
start = best.copy()
for focal in np.arange(best[0] - 150, best[0] + 151, 25):
    trial = start.copy()
    trial[0] = focal
    on_arm = refine._OnArm(fit.frame_size, used, refine._pinhole(start).matrix)
    n, res, _ = least_squares(on_arm.residuals, trial, refine.CAMERA_STEPS, huber=EDGE_HUBER_PX, iterations=200,
                              derivatives=on_arm.derivatives, free=list(range(1, 9)))
    cost = _huber_cost(res, EDGE_HUBER_PX)
    if abs(focal - best[0]) < 1e-6:
        base_cost = cost
    matrix = refine._pinhole(n).matrix
    row = []
    for k, p in points.items():
        d = HANDEDNESS @ refine._seen(matrix, p) - (p.x, p.y)
        row.append(f"({d[0]:+5.2f},{d[1]:+5.2f})")
    rms = np.sqrt(np.mean(np.minimum(np.abs(res[:-2]), 3) ** 2))
    print(f"  {focal:5.0f}  {cost:7.1f}  {rms:5.3f}  ({n[1]:5.0f},{n[2]:5.0f})  {refine._pinhole(n).position[2]:6.1f}  "
          + "  ".join(f"{r:>18}" for r in row))
