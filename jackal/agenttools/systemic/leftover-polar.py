#!/usr/bin/env python3
"""What SAVE leaves at each point, out from the base (reach) and round it (+ counterclockwise seen from above), mm:
where the camera sees the cube less where the arm read the cup. With SAVE's stretch, and with a turn and shift only.

    ../.venv/bin/python agenttools/systemic/leftover-polar.py config_dir
"""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "calibration")]

from applib.camera import store                                                     # noqa: E402
from applib.camera.fit import HANDEDNESS                                            # noqa: E402
from camera import refine                                                           # noqa: E402

config = Path(sys.argv[1])
fit = store.load_camera(config)
points = [p.sighting for p in store.load_refine_points(config, fit)]
arm = np.array([(p.x, p.y) for p in points])
seen_true = np.array([refine._seen(refine.in_frame(fit.true_to_pixel, fit.frame_size, p), p) for p in points])
seen_save = np.linalg.solve(fit.desk_to_true, seen_true.T).T              # SAVE's desk: what the app sends the arm to

# A turn and a shift only, the mirror fixed: the camera's true mm taken as the arm's.
mirrored = seen_true @ HANDEDNESS.T
a, b = arm - arm.mean(0), mirrored - mirrored.mean(0)
u, _, vt = np.linalg.svd(b.T @ a)
turn = u @ vt
seen_rigid = (mirrored - mirrored.mean(0)) @ turn + arm.mean(0)


def polar(seen):
    rows = []
    for (x, y), (sx, sy) in zip(arm, seen):
        bearing = math.atan2(x, -y)
        out = np.array([math.sin(bearing), -math.cos(bearing)])
        clockwise = np.array([math.cos(bearing), math.sin(bearing)])      # toward +x: clockwise from above (mirror)
        d = np.array([sx - x, sy - y])
        rows.append((math.degrees(bearing), math.hypot(x, y), d @ out, -(d @ clockwise)))
    return np.array(rows)


for name, seen in (("SAVE (turn, shift, stretch)", seen_save), ("turn and shift only", seen_rigid)):
    rows = polar(seen)
    print(f"{name}: reach {np.mean(rows[:, 2]):+.2f} mean; round {np.mean(rows[:, 3]):+.2f} mean")
    print("   bearing  reach_out   seen-arm: out   round(+ccw)")
    for row in sorted(rows.tolist()):
        print(f"   {row[0]:+7.1f}  {row[1]:7.1f}      {row[2]:+6.2f}      {row[3]:+6.2f}")
