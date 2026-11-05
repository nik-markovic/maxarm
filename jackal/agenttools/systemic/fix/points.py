#!/usr/bin/env python3
"""The 4 Nov points as SAVE sees them: each point's reading against where the camera puts its cube, and against
where the other points put it (left out), out from the base and round it (+ toward +x), mm.

    points.py [config_dir]        # default work/config-jackal-1104
"""
import math
import sys
from pathlib import Path

import numpy as np

JACKAL = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(JACKAL / "src"), str(JACKAL / "src" / "calibration")]
from applib.camera import store  # noqa: E402
from camera import refine  # noqa: E402

config = Path(sys.argv[1]) if len(sys.argv) > 1 else JACKAL.parent / "work" / "config-jackal-1104"
fit = store.load_camera(config)
stored = store.load_refine_points(config, fit)
points = [p.sighting for p in stored]
refined = refine.refine(fit, points)
print("refined:", refined.refined, "camera at", np.round(refined.camera_position, 1), "focal", round(refined.focal))
print("saved:  ", fit.refined, "camera at", np.round(fit.camera_position, 1), "focal", round(fit.focal))
others = refine.left_out(refined, points)
print(" id  cube   taken          x       y      z   bear  reach |  seen-arm out round | leftout out round | height")
for p, s, put in zip(stored, points, others):
    off = refine.off_by(refined, s)
    b = math.atan2(s.x, -s.y)
    out, rnd = np.array([math.sin(b), -math.cos(b)]), np.array([math.cos(b), math.sin(b)])
    d = off.seen - (s.x, s.y)
    lo = np.linalg.solve(refined.desk_to_true, refined.desk_to_true @ off.seen) 
    dl = off.seen - np.linalg.solve(refined.desk_to_true, put)
    print(f"{p.id:3d}  {str(getattr(p, 'cube', None))[:5]:5s} {getattr(p, 'taken', ''):13s} {s.x:7.1f} {s.y:7.1f} "
          f"{getattr(p, 'z', 0):5.1f} {math.degrees(b):+6.1f} {math.hypot(s.x, s.y):6.1f} | {d @ out:+6.2f} {d @ rnd:+6.2f}"
          f"       | {dl @ out:+6.2f} {dl @ rnd:+6.2f}    | {off.height if off.height is None else round(off.height, 2)}")
