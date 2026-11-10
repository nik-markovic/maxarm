#!/usr/bin/env python3
"""A cup sitting off the arm's line: every reading moved `side` mm square to the arm's direction (+ toward +bearing)
and `out` mm along it, then the camera fitted to all points and each point left out. No arm.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/cup-offset.py camera.json points.json
"""
import dataclasses, json, math, sys
from pathlib import Path
import numpy as np
from applib.camera.fit import EDGE_HUBER_PX, HANDEDNESS, CameraFit
from applib.camera.model import least_squares
from applib.camera.store import RefinePoint
from camera import refine

fit = CameraFit.from_json(json.loads(Path(sys.argv[1]).read_text()))
base = [RefinePoint(**p).sighting for p in json.loads(Path(sys.argv[2]).read_text())["points"]]
first = fit.cube_fit().true_to_pixel


def moved(side, out):
    out_pts = []
    for s in base:
        b = math.atan2(s.x, -s.y)
        dx = out * math.sin(b) + side * math.cos(b)
        dy = -out * math.cos(b) + side * math.sin(b)
        out_pts.append(dataclasses.replace(s, x=s.x + dx, y=s.y + dy))
    return out_pts


def camera(pts, start):
    on_arm = refine._OnArm(fit.frame_size, pts, start)
    n = least_squares(on_arm.residuals, refine._numbers(start), refine.CAMERA_STEPS, huber=EDGE_HUBER_PX,
                      iterations=120, derivatives=on_arm.derivatives)[0]
    return refine._pinhole(n).matrix


def miss(m, s):
    return float(np.linalg.norm(HANDEDNESS @ refine._seen(m, s) - (s.x, s.y)))


grid = [(float(a), 0.0) for a in sys.argv[3].split(",")] if len(sys.argv) > 3 else \
    [(t, 0.0) for t in (-4, -3, -2, -1, 0, 1, 2, 3, 4)]
if len(sys.argv) > 4:
    grid = [(float(sys.argv[3]), float(o)) for o in sys.argv[4].split(",")]
for side, out in grid:
    pts = moved(side, out)
    every = camera(pts, first)
    infit = [miss(every, s) for s in pts]
    left = [miss(camera([o for j, o in enumerate(pts) if j != k], every), pts[k]) for k in range(len(pts))]
    print(f"cup {side:+.1f} mm sideways, {out:+.1f} out: in the fit {' '.join(f'{v:4.2f}' for v in infit)}; "
          f"left out {' '.join(f'{v:4.2f}' for v in left)} (median {np.median(left):.2f}, worst {max(left):.2f})")
