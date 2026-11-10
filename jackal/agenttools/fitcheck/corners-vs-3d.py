#!/usr/bin/env python3
"""tops.py's corners (where the found lines cross) against the 3-D fit's: the all-points camera (cubes at their
readings), and each cube placed by its own edges through it; per corner, pixels.

    PYTHONPATH=src:src/calibration:agenttools/fitcheck ../.venv/bin/python3 agenttools/fitcheck/corners-vs-3d.py
"""
import json, math
from pathlib import Path
import numpy as np
from applib.camera.fit import HANDEDNESS, CameraFit
from applib.camera.model import cube_corners, project
from applib.camera.store import RefinePoint
from camera import refine
from tops import POINTS, top_face
from pretend import fit_camera

fit = CameraFit.from_json(json.loads(Path("config/camera.json").read_text()))
stored = json.loads(POINTS.read_text())["points"]
points = [RefinePoint(**p) for p in stored]
s = [p.sighting for p in points]
numbers, cost, _ = fit_camera(fit.frame_size, s, fit.cube_fit().true_to_pixel)
m = refine._pinhole(numbers).matrix
for p, sight in zip(stored, s):
    mine, _ = top_face(p)
    placed = refine.place(m, sight, refine.start(m, sight)[:3])
    model = project(m, cube_corners(placed[:2], placed[2]))
    d = [np.linalg.norm(mine[k] - model[k]) for k in (4, 5, 6, 7)]
    print(f"#{p['id']}: lines' corners vs the 3-D cube's, px: " + " ".join(f"{v:5.2f}" for v in d)
          + f"   top side ~{np.linalg.norm(model[5] - model[4]):.0f} px")
