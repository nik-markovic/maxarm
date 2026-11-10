#!/usr/bin/env python3
"""Several point files as one set, through one camera (SAVE's fit, every cube at its reading): each point's miss in
that camera, and left out. Points from frames taken hours apart fit one camera only if it did not move.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/merged.py camera.json points.json [points.json ...]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

from applib.camera.fit import EDGE_HUBER_PX, HANDEDNESS, CameraFit
from applib.camera.model import least_squares
from applib.camera.store import RefinePoint
from camera import refine

fit = CameraFit.from_json(json.loads(Path(sys.argv[1]).read_text()))
labels, sightings = [], []
for file_index, name in enumerate(sys.argv[2:]):
    for p in json.loads(Path(name).read_text())["points"]:
        labels.append(f"{'ABCDEFG'[file_index]}{p['id']} {p['taken'][:12]}")
        sightings.append(RefinePoint(**p).sighting)
first = fit.cube_fit().true_to_pixel


def camera(chosen, start=first):
    on_arm = refine._OnArm(fit.frame_size, chosen, start)
    n = least_squares(on_arm.residuals, refine._numbers(start), refine.CAMERA_STEPS, huber=EDGE_HUBER_PX,
                      iterations=300, derivatives=on_arm.derivatives)[0]
    return refine._pinhole(n).matrix


def polar(d, x, y):
    b = math.atan2(x, -y)
    return d @ (math.sin(b), -math.cos(b)), d @ (math.cos(b), math.sin(b))


every = camera(sightings)
n = refine._numbers(every)
print(f"{len(sightings)} points, one camera: f {n[0]:.0f}, centre ({n[1]:.0f}, {n[2]:.0f}), at "
      f"{np.round(refine._pinhole(n).position * [-1, 1, 1], 1).tolist()} (desk x)")
for k, (label, s) in enumerate(zip(labels, sightings)):
    d = HANDEDNESS @ refine._seen(every, s) - (s.x, s.y)
    others = camera([o for j, o in enumerate(sightings) if j != k], every)
    e = HANDEDNESS @ refine._seen(others, s) - (s.x, s.y)
    print(f"  {label:18} ({s.x:7.1f},{s.y:7.1f}): in the fit {np.linalg.norm(d):4.2f} (out {polar(d, s.x, s.y)[0]:+5.2f} "
          f"round {polar(d, s.x, s.y)[1]:+5.2f}); left out {np.linalg.norm(e):4.2f} (out {polar(e, s.x, s.y)[0]:+5.2f} "
          f"round {polar(e, s.x, s.y)[1]:+5.2f})")
