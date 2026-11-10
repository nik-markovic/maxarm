#!/usr/bin/env python3
"""Which way each point misses, out from the base (+ further out) and round it (+ away from the centre line):
through the camera of all the points, and through each three without it (median). No arm.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/which-way.py camera.json points.json
"""
import itertools, json, math, sys
from pathlib import Path
import numpy as np
from applib.camera.fit import EDGE_HUBER_PX, HANDEDNESS, CameraFit
from applib.camera.model import least_squares
from applib.camera.store import RefinePoint
from camera import refine

fit = CameraFit.from_json(json.loads(Path(sys.argv[1]).read_text()))
pts = {p["id"]: RefinePoint(**p).sighting for p in json.loads(Path(sys.argv[2]).read_text())["points"]}
ids = list(pts)


def camera(chosen, start):
    on_arm = refine._OnArm(fit.frame_size, [pts[k] for k in chosen], start)
    return refine._pinhole(least_squares(on_arm.residuals, refine._numbers(start), refine.CAMERA_STEPS,
                                         huber=EDGE_HUBER_PX, iterations=200, derivatives=on_arm.derivatives)[0]).matrix


def polar(k, m):
    s = pts[k]
    d = HANDEDNESS @ refine._seen(m, s) - (s.x, s.y)
    b = math.atan2(s.x, -s.y)
    side = 1.0 if s.x >= 0 else -1.0
    return d @ (math.sin(b), -math.cos(b)), side * (d @ (math.cos(b), math.sin(b)))


every = camera(ids, fit.cube_fit().true_to_pixel)
others = {k: [] for k in ids}
for trio in itertools.combinations(ids, 3):
    m = camera(trio, every)
    for k in ids:
        if k not in trio:
            others[k].append(polar(k, m))
for k in ids:
    s = pts[k]
    o, r = polar(k, every)
    mo, mr = np.median([v[0] for v in others[k]]), np.median([v[1] for v in others[k]])
    print(f"#{k} reach {math.hypot(s.x, s.y):5.1f} bearing {math.degrees(math.atan2(s.x, -s.y)):6.1f}: all six: out {o:+5.2f} "
          f"round {r:+5.2f};  from threes without it (median): out {mo:+5.2f} round {mr:+5.2f}")
