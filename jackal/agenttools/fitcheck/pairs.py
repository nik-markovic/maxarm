#!/usr/bin/env python3
"""Points against each other, the camera fit aside: for two points, how far the cubes moved as the camera sees them
against how far the readings moved -- through several cameras (the first three points', all the points', each
three), which agree on a short move whatever they disagree on elsewhere. Desk mm; out = away from the base axis,
round = toward +bearing, at the first point of the pair. No arm, no app.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/pairs.py camera.json refine-points.json 1-6,2-5
"""
import itertools
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
stored = json.loads(Path(sys.argv[2]).read_text())["points"]
PAIRS = [tuple(int(k) for k in pair.split("-")) for pair in sys.argv[3].split(",")] if len(sys.argv) > 3 else []
points = {p["id"]: RefinePoint(**p).sighting for p in stored}
ids = list(points)
start = fit.cube_fit().true_to_pixel


def camera(chosen, first=start):
    sightings = [points[k] for k in chosen]
    on_arm = refine._OnArm(fit.frame_size, sightings, first)
    n = least_squares(on_arm.residuals, refine._numbers(first), refine.CAMERA_STEPS, huber=EDGE_HUBER_PX,
                      iterations=200, derivatives=on_arm.derivatives)[0]
    return refine._pinhole(n).matrix


def seen(matrix, k):
    return HANDEDNESS @ refine._seen(matrix, points[k])


def polar(vector, at):
    b = math.atan2(at[0], -at[1])
    out_dir = np.array([math.sin(b), -math.cos(b)])
    round_dir = np.array([math.cos(b), math.sin(b)])
    return vector @ out_dir, vector @ round_dir


first3 = camera(ids[:3])
cameras = {"first three": first3, "all": camera(ids, first3)}
for trio in itertools.combinations(ids, 3):
    if list(trio) != ids[:3]:
        cameras["+".join(map(str, trio))] = camera(trio, first3)
for name, matrix in cameras.items():
    n = refine._numbers(matrix)
    print(f"camera {name:12}: f {n[0]:6.0f} centre ({n[1]:5.0f}, {n[2]:5.0f}) at "
          f"{np.round(refine._pinhole(n).position * [-1, 1, 1], 1).tolist()} (desk x)")
print("\neach point, seen less read, out / round, mm:")
for k in ids:
    p = points[k]
    row = []
    for name, matrix in cameras.items():
        d = seen(matrix, k) - (p.x, p.y)
        row.append("%5.2f/%5.2f" % polar(d, (p.x, p.y)))
    print(f"  #{k:2} ({p.x:7.1f},{p.y:7.1f}): " + "  ".join(row))
for a, b in PAIRS:
    pa, pb = points[a], points[b]
    read = np.array([pb.x - pa.x, pb.y - pa.y])
    print(f"\npair #{a}-#{b}: readings {np.round(read, 2).tolist()} apart ({np.linalg.norm(read):.2f} mm)")
    for name, matrix in cameras.items():
        moved = seen(matrix, b) - seen(matrix, a)
        off = moved - read
        o, r = polar(off, (pa.x, pa.y))
        print(f"  {name:12}: cubes moved {np.round(moved, 2).tolist()}, less the readings {np.round(off, 2).tolist()}"
              f" = {np.linalg.norm(off):.2f} mm (out {o:+.2f}, round {r:+.2f})")
