#!/usr/bin/env python3
"""Every three points as a calibration (SAVE's fit, cubes at their readings), each other point predicted from it;
and all points at once, each point's miss in it. No arm.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/triples.py camera.json points.json
"""
import itertools, json, sys
from pathlib import Path
import numpy as np
from applib.camera.fit import EDGE_HUBER_PX, HANDEDNESS, CameraFit
from applib.camera.model import least_squares
from applib.camera.store import RefinePoint
from camera import refine

fit = CameraFit.from_json(json.loads(Path(sys.argv[1]).read_text()))
pts = {p["id"]: RefinePoint(**p).sighting for p in json.loads(Path(sys.argv[2]).read_text())["points"]}
ids = list(pts)
first = fit.cube_fit().true_to_pixel


def camera(chosen, start=first):
    on_arm = refine._OnArm(fit.frame_size, [pts[k] for k in chosen], start)
    n = least_squares(on_arm.residuals, refine._numbers(start), refine.CAMERA_STEPS, huber=EDGE_HUBER_PX,
                      iterations=200, derivatives=on_arm.derivatives)[0]
    return n


def miss(n, k):
    s = pts[k]
    return HANDEDNESS @ refine._seen(refine._pinhole(n).matrix, s) - (s.x, s.y)


every = camera(ids)
print(f"all {len(ids)}: f {every[0]:.0f} aim ({every[1]:.0f},{every[2]:.0f}); each point's miss in it: "
      + "  ".join(f"#{k} {np.linalg.norm(miss(every, k)):.2f}" for k in ids))
table = {k: [] for k in ids}
for trio in itertools.combinations(ids, 3):
    n = camera(trio, refine._pinhole(every).matrix)
    rest = [k for k in ids if k not in trio]
    row = {k: float(np.linalg.norm(miss(n, k))) for k in rest}
    for k, v in row.items():
        table[k].append((v, trio))
    print(f"  from {trio}: f {n[0]:5.0f} aim ({n[1]:5.0f},{n[2]:5.0f}) -> " + "  ".join(f"#{k} {v:5.2f}" for k, v in row.items()))
print("each point predicted by the 10 threes without it: best, median, worst")
for k in ids:
    v = sorted(x for x, _ in table[k])
    best = min(table[k])[1]
    print(f"  #{k} ({pts[k].x:7.1f},{pts[k].y:7.1f}): {v[0]:.2f} (from {best}), {np.median(v):.2f}, {v[-1]:.2f}")
