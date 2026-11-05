#!/usr/bin/env python3
"""A constant offset in the arm's own frame, per session: each point's cube where the session's saved camera sees
it (edges, or the pixel for older points), the arm's readings moved by c mm out from the base (and t round it,
which a turn mostly takes), then laid on the cubes by a turn and a shift. Left out, typically and at worst.

    session-offset.py config_dir ...
"""
import math
import sys
from pathlib import Path

import numpy as np

JACKAL = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(JACKAL / "src"), str(JACKAL / "src" / "calibration")]
from applib.camera import store  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tie import link_arm  # noqa: E402
from camera import refine  # noqa: E402


def moved(arm, c, t):
    r, b = np.hypot(arm[:, 0], arm[:, 1]), np.arctan2(arm[:, 0], -arm[:, 1])
    out = np.column_stack((np.sin(b), -np.cos(b)))
    rnd = np.column_stack((np.cos(b), np.sin(b)))
    return arm + c * out + t * rnd


def misses(arm, seen, c, t, rows):
    a = moved(arm[rows], c, t)
    turn, shift, mirror = link_arm(a, seen[rows])
    return ((turn @ mirror @ a.T).T + shift - seen[rows]).ravel(), (turn, shift, mirror)


def fit(arm, seen, names, rows):
    x = np.zeros(2)
    free = [("c", "t").index(n) for n in names]
    for _ in range(25 if free else 0):
        r, _ = misses(arm, seen, *x, rows)
        J = np.column_stack([(misses(arm, seen, *(x + np.eye(2)[i] * 1e-5), rows)[0] - r) / 1e-5 for i in free])
        x[free] += np.linalg.lstsq(J, -r, rcond=None)[0]
    return x


for config in map(Path, sys.argv[1:]):
    saved = store.load_camera(config)
    points = [p.sighting for p in store.load_refine_points(config, saved)
              if p.cube is None]                                  # the session's own points, not CALIBRATE's cubes
    camera = refine.refine(saved, points) if any(p.edges is not None for p in points) else saved
    seen = np.array([refine._seen(refine.in_frame(camera.true_to_pixel, camera.frame_size, p), p) for p in points])
    arm = np.array([(p.x, p.y) for p in points])
    bear = np.degrees(np.arctan2(arm[:, 0], -arm[:, 1]))
    print(f"{config.name}: {len(points)} points, bearings {bear.min():+.0f} to {bear.max():+.0f}")
    for names in ([], ["c"], ["c", "t"]):
        x = fit(arm, seen, names, list(range(len(arm))))
        lo = []
        for i in range(len(arm)):
            rest = [j for j in range(len(arm)) if j != i]
            xi = fit(arm, seen, names, rest)
            _, (turn, shift, mirror) = misses(arm, seen, *xi, rest)
            lo.append(np.linalg.norm(turn @ mirror @ moved(arm[[i]], *xi)[0] + shift - seen[i]))
        print(f"   {'+'.join(names) or 'none':5s} c {x[0]:+5.2f} t {x[1]:+5.2f}: left out {np.median(lo):.2f} typ, "
              f"{max(lo):.2f} worst")
