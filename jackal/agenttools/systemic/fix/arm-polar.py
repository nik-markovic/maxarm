#!/usr/bin/env python3
"""The arm's readings with a polar correction about the base axis -- reach + c, reach x k, bearing x (1 + s),
or a turning wave -- then laid on the cubes by a turn and a shift. The camera: the app's, fitted to every point's
edges alone (lens-vs-arm.py). Each point left out of the whole fit, the correction too.

    arm-polar.py [config_dir]
"""
import importlib.util
import itertools
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lva", HERE / "lens-vs-arm.py")
lva = importlib.util.module_from_spec(spec)
sys.argv = sys.argv[:2]
spec.loader.exec_module(lva)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tie import link_arm  # noqa: E402

camera, cubes, _ = lva.fit_camera([0, 2, 3, 4, 5, 6, 7, 8, 9])
arm, ids = lva.arm, [p.id for p in lva.stored]


def corrected(xy, c=0.0, k=1.0, s=0.0, w=0.0, b0=0.0):
    r, b = np.hypot(xy[:, 0], xy[:, 1]), np.arctan2(xy[:, 0], -xy[:, 1])
    b = b + s * (b - b0) + math.radians(w) * np.sin(2 * b)
    r = r * k + c
    return np.column_stack((r * np.sin(b), -r * np.cos(b)))


def misses(params, names, rows):
    a = corrected(arm[rows], **dict(zip(names, params)))
    t, sh, m = link_arm(a, cubes[rows])
    return ((t @ m @ a.T).T + sh - cubes[rows]).ravel(), (t, sh, m)


def fit(names, rows):
    x = np.zeros(len(names)) + np.array([1.0 if n == "k" else 0.0 for n in names])
    for _ in range(30):
        r, _ = misses(x, names, rows)
        J = np.empty((len(r), len(names)))
        for i in range(len(names)):
            xx = x.copy(); xx[i] += 1e-6
            J[:, i] = (misses(xx, names, rows)[0] - r) / 1e-6
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        x += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return x


def run(names):
    rows = list(range(len(arm)))
    x = fit(names, rows) if names else np.zeros(0)
    r, _ = misses(x, names, rows)
    d = np.linalg.norm(r.reshape(-1, 2), axis=1)
    lo = []
    for i in rows:
        rest = [j for j in rows if j != i]
        xi = fit(names, rest) if names else np.zeros(0)
        _, (t, sh, m) = misses(xi, names, rest)
        a = corrected(arm[[i]], **dict(zip(names, xi)))[0]
        lo.append(np.linalg.norm(t @ m @ a + sh - cubes[i]))
    shown = ", ".join(f"{n} {v:+.4f}" for n, v in zip(names, x)) or "nothing"
    print(f"{shown:44s} fit {np.median(d):.2f} typ {d.max():.2f} worst | left out {np.median(lo):.2f} typ "
          f"{max(lo):.2f} worst  [{' '.join(f'{v:.1f}' for v in lo)}]")


print("ids", ids)
for names in ([], ["c"], ["k"], ["s"], ["w"], ["c", "s"], ["c", "k"], ["c", "w"], ["k", "s"], ["c", "k", "s"],
              ["c", "s", "w"]):
    run(names)
