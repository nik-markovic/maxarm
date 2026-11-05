#!/usr/bin/env python3
"""How well the pictures fix the camera's scale: cameras fitted to the edges of different subsets of the points
(the arm has no say), and for each, the scale k and reach offset c the arm's readings need to lie on all the cubes
it places (turn and shift besides).

    scale-by-subset.py [config_dir]
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.argv = sys.argv[:2]
lva = load("lens-vs-arm")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tie import link_arm  # noqa: E402
from applib.camera.sighting import place, in_frame  # noqa: E402

ids = [p.id for p in lva.stored]
n = len(ids)
base = [0, 2, 3, 4, 5, 6, 7, 8, 9]


def all_cubes(camera):
    """Every point's cube placed through `camera` (its edges, 40 mm), true x, y."""
    out = []
    for p in lva.points:
        x = lva.start(lva.fit.true_to_pixel, p)[:3]
        x = lva.solve(lambda c: lva.misses(camera, p, c), x, [0, 1, 2], 30)
        out.append(x[:2])
    return np.array(out)


def radial(cubes, names):
    def corrected(params):
        d = dict(zip(names, params))
        r, b = np.hypot(*lva.arm.T), np.arctan2(lva.arm[:, 0], -lva.arm[:, 1])
        r = r * d.get("k", 1.0) + d.get("c", 0.0)
        return np.column_stack((r * np.sin(b), -r * np.cos(b)))

    def res(params):
        a = corrected(params)
        t, s, m = link_arm(a, cubes)
        return ((t @ m @ a.T).T + s - cubes).ravel()

    x = np.array([1.0 if nm == "k" else 0.0 for nm in names])
    for _ in range(30 if names else 0):
        r = res(x)
        J = np.column_stack([(res(x + np.eye(len(x))[i] * 1e-6) - r) / 1e-6 for i in range(len(x))])
        x += np.linalg.lstsq(J, -r, rcond=None)[0]
    return x, np.median(np.linalg.norm(res(x).reshape(-1, 2), axis=1))


subsets = {"CALIBRATE's 3 cubes": [0, 1, 2], "all": list(range(n)), "Fine Tune's alone": list(range(3, n)),
           "near half (reach < 200)": [i for i in range(n) if np.hypot(*lva.arm[i]) < 200],
           "far half (reach >= 200)": [i for i in range(n) if np.hypot(*lva.arm[i]) >= 200],
           "without 4, 5": [i for i in range(n) if ids[i] not in (4, 5)]}
for name, keep in subsets.items():
    camera, _, edge = lva.fit_camera(base, keep)
    cubes = all_cubes(camera)
    (k,), typ_k = radial(cubes, ["k"])
    (c,), typ_c = radial(cubes, ["c"])
    _, typ0 = radial(cubes, [])
    rot = lva.cv2.Rodrigues(camera[4:7])[0]
    print(f"{name:26s} ({len(keep):2d}) edges {edge:.2f} px, focal {camera[0]:.0f}, lens {(-rot.T @ camera[7:10])[2]:.0f} up"
          f" | arm: none {typ0:.2f} | k {k:.4f} -> {typ_k:.2f} | c {c:+.2f} -> {typ_c:.2f}")
