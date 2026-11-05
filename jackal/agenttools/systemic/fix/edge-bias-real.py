#!/usr/bin/env python3
"""The cubes' edges under the camera that matches the grid sheets (today-joint.py): every edge point found, signed
outward from the cube (+ the found edge lies outside the 40 mm cube drawn), by kind of edge, each cube placed by its
edges through that camera. A bias here is what a cube-only fit takes up by bending the camera.

    edge-bias-real.py
"""
import importlib.util
import math
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tj", HERE / "today-joint.py")
tj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tj)
from applib.camera.model import camera_position, cube_corners, visible_edges  # noqa: E402

state = pickle.load(open("/tmp/today-joint.pkl", "rb"))
KINDS = ("top outline", "upright outline", "bottom outline", "top inner", "upright inner")
for name in ("both", "saved"):
    x = state[name]
    T = tj.matrix(x[:tj.CAM])
    lens = camera_position(T)
    print(f"\n{name}: lens {np.round(lens, 0)}")
    totals = {k: [] for k in KINDS}
    for i, c in enumerate(tj.COLOURS):
        cx, cy, turn = x[tj.CUBES + 3 * i: tj.CUBES + 3 * i + 3]
        corners3 = cube_corners((cx, cy), turn)
        p = T @ np.c_[corners3, np.ones(8)].T
        corners = (p[:2] / p[2]).T
        m = T @ np.array([cx, cy, 20.0, 1.0]); middle = m[:2] / m[2]
        outline = {tuple(sorted((e.start, e.end))): e.is_outline for e in visible_edges((cx, cy), turn, lens)}
        starts, ends, found = tj.FIT["cube_edges"][c][1]
        by = {k: [] for k in KINDS}
        for a, b, q in zip(starts, ends, found):
            key = tuple(sorted((int(a), int(b))))
            if key not in outline:
                continue
            side = corners[b] - corners[a]
            normal = np.array([-side[1], side[0]]) / np.linalg.norm(side)
            if (middle - corners[a]) @ normal > 0:
                normal = -normal
            d = (q - corners[a]) @ normal
            lo, hi = sorted(key)
            kind = ("top" if lo >= 4 else "bottom" if hi < 4 else "upright") + (" outline" if outline[key] else " inner")
            if abs(d) < 5:
                by[kind].append(d)
        dist = np.linalg.norm(lens - (cx, cy, 20.0))
        print(f"   {c:5s} {dist:4.0f} mm from the lens: " + " | ".join(
            f"{k} {np.mean(v):+5.2f} ({len(v)})" if v else f"{k} -" for k, v in by.items()))
        for k in KINDS:
            totals[k] += by[k]
    print("   all: " + " | ".join(f"{k} {np.mean(v):+5.2f}" for k, v in totals.items() if v))
