#!/usr/bin/env python3
"""No arm at all: the 3 Nov three cubes (12:18 frame) fitted alone, with the camera's focal length and lens middle
free (the app's way) and held at what today's grid sheets gave; each camera then read against the 3 Nov grid paper.

    held-3nov.py
"""
import importlib.util
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("jm", HERE / "joint-model.py")
jm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jm)

both = pickle.load(open("/tmp/today-joint.pkl", "rb"))["both"]
T0 = jm.FIT["true_to_pixel"]
first = jm.refine._numbers(T0)
cubes = []
for c in jm.COLOURS:
    w = np.linalg.solve(T0[:, [0, 1, 3]], np.array([*jm.FIT["cubes"][c]["pixel"], 1.0]))
    cubes += [w[0] / w[2], w[1] / w[2], np.radians(jm.FIT["cube_edges"][c][0])]
x0 = np.concatenate([first, [1.0, 0.0, 0.0, 0.0], cubes, [0.0, 0.0, 0.0]])
cube_ids = list(range(jm.CAM, jm.PAPER))
paper_ids = [jm.PAPER, jm.PAPER + 1, jm.PAPER + 2]
for name, held in (("free (the app's)", False), ("held at today's sheets", True)):
    x = x0.copy()
    free = list(range(9)) + cube_ids
    if held:
        x[0], x[1], x[2] = both[0], both[1], both[2]
        free = list(range(3, 9)) + cube_ids
    x = jm.solve(lambda y: np.concatenate([jm.edge_res(y), jm.held(y)]), x, free)
    best = None
    for flip in (1.0, -1.0):
        for theta in np.radians(np.arange(0, 360, 10)):
            y = x.copy(); y[jm.PAPER:jm.PAPER + 3] = (theta, 0, 0)
            y = jm.solve(lambda z: jm.paper_res(z, flip), y, paper_ids, 25)
            r = np.median(np.abs(jm.paper_res(y, flip)))
            if best is None or r < best[0]:
                best = (r, y, flip)
    r, y, flip = best
    p = jm.paper_res(y, flip)
    # how its desk mm read the paper's along the paper's two axes
    reads = []
    for axis in (True, False):
        sel = jm.IS_X == axis
        P = jm.on_desk(y[:jm.CAM], jm.PIX[sel]) - y[jm.PAPER + 1: jm.PAPER + 3]
        c, s = np.cos(y[jm.PAPER]), np.sin(y[jm.PAPER])
        d = np.array([c, s]) if axis else flip * np.array([-s, c])
        reads.append(abs(np.polyfit(jm.MM[sel], P @ d, 1)[0]))
    print(f"3 Nov cubes, {name:24s}: edges {np.median(np.abs(jm.edge_res(y))):.2f} px; f {y[0]:.0f}, middle ({y[1]:.0f}, "
          f"{y[2]:.0f}); the paper through it {np.median(np.abs(p)):.3f} mm typ, {np.percentile(np.abs(p), 95):.2f} at "
          f"95%; reads it x{reads[0]:.4f} and x{reads[1]:.4f} along its two axes")
