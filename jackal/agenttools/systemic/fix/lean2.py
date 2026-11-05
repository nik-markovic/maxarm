#!/usr/bin/env python3
"""Cube edges found off by which way they face in the picture: every found point taken to sit b px outside where the
finder put it times how much its edge faces sideways (|n_x| of its outward normal) -- the left and right sides most,
the tops and bottoms not -- or likewise up and down (|n_y|). As sens.py: the 4 Nov points' cube-only fit, the focal
length, optical centre, how today's first sheets read, and the arm laid by a turn and a shift.

    lean2.py
"""
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sens", HERE / "sens.py")
sens = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sens)
rd, onedim, cube_corners, pn = sens.rd, sens.onedim, sens.cube_corners, sens.pn


def residual(camera, cube, data, how):
    axis, b = how
    starts, ends, found = data
    d = sens.residual(camera, cube, data, {})
    pix = onedim.project(camera, cube_corners(cube[:2], cube[2]))
    side = pix[ends] - pix[starts]
    normal = np.column_stack((-side[:, 1], side[:, 0])) / np.linalg.norm(side, axis=1)[:, None]
    return d + b * np.abs(normal[:, 0 if axis == "x" else 1])


rig = onedim.Rig(Path("/tmp/c1104"))
start = np.array([rig.first[0], 1.0, *rig.first[1:9], 0.0, 0.0])
cubes0 = np.array(rig.starts, float)
datas = [p.edges for p in rig.points]
rd.residual = lambda camera, cube, data, how, _base: residual(camera, cube, data, how)
for how in (("x", 0.0), ("x", -1.0), ("x", -2.0), ("x", +1.0), ("x", +2.0), ("y", -1.0), ("y", +1.0)):
    camera, cubes, edge = rd.solve(start, cubes0, datas, how, True, [0, 2, 3, 4, 5, 6, 7, 8, 9])
    _, lo = rig.arm_misses(cubes[:, :2])
    read = pn.sheets_read(camera).replace("sheets read: ", "")
    word = "sideways" if how[0] == "x" else "up/down"
    print(f"edges facing {word:8s} {how[1]:+.1f} px out: edges {edge:.3f}, f {camera[0]:6.1f} centre ({camera[2]:6.1f}, "
          f"{camera[3]:5.1f}) | {read} | arm left out {np.median(lo):.2f} / {lo.max():.2f}", flush=True)
