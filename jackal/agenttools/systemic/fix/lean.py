#!/usr/bin/env python3
"""The uprights leaning: the 4 Nov points' cube-only fit again with every upright edge taken to sit L px outside where
the finder put it at the cube's foot and L px inside at its top, in a straight line between (a contact shadow at the
foot, rounding or light at the top would do so); the other edges as found. As sens.py: focal length, optical centre,
how today's first sheets read, the arm laid by a turn and a shift.

    lean.py
"""
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sens", HERE / "sens.py")
sens = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sens)
rd, onedim, cube_corners, pn = sens.rd, sens.onedim, sens.cube_corners, sens.pn


def residual(camera, cube, data, lean):
    starts, ends, found = data
    d = sens.residual(camera, cube, data, {})
    pix = onedim.project(camera, cube_corners(cube[:2], cube[2]))
    up = (np.minimum(starts, ends) < 4) & (np.maximum(starts, ends) >= 4)
    foot = np.where(starts < 4, starts, ends)
    top = np.where(starts < 4, ends, starts)
    a, b = pix[foot], pix[top]
    t = np.clip(np.sum((found - a) * (b - a), axis=1) / np.sum((b - a) ** 2, axis=1), 0, 1)
    return d + np.where(up, lean * (1 - 2 * t), 0.0)


rig = onedim.Rig(Path("/tmp/c1104"))
start = np.array([rig.first[0], 1.0, *rig.first[1:9], 0.0, 0.0])
cubes0 = np.array(rig.starts, float)
datas = [p.edges for p in rig.points]
rd.residual = lambda camera, cube, data, lean, _base: residual(camera, cube, data, lean)
for lean in (0.0, 0.5, 1.0, -0.5, -1.0):
    camera, cubes, edge = rd.solve(start, cubes0, datas, lean, True, [0, 2, 3, 4, 5, 6, 7, 8, 9])
    _, lo = rig.arm_misses(cubes[:, :2])
    read = pn.sheets_read(camera).replace("sheets read: ", "")
    print(f"uprights {lean:+.1f} px out at the foot, {-lean:+.1f} at the top: edges {edge:.3f}, f {camera[0]:6.1f} "
          f"centre ({camera[2]:6.1f}, {camera[3]:5.1f}) | {read} | arm left out {np.median(lo):.2f} / {lo.max():.2f}",
          flush=True)
