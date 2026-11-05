#!/usr/bin/env python3
"""How much the cubes' outlines steer the camera: the 4 Nov points' cube-only fit (each cube placed freely, as SAVE
does) again with one kind of edge taken to sit half a pixel outside where the finder put it (or inside), every other
edge as found. For each: the focal length, the optical centre, where the lens stands, how today's first grid sheets
read through the camera (the 4 px move taken out), and the arm laid on the cubes by a turn and a shift.

    sens.py
"""
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rounded", HERE / "rounded.py")
rd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rd)
pn, onedim, edges, cube_corners = rd.pn, rd.onedim, rd.edges, rd.cube_corners
from applib.camera.model import visible_edges  # noqa: E402

KINDS = ("top outline", "upright outline", "bottom outline", "top inner", "upright inner")


def residual(camera, cube, data, offsets):
    """Found points against the 40 mm cube's edges through `camera`, outward +, less each kind's offset."""
    starts, ends, found = data
    c3 = cube_corners(cube[:2], cube[2])
    pix = onedim.project(camera, c3)
    middle = onedim.project(camera, np.array([[cube[0], cube[1], 20.0]]))[0]
    outline = {tuple(sorted((e.start, e.end))): e.is_outline for e in visible_edges(cube[:2], cube[2], rd.lens_of(camera))}
    side = pix[ends] - pix[starts]
    normal = np.column_stack((-side[:, 1], side[:, 0])) / np.linalg.norm(side, axis=1)[:, None]
    flip = np.sum((middle - pix[starts]) * normal, axis=1) > 0
    normal[flip] *= -1
    d = np.sum((found - pix[starts]) * normal, axis=1)
    shift = np.zeros(len(d))
    for i, (a, b) in enumerate(zip(starts.tolist(), ends.tolist())):
        key = tuple(sorted((a, b)))
        if key in outline:
            lo, hi = key
            kind = ("top" if lo >= 4 else "bottom" if hi < 4 else "upright") + (" outline" if outline[key] else " inner")
            shift[i] = offsets.get(kind, 0.0)
    return d + shift


def main():
    rig = onedim.Rig(Path("/tmp/c1104"))
    start = np.array([rig.first[0], 1.0, *rig.first[1:9], 0.0, 0.0])
    cubes0 = np.array(rig.starts, float)
    datas = [p.edges for p in rig.points]
    free = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    rd.residual = lambda camera, cube, data, offsets, _base: residual(camera, cube, data, offsets)
    trials = [("as found (the app's)", {})]
    for kind in KINDS:
        trials += [(f"{kind} +0.5 px out", {kind: 0.5}), (f"{kind} -0.5 px in", {kind: -0.5})]
    trials += [("tops 0.5 out, bottoms 0.5 in", {"top outline": 0.5, "top inner": 0.5, "bottom outline": -0.5}),
               ("tops 0.5 in, bottoms 0.5 out", {"top outline": -0.5, "top inner": -0.5, "bottom outline": 0.5})]
    for name, offsets in trials:
        camera, cubes, edge = rd.solve(start, cubes0, datas, offsets, True, free)
        _, lo = rig.arm_misses(cubes[:, :2])
        read = pn.sheets_read(camera).replace("sheets read: ", "")
        print(f"{name:30s} f {camera[0]:6.1f} centre ({camera[2]:6.1f}, {camera[3]:5.1f}) | {read} | arm left out "
              f"{np.median(lo):.2f} / {lo.max():.2f}", flush=True)


if __name__ == "__main__":
    main()
