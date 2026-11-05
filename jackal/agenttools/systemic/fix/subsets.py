#!/usr/bin/env python3
"""Which points disagree: the camera fitted to the edges of a subset of the points (the arm no say), the arm laid
on that subset by a turn and a shift, and every point's miss through it (out / round, mm), the subset's own left
out. Points named by id.

    subsets.py config_dir id,id,... [id,id,...]
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("onedim", HERE / "one-dimension.py")
onedim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(onedim)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tie import link_arm  # noqa: E402

rig = onedim.Rig(Path(sys.argv[1]))
for spec_ids in sys.argv[2:]:
    keep_ids = [int(i) for i in spec_ids.split(",")]
    keep = [rig.ids.index(i) for i in keep_ids]
    sub = onedim.Rig.__new__(onedim.Rig)
    sub.__dict__.update(rig.__dict__)
    sub.points = [rig.points[i] for i in keep]
    sub.arm, sub.starts = rig.arm[keep], [rig.starts[i] for i in keep]
    camera, _, edge = sub.fit({})
    # every point's cube through that camera
    cubes = []
    for p, s0 in zip(rig.points, rig.starts):
        x = onedim.solve(lambda c: onedim.edges.signed_distances(
            onedim.project(camera, onedim.cube_corners(c[:2], c[2])), *p.edges), np.array(s0, float), [0, 1, 2])
        cubes.append(x[:2])
    cubes = np.array(cubes)
    turn, shift, mirror = link_arm(rig.arm[keep], cubes[keep])
    _, lo = sub.arm_misses(cubes[keep])
    print(f"\nsubset {keep_ids}: edges {edge:.2f} px; its own left out {np.median(lo):.2f} typ, {lo.max():.2f} worst")
    for i, (pid, a, c) in enumerate(zip(rig.ids, rig.arm, cubes)):
        d = np.linalg.solve(turn @ mirror, c - shift) - a
        b = math.atan2(a[0], -a[1])
        tag = "in " if i in keep else "OUT"
        print(f"   {tag} {pid:3d} bear {math.degrees(b):+6.1f} reach {np.hypot(*a):5.0f}: out {d @ (math.sin(b), -math.cos(b)):+5.2f}"
              f" round {d @ (math.cos(b), math.sin(b)):+5.2f}  |{np.linalg.norm(d):4.1f}|")
