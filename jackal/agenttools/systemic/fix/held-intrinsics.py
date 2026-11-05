#!/usr/bin/env python3
"""The camera's own numbers held, the rest from the cubes: each rig's camera fitted to its points' cube edges alone
(pose free, each cube placed freely) with the focal length and the lens's middle held at what the grid sheets gave
(today-joint.py), then the arm laid on the cubes by a turn and a shift as a check. Also CALIBRATE's three cubes alone.

    held-intrinsics.py
"""
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("onedim", HERE / "one-dimension.py")
onedim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(onedim)

both = pickle.load(open("/tmp/today-joint.pkl", "rb"))["both"]
real = {"focal": float(both[0]), "cx": float(both[1]), "cy": float(both[2])}
print(f"held at the sheets' camera: focal {real['focal']:.1f}, middle ({real['cx']:.1f}, {real['cy']:.1f})")
for config in (Path("/tmp/c1104"), Path("/tmp/cimp")):
    rig = onedim.Rig(config)
    for name, held, keep in (("free (the app's)", {}, None), ("held", real, None),
                             ("held, CALIBRATE's 3 cubes alone", real, [0, 1, 2]) if config.name == "c1104" else
                             ("held, 3 cubes alone", real, [0, 1, 2])):
        sub = rig
        if keep is not None:
            sub = onedim.Rig.__new__(onedim.Rig); sub.__dict__.update(rig.__dict__)
            sub.points = [rig.points[i] for i in keep]; sub.starts = [rig.starts[i] for i in keep]
            sub.arm = rig.arm[keep]
        camera, _, edge = sub.fit(held)
        cubes = []                                    # every point's cube through this camera
        for p, s0 in zip(rig.points, rig.starts):
            x = onedim.solve(lambda c: onedim.edges.signed_distances(
                onedim.project(camera, onedim.cube_corners(c[:2], c[2])), *p.edges), np.array(s0, float), [0, 1, 2])
            cubes.append(x[:2])
        rows, lo = rig.arm_misses(np.array(cubes))
        print(f"{config.name:6s} {name:34s}: edges {edge:.3f} px, f {camera[0]:.0f}, middle ({camera[2]:.0f}, "
              f"{camera[3]:.0f}); the arm, left out: {np.median(lo):.2f} typ, {lo.max():.2f} worst; mean out "
              f"{rows[:, 0].mean():+.2f}", flush=True)
