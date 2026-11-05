#!/usr/bin/env python3
"""One camera, one focal length: the C920 at 2560 x 1472, focus 50, on 2 and 4 Nov. Each rig's camera fitted to its
points' cube edges alone with the focal length held at a value (the pose and the lens's middle free), then the arm
laid on the cubes by a turn and a shift: edges, the arm's misses left out, and where the lens stands.

    focal.py [config_dir ...]       # default the 1104 copy and impala's
"""
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("onedim", HERE / "one-dimension.py")
onedim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(onedim)

configs = [Path(a) for a in sys.argv[1:]] or [Path("/tmp/c1104"), Path("/tmp/cimp")]
rigs = [onedim.Rig(c) for c in configs]
for focal in (1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2200):
    cells = []
    for rig in rigs:
        camera, cubes, edge = rig.fit({"focal": float(focal)})
        rows, lo = rig.arm_misses(cubes)
        lens = -cv2.Rodrigues(camera[4:7])[0].T @ camera[7:10]
        cells.append(f"{rig.name}: edges {edge:.3f}, left out {np.median(lo):.2f} typ {lo.max():.2f} worst, "
                     f"mean out {rows[:, 0].mean():+.2f}, lens {lens[2]:.0f} up")
    print(f"focal {focal}: " + " | ".join(cells), flush=True)
