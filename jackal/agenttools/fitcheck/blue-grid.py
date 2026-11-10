#!/usr/bin/env python3
"""blue.py's fit, the blue cube's top height and its footprint both swept, per session: the edge cost of all points
at their readings, and, through CALIBRATE's own three cubes (blue, green, red) refitted with that blue cube, every
other point's miss -- the tests page's 'XY at CALIBRATE'.

    PYTHONPATH=src:src/calibration:agenttools/fitcheck ../.venv/bin/python3 agenttools/fitcheck/blue-grid.py
"""
import json
from pathlib import Path
import numpy as np
from applib.camera.fit import CameraFit
from applib.camera.store import RefinePoint
from blue import SESSIONS, camera, placed_miss

for name, (camera_file, points_file) in SESSIONS.items():
    fit = CameraFit.from_json(json.loads(Path(camera_file).read_text()))
    stored = [RefinePoint(**p) for p in json.loads(Path(points_file).read_text())["points"] if p.get("edges")]
    sights = [p.sighting for p in stored]
    start = fit.cube_fit().true_to_pixel
    calib = [k for k, p in enumerate(stored) if p.cube]
    print(f"\n=== {name}: rows the blue cube's top over the desk, columns its side; all points' edge cost | "
          f"through CALIBRATE's three, each other point's miss, mm")
    for top in (40.0, 41.0, 41.5, 42.0, 42.5):
        row = []
        for side in (39.5, 40.0, 40.5):
            sizes = [(side, top) if p.cube == "blue" else (40.0, 40.0) for p in stored]
            _, cost, _ = camera(fit.frame_size, sights, sizes, start)
            m3, _, _ = camera(fit.frame_size, [sights[k] for k in calib], [sizes[k] for k in calib], start)
            offs = [float(np.linalg.norm(placed_miss(m3, sights[k], sizes[k]))) for k in range(len(stored))
                    if k not in calib]
            row.append(f"{cost:7.0f} | " + " ".join(f"{o:4.2f}" for o in offs))
        print(f"  top {top:4.1f}: " + "   ".join(f"side {s}: {r}" for s, r in zip((39.5, 40.0, 40.5), row)))
