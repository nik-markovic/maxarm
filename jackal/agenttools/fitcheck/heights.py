#!/usr/bin/env python3
"""How tall each cube looks: its height free, placed by its own edges, through one camera per session (all points
at their readings, every cube taken as 40 mm: the app's camera). The 5 Nov pairs stood at one spot, so the camera
sees both cubes of a pair the same way: a difference between them is the cubes'. Also each cube's top square's size.

    PYTHONPATH=src:src/calibration:agenttools/fitcheck ../.venv/bin/python3 agenttools/fitcheck/heights.py
"""
import json, math
from pathlib import Path
import numpy as np
from applib.camera.fit import EDGE_HUBER_PX, CameraFit
from applib.camera.model import least_squares
from applib.camera.store import RefinePoint
from camera import refine
from pretend import fit_camera
from blue import misses

SESSIONS = {
    "10 Nov": ("config/camera.json", "config/camera-refine-points.json"),
    "9-10 Nov": ("config/camera.json", "config/camera-refine-points-previous.json"),
    "5 Nov": ("../work/config-jackal-1105/camera.json", "../work/config-jackal-1105/camera-refine-points.json"),
    "4 Nov": ("../work/config-jackal-1104/camera.json", "../work/config-jackal-1104/camera-refine-points.json"),
}
for name, (camera_file, points_file) in SESSIONS.items():
    fit = CameraFit.from_json(json.loads(Path(camera_file).read_text()))
    stored = [RefinePoint(**p) for p in json.loads(Path(points_file).read_text())["points"] if p.get("edges")]
    sights = [p.sighting for p in stored]
    numbers, _, _ = fit_camera(fit.frame_size, sights, fit.cube_fit().true_to_pixel)
    m = refine._pinhole(numbers).matrix
    print(f"\n=== {name}: each cube placed by its own edges, through the all-points camera (f {numbers[0]:.0f})")
    for p, s in zip(stored, sights):
        start = np.array([*refine.on_desk(m, s), math.radians(s.turn_deg)])
        free_h = least_squares(lambda t: misses(m, s, t[:2], t[2], 40.0, t[3]), np.append(start, 40.0),
                               np.array([0.01, 0.01, 1e-5, 0.01]), huber=EDGE_HUBER_PX, iterations=40)[0]
        free_both = least_squares(lambda t: misses(m, s, t[:2], t[2], t[4], t[3]), np.append(start, [40.0, 40.0]),
                                  np.array([0.01, 0.01, 1e-5, 0.01, 0.01]), huber=EDGE_HUBER_PX, iterations=40)[0]
        print(f"  #{p.id:2} {p.cube or 'red':5} ({p.x:7.1f}, {p.y:7.1f}): looks {free_h[3]:5.2f} mm tall (40 mm square);"
              f" side and height both free: {free_both[4]:5.2f} wide, {free_both[3]:5.2f} tall")
