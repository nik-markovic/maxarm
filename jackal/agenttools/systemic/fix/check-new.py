#!/usr/bin/env python3
"""The new CALIBRATE and SAVE (the cubes where the arm read them) on the saved 4 Nov data: no arm, no app. Each
camera read against the grid sheets of 4 Nov afternoon (the camera's 4 px move since taken out, pinned.py), and SAVE's
points left out as the page shows them.

    check-new.py
"""
import importlib.util
import json
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pinned", HERE / "pinned.py")
pn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pn)
from applib.camera import store  # noqa: E402
from applib.camera.fit import fit_camera  # noqa: E402
from camera import refine  # noqa: E402

CONFIG = Path("/tmp/c1104")


def as12(true_to_pixel):
    n = refine._numbers(true_to_pixel)
    return np.array([n[0], 1.0, *n[1:9], 0.0, 0.0])


fit = store.load_camera(CONFIG)
points = [p.sighting for p in store.load_refine_points(CONFIG, fit) if p.edges]
started = time.time()
refined = refine.refine(fit, points)
took = time.time() - started
cam = as12(refined.true_to_pixel)
print(f"SAVE, 12 points on the arm ({took:.1f} s): {pn.describe(cam)}\n   {pn.sheets_read(cam)}\n   refined {refined.refined}"
      f", stretch {np.round(refined.measured_stretch, 4).tolist()}")
started = time.time()
put = refine.left_out(refined, points)
offs = [float(np.linalg.norm(o - refined.desk_to_true @ refine.off_by(refined, p).seen)) for o, p in zip(put, points)]
print(f"   left out ({time.time() - started:.1f} s): {np.median(offs):.2f} typ, {max(offs):.2f} worst")
frame = cv2.imread(str(CONFIG / "camera-frame.png"))
recorded = {colour: (p["x"], p["y"]) for colour, p in json.loads((CONFIG / "camera-points.json").read_text())["cubes"].items()}
started = time.time()
calibrated = fit_camera(frame, recorded)
cam3 = as12(calibrated.true_to_pixel)
print(f"CALIBRATE on the arm ({time.time() - started:.1f} s): {pn.describe(cam3)}\n   {pn.sheets_read(cam3)}\n   edges "
      f"{calibrated.edge_px}, cubes off {[round(calibrated.cube_off_mm(c), 2) for c in calibrated.cubes]}, stretch "
      f"{np.round(calibrated.measured_stretch, 4).tolist()}")
others = [refine.off_by(calibrated, p).across for p in points[3:]]
print(f"   the other 9 points against the arm through it: {np.median(others):.2f} typ, {max(others):.2f} worst")
