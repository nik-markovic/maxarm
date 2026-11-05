#!/usr/bin/env python3
"""5 Nov night: the owner's 6 points through the code as it was before 4 Nov's change (git HEAD, /tmp/oldsrc):
CALIBRATE's camera from the cubes' shapes alone, the arm laid on it by a turn and a shift; SAVE the same with every
point. Run with PYTHONPATH=/tmp/oldsrc/src:/tmp/oldsrc/src/calibration. Same table as nov5.py.

    nov5-old.py
"""
import json
from pathlib import Path

import cv2

from applib.camera import store
from applib.camera.fit import fit_camera
from tests import refine as table

CONFIG = Path("/tmp/c1105")
frame = cv2.imread(str(CONFIG / "camera-frame.png"))
recorded = {c: (p["x"], p["y"]) for c, p in json.loads((CONFIG / "camera-points.json").read_text())["cubes"].items()}
fit = fit_camera(frame, recorded)
points = store.load_refine_points(CONFIG, store.load_camera(CONFIG))
for count in range(3, len(points) + 1):
    shown = table.describe(fit, points[:count])
    print(f"\n{count} points: problem {shown['problem']}")
    if shown["fit"]:
        print("  fit", {k: shown["fit"][k] for k in ("off_mm", "height_mm", "change_mm")})
    for row in shown["points"]:
        print(f"  #{row['id']} {row['cube'] or 'red*':6} XY calib {row['off']['calibrate']:5.2f} after "
              f"{row['off']['after']}  Z calib {row['height']['calibrate']} after {row['height']['after']}")
