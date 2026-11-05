#!/usr/bin/env python3
"""Point 6's dialog picture (GET /api/tests/record.png, w=900, drawn after SAVE through SAVE's camera) with the cube
also drawn through CALIBRATE's camera, the one it was found with: magenta. The crop and zoom recomputed as
record.zoomed() made them. How far apart the two drawings are, frame px.

    nov5-record6-fix.py
"""
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from applib.camera import store
from applib.camera.model import project

CONFIG = Path("/tmp/c1105")
OUT = Path(__file__).resolve().parent / "out"
fit = store.load_camera(CONFIG)
found = json.load(urllib.request.urlopen("http://127.0.0.1:8080/api/state", timeout=10))["tests"]["found"]
corners = np.array(found["corners"])
saved = project(fit.desk_to_pixel, corners)
calibrate = project(fit.cube_fit().desk_to_pixel, corners)
apart = np.linalg.norm(saved - calibrate, axis=1)
print(f"corners drawn through SAVE's camera vs CALIBRATE's: {apart.round(1).tolist()} px")
low, high = saved.min(axis=0), saved.max(axis=0)
margin = 0.6 * float(np.max(high - low))
x0, y0 = (max(0, int(np.floor(v - margin))) for v in low)
x1 = min(2560, int(np.ceil(high[0] + margin)))
zoom = 900 / (x1 - x0)
picture = cv2.imread(str(OUT / "nov5-record6.png"))
at = (calibrate - (x0, y0) + 0.5) * zoom - 0.5
for a, b in [(edge[0], edge[1]) for edge in found["seen"]]:
    cv2.line(picture, tuple(np.round(at[a] * 16).astype(int)), tuple(np.round(at[b] * 16).astype(int)),
             (255, 0, 255), 1, cv2.LINE_AA, 4)
band = np.full((70, picture.shape[1], 3), 32, np.uint8)
for i, text in enumerate(["yellow: as the dialog drew it after SAVE, through SAVE's camera (wrong)",
                          "magenta: the same cube through CALIBRATE's camera, the one it was found with"]):
    cv2.putText(band, text, (10, 26 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
cv2.imwrite(str(OUT / "nov5-record6-fix.png"), np.vstack((picture, band)))
