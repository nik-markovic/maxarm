#!/usr/bin/env python3
"""5 Nov night: do CALIBRATE and Fine Tune measure a cube the same way? CALIBRATE's frame, CALIBRATE's camera: each
cube placed (a) from the edges CALIBRATE kept for it (points 1-3), (b) by Fine Tune's finder run on the same frame
(locate.py: the blob, its outline, then the edges; the red cube's blob of warm pixels only, as FIND RED CUBE has it;
the green and blue cubes' from all hues). Both placed through the same camera, 40 mm, x, y, turn free. /tmp/c1105.

    nov5-same.py
"""
import math
from pathlib import Path

import cv2
import numpy as np

from applib.camera import cubes, locate, store
from applib.camera.fit import HANDEDNESS
from applib.camera.sighting import Sighting, in_frame, place, start

CONFIG = Path("/tmp/c1105")
frame = cv2.imread(str(CONFIG / "camera-frame.png"))
fit = store.load_camera(CONFIG)
alone = fit.cube_fit()
matrix = alone.true_to_pixel
points = {p.cube: p for p in store.load_refine_points(CONFIG, fit) if p.cube}
blobs = cubes.find_blobs(frame)
warm = [b for b in cubes.find_blobs(frame, is_warm_only=True) if cubes.is_warm(b)]
for colour, point in points.items():
    kept = place(matrix, point.sighting, start(matrix, point.sighting)[:3])
    pixel = np.array(point.pixel)
    if colour == "red":
        blob = max(warm, key=lambda b: b.area)
    else:
        blob = min(blobs, key=lambda b: np.linalg.norm(b.centroid - pixel))
    overlap, params, distances, found = locate._fit(frame, matrix, blob)
    again = Sighting(point.x, point.y, point.pixel, tuple(fit.frame_size), math.degrees(params[2]), found)
    placed = place(matrix, again, start(matrix, again)[:3])
    a, b = HANDEDNESS @ kept[:2], HANDEDNESS @ placed[:2]
    print(f"{colour:5}: CALIBRATE's edges ({len(point.sighting.edges[2])}) put it at ({a[0]:.2f}, {a[1]:.2f}); "
          f"Fine Tune's finder ({len(found[2])} edge points, {np.median(np.abs(distances)):.2f} px, blob area "
          f"{blob.area}) at ({b[0]:.2f}, {b[1]:.2f}): {np.linalg.norm(a - b):.2f} mm apart")
