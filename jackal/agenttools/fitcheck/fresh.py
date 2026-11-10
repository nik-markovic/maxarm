#!/usr/bin/env python3
"""The fresh frame (out/scattered.png, 16:00, the cubes put back near their spots): has the camera moved since
CALIBRATE's frame (10:31)? Phase correlation of desk patches away from the cubes. Then each cube found as Fine Tune
finds the red one (locate.py's fit through CALIBRATE's camera), and measured with its height, then its size, free.

    PYTHONPATH=src:src/calibration:agenttools/fitcheck ../.venv/bin/python3 agenttools/fitcheck/fresh.py
"""
import json, math
from pathlib import Path
import cv2
import numpy as np
from applib.camera import cubes, locate
from applib.camera.fit import CameraFit, EDGE_HUBER_PX, HANDEDNESS
from applib.camera.model import least_squares
from applib.camera.sighting import Sighting
from blue import misses

old = cv2.imread("config/camera-frame.png")
new = cv2.imread("agenttools/fitcheck/out/scattered.png")
fit = CameraFit.from_json(json.loads(Path("config/camera.json").read_text()))
alone = fit.cube_fit()
matrix = alone.true_to_pixel

# Patches of bare desk in both frames: away from every cube blob of either frame.
masks = np.zeros(old.shape[:2], np.uint8)
for frame in (old, new):
    for blob in cubes.find_blobs(frame):
        masks |= cv2.dilate(blob.mask, np.ones((151, 151), np.uint8))
g_old = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY).astype(np.float32)
g_new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY).astype(np.float32)
shifts = []
for y in range(200, 1300, 220):
    for x in range(200, 2300, 220):
        if masks[y - 96:y + 96, x - 96:x + 96].any():
            continue
        a = g_old[y - 96:y + 96, x - 96:x + 96]
        b = g_new[y - 96:y + 96, x - 96:x + 96]
        a = cv2.GaussianBlur(a, (0, 0), 1.5) - cv2.GaussianBlur(a, (0, 0), 12)
        b = cv2.GaussianBlur(b, (0, 0), 1.5) - cv2.GaussianBlur(b, (0, 0), 12)
        (dx, dy), response = cv2.phaseCorrelate(a, b, cv2.createHanningWindow((192, 192), cv2.CV_32F))
        if response > 0.2:
            shifts.append((x, y, dx, dy, response))
shifts = np.array(shifts)
print(f"camera motion, 10:31 -> 16:00, from {len(shifts)} bare-desk patches: median shift "
      f"({np.median(shifts[:, 2]):+.2f}, {np.median(shifts[:, 3]):+.2f}) px, spread "
      f"{np.percentile(np.abs(shifts[:, 2] - np.median(shifts[:, 2])), 90):.2f} / "
      f"{np.percentile(np.abs(shifts[:, 3] - np.median(shifts[:, 3])), 90):.2f} px (90%)")
for x, y, dx, dy, r in shifts:
    print(f"   at ({x:4.0f}, {y:4.0f}): ({dx:+5.2f}, {dy:+5.2f}) px, response {r:.2f}")

# A shift of the whole picture, taken out of the camera, for the cubes' measurement.
shift = np.array([[1, 0, np.median(shifts[:, 2])], [0, 1, np.median(shifts[:, 3])], [0, 0, 1.0]])
moved = shift @ matrix
for blob in cubes.find_blobs(new):
    try:
        overlap, params, distances, found = locate._fit(new, moved, blob)
    except Exception as reason:
        print(f"blob at {np.round(blob.centroid)}: {reason}")
        continue
    if overlap < 0.6:
        continue
    sight = Sighting(0.0, 0.0, (0.0, 0.0), tuple(fit.frame_size), math.degrees(params[2]), found)
    tall = least_squares(lambda t: misses(moved, sight, t[:2], t[2], 40.0, t[3]), np.append(params, 40.0),
                         np.array([0.01, 0.01, 1e-5, 0.01]), huber=EDGE_HUBER_PX, iterations=40)[0]
    both = least_squares(lambda t: misses(moved, sight, t[:2], t[2], t[4], t[3]), np.append(params, [40.0, 40.0]),
                         np.array([0.01, 0.01, 1e-5, 0.01, 0.01]), huber=EDGE_HUBER_PX, iterations=40)[0]
    desk = HANDEDNESS @ params[:2]
    print(f"cube hue {blob.hue:6.1f} at desk ({desk[0]:7.1f}, {desk[1]:7.1f}): overlap {overlap:.2f}, edges "
          f"{np.median(np.abs(distances)):.2f} px; looks {tall[3]:5.2f} mm tall (40 mm square); size free "
          f"{both[4]:5.2f} wide, {both[3]:5.2f} tall")
