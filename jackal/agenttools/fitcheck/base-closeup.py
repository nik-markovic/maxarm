#!/usr/bin/env python3
"""Close-ups, 8x, of each CALIBRATE cube's lowest corner region (config/camera-frame.png, 10:31): the stored edge
points (yellow base, green uprights, blue top) and the 40 mm cube placed by them through CALIBRATE's camera (white
lines). Writes out/base-<colour>.jpg.

    PYTHONPATH=src:src/calibration:agenttools/fitcheck ../.venv/bin/python3 agenttools/fitcheck/base-closeup.py
"""
import json, math
from pathlib import Path
import cv2
import numpy as np
from applib.camera.fit import CameraFit, EDGE_HUBER_PX
from applib.camera.store import RefinePoint
from applib.camera.model import least_squares, project, visible_edges, camera_position
from camera import refine
from blue import corners, misses

ZOOM = 8
frame = cv2.imread("config/camera-frame.png")
fit = CameraFit.from_json(json.loads(Path("config/camera.json").read_text()))
stored = [RefinePoint(**p) for p in json.loads(Path("config/camera-refine-points.json").read_text())["points"]]
alone = fit.cube_fit().true_to_pixel
COL = {"base": (0, 220, 255), "upright": (0, 255, 0), "top": (255, 140, 0)}
for p in stored[:3]:
    s = p.sighting
    start = np.array([*refine.on_desk(alone, s), math.radians(s.turn_deg)])
    cube = least_squares(lambda t: misses(alone, s, t[:2], t[2], 40.0, 40.0), start, np.array([0.01, 0.01, 1e-5]),
                         huber=EDGE_HUBER_PX, iterations=40)[0]
    pix = project(alone, corners(cube[:2], cube[2], 40.0, 40.0))
    seen = visible_edges(cube[:2], cube[2], camera_position(alone))
    base = [e for e in seen if e.start < 4 and e.end < 4]
    lowest = max(range(4), key=lambda k: pix[k][1])                 # the base corner lowest in the picture
    c = pix[lowest]
    half = 45
    x0, y0 = int(c[0] - half), int(c[1] - half)
    crop = cv2.resize(frame[y0:y0 + 2 * half, x0:x0 + 2 * half], None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)

    def at(q):
        return (int(round((q[0] - x0 + 0.5) * ZOOM - 0.5)), int(round((q[1] - y0 + 0.5) * ZOOM - 0.5)))

    for e in seen:
        cv2.line(crop, at(pix[e.start]), at(pix[e.end]), (255, 255, 255), 1, cv2.LINE_AA)
    starts, ends, found = s.edges
    for a, b, q in zip(starts, ends, found):
        k = "base" if max(a, b) < 4 else "top" if min(a, b) >= 4 else "upright"
        cv2.circle(crop, at(q), 5, COL[k], -1, cv2.LINE_AA)
    cv2.putText(crop, f"#{p.id} {p.cube}: lowest base corner, 8x; 1 frame px = 8 here", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(crop, "yellow base pts, green upright pts, white: 40 mm cube via CALIBRATE", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(f"agenttools/fitcheck/out/base-{p.cube}.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(p.cube, "lowest corner", lowest, np.round(c, 1))
