#!/usr/bin/env python3
"""Zoomed crops of the three cubes in CALIBRATE's frame as it is now (config/), with the edge points it found
(camera-refine-points.json, points 1-3) coloured by kind and the 40 mm cube through camera.json (white): what the
edge search took for each edge. Writes out/now-<colour>.jpg.

    PYTHONPATH=src:src/calibration:agenttools/fitcheck ../.venv/bin/python3 agenttools/fitcheck/crops-now.py
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

ZOOM = 3
frame = cv2.imread("config/camera-frame.png")
fit = CameraFit.from_json(json.loads(Path("config/camera.json").read_text()))
stored = [RefinePoint(**p) for p in json.loads(Path("config/camera-refine-points.json").read_text())["points"]]
m = fit.true_to_pixel
COL = {"base": (0, 220, 255), "upright": (0, 255, 0), "top": (255, 140, 0)}
for p in stored[:3]:
    s = p.sighting
    start = np.array([*refine.on_desk(m, s), math.radians(s.turn_deg)])
    cube = least_squares(lambda t: misses(m, s, t[:2], t[2], 40.0, 40.0), start, np.array([0.01, 0.01, 1e-5]),
                         huber=EDGE_HUBER_PX, iterations=40)[0]
    pix = project(m, corners(cube[:2], cube[2], 40.0, 40.0))
    seen = visible_edges(cube[:2], cube[2], camera_position(m))
    starts, ends, found = s.edges
    lo, hi = np.minimum(found.min(axis=0), pix.min(axis=0)) - 60, np.maximum(found.max(axis=0), pix.max(axis=0)) + 60
    x0, y0, x1, y1 = int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])
    crop = cv2.resize(frame[y0:y1, x0:x1], None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)

    def at(q):
        return (int(round((q[0] - x0 + 0.5) * ZOOM - 0.5)), int(round((q[1] - y0 + 0.5) * ZOOM - 0.5)))

    for e in seen:
        cv2.line(crop, at(pix[e.start]), at(pix[e.end]), (255, 255, 255), 2, cv2.LINE_AA)
    for a, b, q in zip(starts, ends, found):
        k = "base" if max(a, b) < 4 else "top" if min(a, b) >= 4 else "upright"
        cv2.circle(crop, at(q), 5, COL[k], -1, cv2.LINE_AA)
    d = misses(m, s, cube[:2], cube[2], 40.0, 40.0)
    cv2.putText(crop, f"#{p.id} {p.cube}: {len(found)} edge points, {np.median(np.abs(d)):.1f} px typ, "
                f"{np.percentile(np.abs(d), 90):.1f} px 90%", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    cv2.putText(crop, "yellow base, green uprights, blue top; white: the 40 mm cube via camera.json", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(f"agenttools/fitcheck/out/now-{p.cube}.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(p.cube, crop.shape, f"median {np.median(np.abs(d)):.2f}, 90% {np.percentile(np.abs(d), 90):.2f} px")
