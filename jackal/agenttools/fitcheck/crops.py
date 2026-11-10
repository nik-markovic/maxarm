#!/usr/bin/env python3
"""Zoomed crops of CALIBRATE's three cubes (config/camera-frame.png, 10 Nov 10:31): the edge points stored for each
(camera-refine-points.json), coloured by kind -- base edges, uprights, top edges -- and the 40 mm cube through
CALIBRATE's camera placed by those edges (white), and the blue cube with its top at 41.5 mm (magenta). Writes
out/crops.jpg.

    PYTHONPATH=src:src/calibration:agenttools/fitcheck ../.venv/bin/python3 agenttools/fitcheck/crops.py
"""
import json, math
from pathlib import Path
import cv2
import numpy as np
from applib.camera.fit import CameraFit
from applib.camera.store import RefinePoint
from camera import refine
from blue import camera, corners, misses
from applib.camera.model import least_squares, project, visible_edges, camera_position
from applib.camera.fit import EDGE_HUBER_PX

ZOOM = 4
frame = cv2.imread("config/camera-frame.png")
fit = CameraFit.from_json(json.loads(Path("config/camera.json").read_text()))
stored = [RefinePoint(**p) for p in json.loads(Path("config/camera-refine-points.json").read_text())["points"]]
sights = [p.sighting for p in stored]
alone = fit.cube_fit().true_to_pixel
KIND = {"base": (0, 200, 255), "upright": (0, 255, 0), "top": (255, 120, 0)}


def kind(a, b):
    return "base" if a < 4 and b < 4 else "top" if a >= 4 and b >= 4 else "upright"


def placed(matrix, s, side, height):
    start = np.array([*refine.on_desk(matrix, s), math.radians(s.turn_deg)])
    return least_squares(lambda t: misses(matrix, s, t[:2], t[2], side, height), start, np.array([0.01, 0.01, 1e-5]),
                         huber=EDGE_HUBER_PX, iterations=40)[0]


tall, _, _ = camera(fit.frame_size, sights[:3], [(40.0, 41.5), (40.0, 40.0), (40.0, 40.0)], alone)
tiles = []
for k in range(3):
    s = sights[k]
    starts, ends, found = s.edges
    lo, hi = found.min(axis=0) - 40, found.max(axis=0) + 40
    x0, y0 = int(lo[0]), int(lo[1])
    x1, y1 = int(hi[0]), int(hi[1])
    crop = cv2.resize(frame[y0:y1, x0:x1], None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)

    def at(p):
        return (int(round((p[0] - x0) * ZOOM)), int(round((p[1] - y0) * ZOOM)))

    for name, matrix, size, colour in (("40 mm", alone, (40.0, 40.0), (255, 255, 255)),
                                        ("top 41.5", tall, (40.0, 41.5), (255, 0, 255))):
        if name == "top 41.5" and stored[k].cube != "blue":
            continue
        cube = placed(matrix, s, *size)
        pix = project(matrix, corners(cube[:2], cube[2], *size))
        seen = visible_edges(cube[:2], cube[2], camera_position(matrix))
        for e in seen:
            cv2.line(crop, at(pix[e.start]), at(pix[e.end]), colour, 1, cv2.LINE_AA)
    for a, b, p in zip(starts, ends, found):
        cv2.circle(crop, at(p), 3, KIND[kind(min(a, b), max(a, b))], -1, cv2.LINE_AA)
    cv2.putText(crop, f"#{stored[k].id} {stored[k].cube}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    tiles.append(crop)
height = max(t.shape[0] for t in tiles)
tiles = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 10, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for t in tiles]
sheet = cv2.hconcat(tiles)
legend = np.full((70, sheet.shape[1], 3), 40, np.uint8)
x = 10
for text, colour in (("base edge points", KIND["base"]), ("upright points", KIND["upright"]),
                     ("top edge points", KIND["top"]), ("40 mm cube, CALIBRATE's camera", (255, 255, 255)),
                     ("blue with top at 41.5 mm", (255, 0, 255))):
    cv2.circle(legend, (x + 10, 35), 8, colour, -1)
    cv2.putText(legend, text, (x + 28, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
    x += 40 + 19 * len(text)
sheet = cv2.vconcat([sheet, legend])
cv2.imwrite("agenttools/fitcheck/out/crops.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(sheet.shape)
