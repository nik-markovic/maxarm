#!/usr/bin/env python3
"""The three cubes in a snapshot as the calibrated camera fits them (a 40 mm box standing on the desk, in perspective,
as FIND RED CUBE and SAVE fit one), each with its height left free too; and a picture of what perspective predicts
round each cube: Fine Tune's grid at z 0 (magenta), the same grid at z 40 (cyan), the cube's fitted base square at z 0
and top square at z 40 (yellow). No arm.

    .venv/bin/python impala/agenttools/cube-grid/cubes.py frame.png out.jpg
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera import cubes as cube_finder, locate, store                      # noqa: E402
from camera.model import cube_corners, project                             # noqa: E402
from camera.sighting import Sighting, measure, start                        # noqa: E402

CROP = 330


def main():
    frame = cv2.imread(str(HERE / sys.argv[1]))
    fit = store.load_camera(ROOT / "config")
    matrix = fit.true_to_pixel
    desk_to_pixel = fit.desk_to_pixel if hasattr(fit, "desk_to_pixel") else \
        np.array(json.loads((ROOT / "config" / "camera.json").read_text())["desk_to_pixel"])
    blobs = cube_finder.find_blobs(frame)
    naming = {("red" if cube_finder.is_warm(blob) else f"cool {index}"): blob for index, blob in enumerate(blobs)}
    picture = frame.copy()
    crops = []

    def px(point, m):
        p = m @ (*point, 1.0)
        return p[:2] / p[2]

    def poly(points, colour, width, m):
        line = np.array([px(p, m) for p in points])
        cv2.polylines(picture, [np.round(line * 4).astype(np.int32)], False, colour, width, cv2.LINE_AA, shift=2)

    for colour, blob in naming.items():
        overlap, params, distances, found = locate._fit(frame, matrix, blob)
        sighting = Sighting(0.0, 0.0, tuple(project(matrix, np.array([[params[0], params[1], 0.0]]))[0]),
                            (frame.shape[1], frame.shape[0]), math.degrees(params[2]), found)
        free = measure(matrix, sighting, start(matrix, sighting))
        print(f"{colour:5s}: true ({params[0]:7.1f},{params[1]:7.1f}), turn {math.degrees(params[2]) % 90:5.1f} deg, "
              f"edges {np.median(np.abs(distances)):.2f} px; height left free: {free[3]:.2f} mm tall, moves "
              f"{math.dist(free[:2], params[:2]):.2f} mm")
        corners = cube_corners(params[:2], params[2])                 # 8 x 3, true mm: base then top
        centre = np.array([params[0], params[1]])
        # grids round the cube, true mm turned as the desk's x, y are (Fine Tune draws desk mm: near enough here)
        to_true = fit.desk_to_true
        desk_centre = np.linalg.solve(to_true, centre)
        for z, ink in ((0.0, (255, 0, 255)), (40.0, (255, 255, 0))):
            for k in np.arange(-60, 61, 10):
                a = np.arange(-60, 61, 5)
                for line in ([(desk_centre[0] + k, desk_centre[1] + t) for t in a],
                             [(desk_centre[0] + t, desk_centre[1] + k) for t in a]):
                    poly([(x, y, z) for x, y in line], ink, 1, desk_to_pixel)
        for ring, ink in ((corners[:4], (0, 255, 255)), (corners[4:], (0, 200, 255))):
            poly(list(ring) + [ring[0]], ink, 3, matrix)
        u, v = px((*centre, 20.0), matrix)
        crops.append((colour, int(u), int(v)))
    tiles = []
    for colour, u, v in crops:
        x0 = int(np.clip(u - CROP, 0, frame.shape[1] - 2 * CROP))
        y0 = int(np.clip(v - CROP, 0, frame.shape[0] - 2 * CROP))
        tile = picture[y0:y0 + 2 * CROP, x0:x0 + 2 * CROP].copy()
        cv2.putText(tile, colour, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        tiles.append(tile)
    sheet = np.hstack(tiles)
    legend = ("magenta: Fine Tune's grid, desk z 0, 10 mm   cyan: the same grid 40 mm up   "
              "yellow/orange: the cube as the camera fits it, base square (z 0) / top square (z 40)")
    band = np.zeros((50, sheet.shape[1], 3), np.uint8)
    cv2.putText(band, legend, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(HERE / sys.argv[2]), np.vstack((band, sheet)), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
