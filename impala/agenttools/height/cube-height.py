#!/usr/bin/env python3
"""How tall each cube of a CALIBRATE frame looks through a camera: its base held on the desk, its height free.

    .venv/bin/python impala/agenttools/height/cube-height.py [config-dir] [suffix]

Reads camera{suffix}.json, camera-frame{suffix}.png, camera-points{suffix}.json (suffix "-previous" by default).
For the three cubes' own camera and the refined one: each cube fitted (x, y, turn) at 40 mm, then (x, y, turn,
height); how far its base is from the arm's point, and how tall it looks.
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera import cubes, edges                                   # noqa: E402
from camera.fit import EDGE_HUBER_PX, REACH, CameraFit            # noqa: E402
from camera.locate import _fit, view_camera                      # noqa: E402
from camera.model import (CUBE_MM, _SQUARE, camera_position, least_squares, project,   # noqa: E402
                          visible_edges)


def box(centre, turn, height):
    c, s = math.cos(turn), math.sin(turn)
    return np.array([(centre[0] + CUBE_MM * (c * dx - s * dy), centre[1] + CUBE_MM * (s * dx + c * dy), z)
                     for z in (0.0, height) for dx, dy in _SQUARE])


def fit_height(frame, matrix, blob, start):
    crop = edges.crop(frame, blob, cubes.desk_chroma(frame))
    camera = camera_position(matrix)
    params = np.array([*start, CUBE_MM])
    for reach in REACH:
        seen = visible_edges(params[:2], params[2], camera)
        corners = project(matrix, box(params[:2], params[2], params[3]))
        starts, ends, targets = edges.search(crop, corners, seen, reach)
        params, distances, normal = least_squares(
            lambda q: edges.signed_distances(project(matrix, box(q[:2], q[2], q[3])), starts, ends, targets),
            params, np.array([0.01, 0.01, 1e-5, 0.01]), huber=EDGE_HUBER_PX, iterations=30)
    spread = np.sqrt(np.diag(np.linalg.inv(normal)) * np.mean(np.minimum(distances ** 2, 1.0)))
    return params, distances, spread, (starts, ends, targets)


def main():
    config = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "config"
    suffix = sys.argv[2] if len(sys.argv) > 2 else "-previous"
    frame = cv2.imread(str(config / f"camera-frame{suffix}.png"))
    saved = CameraFit.from_json(json.loads((config / f"camera{suffix}.json").read_text()))
    arm = json.loads((config / f"camera-points{suffix}.json").read_text())["cubes"]
    blobs = cubes.find_blobs(frame)
    for name, fit in (("cubes alone", saved.cube_fit()), ("refined", saved)):
        matrix = view_camera(fit, frame)
        print(f"--- {name}: camera at {np.round(fit.camera_position, 1)}, focal {fit.focal:.1f}")
        for colour, cube in saved.cube_fit().cubes.items():
            pixel = np.array(cube["pixel"])
            blob = min(blobs, key=lambda b: np.linalg.norm(b.centroid - pixel))
            overlap, params, distances, _ = _fit(frame, matrix, blob)
            q, d, spread, _ = fit_height(frame, matrix, blob, params)
            to_desk = np.linalg.inv(fit.desk_to_true)
            at40 = to_desk @ params[:2]
            atq = to_desk @ q[:2]
            ax, ay = arm[colour]["x"], arm[colour]["y"]
            print(f"{colour:5s} 40 mm: off {math.dist(at40, (ax, ay)):.2f} mm, edges {np.median(np.abs(distances)):.2f}"
                  f" px | free height {q[3]:.2f} mm (+-{spread[3]:.2f}), base off {math.dist(atq, (ax, ay)):.2f}"
                  f" (moved {math.dist(atq, at40):.2f}), edges {np.median(np.abs(d)):.2f} px, {len(d)} points")


if __name__ == "__main__":
    main()
