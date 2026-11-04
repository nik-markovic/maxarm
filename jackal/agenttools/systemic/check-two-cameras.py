#!/usr/bin/env python3
"""CALIBRATE's frame with each cube drawn twice: through CALIBRATE's camera (green) and as SAVE's check image draws
it (pink). Writes check-two-cameras.jpg beside this script.

    ../.venv/bin/python agenttools/systemic/check-two-cameras.py [config_dir]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "calibration")]

from applib.camera import store                                                     # noqa: E402
from applib.camera.model import cube_corners, project                               # noqa: E402
from applib.camera.sighting import place, start                                     # noqa: E402

EDGES = [(k, (k + 1) % 4) for k in range(4)] + [(4 + k, 4 + (k + 1) % 4) for k in range(4)] + [(k, k + 4) for k in range(4)]

config = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "config"
fit = store.load_camera(config)
alone = fit.cube_fit()
image = store.load_frame(config).copy()
points = {p.cube: p.sighting for p in store.load_refine_points(config, fit) if p.cube}


def draw(corners, colour):
    for start_index, end_index in EDGES:
        cv2.line(image, tuple(np.round(corners[start_index]).astype(int)), tuple(np.round(corners[end_index]).astype(int)),
                 colour, 2, cv2.LINE_AA)


for colour, point in points.items():
    cube = place(alone.true_to_pixel, point, start(alone.true_to_pixel, point)[:3])
    draw(project(alone.true_to_pixel, cube_corners(cube[:2], cube[2])), (60, 255, 60))
    draw(fit.wireframe(colour).corners, (255, 60, 255))
cv2.rectangle(image, (0, 0), (image.shape[1], 70), (0, 0, 0), -1)
cv2.putText(image, "green: CALIBRATE's camera (3 cubes)   pink: SAVE's camera, as config/camera-check.jpg draws it",
            (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
cv2.imwrite(str(Path(__file__).with_name("check-two-cameras.jpg")), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
