#!/usr/bin/env python3
"""How much the camera fit amplifies a small error: one point's reading moved 1 mm, the camera refitted (SAVE's
objective: every cube at its reading, edges), and how far that moves where the camera sends the arm elsewhere.
No arm, no app.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/amplify.py [config-dir] [count]
"""
import dataclasses
import sys
from pathlib import Path

import numpy as np

from applib.camera import store
from applib.camera.fit import EDGE_HUBER_PX, HANDEDNESS
from applib.camera.model import least_squares
from camera import refine

CONFIG = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/c1110")
COUNT = int(sys.argv[2]) if len(sys.argv) > 2 else 3

fit = store.load_camera(CONFIG)
points = [p.sighting for p in store.load_refine_points(CONFIG, fit)]
alone = fit.cube_fit()
start = alone.true_to_pixel


def camera(sightings, first=start, held=None):
    on_arm = refine._OnArm(fit.frame_size, sightings, first)
    numbers = refine._numbers(first)
    free = None
    if held is not None:
        free = [k for k in range(9) if k not in held]
    n = least_squares(on_arm.residuals, numbers, refine.CAMERA_STEPS, huber=EDGE_HUBER_PX, iterations=200,
                      derivatives=on_arm.derivatives, free=free)[0]
    return refine._pinhole(n).matrix


# Desk points to judge by: the work area, every 20 mm, reach 110-270, bearings -60..60; at the desk and cube tops.
grid = []
for reach in range(110, 271, 20):
    for bearing in range(-60, 61, 10):
        b = np.radians(bearing)
        grid.append((reach * np.sin(b), -reach * np.cos(b)))
grid = np.array(grid)


def desk_xy(matrix, pixels, height):
    """Desk x, y of pixels on the plane `height` up, through a true 3x4."""
    plane = np.column_stack((matrix[:, 0], matrix[:, 1], matrix[:, 3] + height * matrix[:, 2]))
    out = []
    for p in pixels:
        x, y, w = np.linalg.solve(plane, (*p, 1.0))
        out.append(HANDEDNESS @ (x / w, y / w))
    return np.array(out)


def pixels_of(matrix, desk, height):
    true = (HANDEDNESS @ desk.T).T
    homog = np.column_stack((true, np.full(len(true), height), np.ones(len(true)))) @ matrix.T
    return homog[:, :2] / homog[:, 2:]


base = camera(points[:COUNT])
print(f"{COUNT} points; camera: f {refine._numbers(base)[0]:.0f}, centre {np.round(refine._numbers(base)[1:3], 0)}")
pix0, pix40 = pixels_of(base, grid, 0.0), pixels_of(base, grid, 40.0)
for index in range(COUNT):
    for move in ((1, 0), (0, 1)):
        moved = list(points[:COUNT])
        p = moved[index]
        moved[index] = dataclasses.replace(p, x=p.x + move[0], y=p.y + move[1])
        for label, held in (("free", None), ("centre held", (1, 2))):
            other = camera(moved, held=held) if held is None else camera(moved, first=camera(points[:COUNT], held=held), held=held)
            ref = base if held is None else camera(points[:COUNT], held=held)
            # Where the arm would be sent for the same pixels: grid pixels through the reference camera
            r0 = pixels_of(ref, grid, 0.0)
            r40 = pixels_of(ref, grid, 40.0)
            d0 = np.linalg.norm(desk_xy(other, r0, 0.0) - grid, axis=1)
            d40 = np.linalg.norm(desk_xy(other, r40, 40.0) - grid, axis=1)
            print(f"  #{index + 1} moved {move} mm, {label:11}: work area moves at the desk median {np.median(d0):.2f}, "
                  f"worst {d0.max():.2f}; at 40 mm median {np.median(d40):.2f}, worst {d40.max():.2f}")
