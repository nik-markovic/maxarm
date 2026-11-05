#!/usr/bin/env python3
"""5 Nov night: how big each point's cube looks through CALIBRATE's camera: a cube of free size (x, y, turn and one
scale for all its sides) fitted to its found edges; and, the cube 40 mm, how far its outline sits outside the dots.
/tmp/c1105, no arm.

    nov5-size.py
"""
from pathlib import Path

import numpy as np

from applib.camera import edges, store
from applib.camera.fit import EDGE_HUBER_PX
from applib.camera.model import cube_corners, least_squares, project
from applib.camera.sighting import in_frame, place, start

CONFIG = Path("/tmp/c1105")
fit = store.load_camera(CONFIG)
cameras = {"CALIBRATE": fit.cube_fit().true_to_pixel, "SAVE": fit.true_to_pixel}


def corners(matrix, n):
    c = cube_corners(n[:2], n[2])
    middle = np.array([n[0], n[1], 0.0])
    return project(matrix, middle + (c - middle) * n[3])


for name, camera in cameras.items():
    print(name)
    for point in store.load_refine_points(CONFIG, fit):
        s = point.sighting
        matrix = in_frame(camera, fit.frame_size, s)
        starts, ends, found = s.edges
        first = place(matrix, s, start(matrix, s)[:3])
        sized = least_squares(lambda n: edges.signed_distances(corners(matrix, n), starts, ends, found),
                              np.append(first, 1.0), np.array([0.01, 0.01, 1e-5, 1e-5]), huber=EDGE_HUBER_PX,
                              iterations=40)[0]
        print(f"  #{point.id} {point.cube or 'red*':6} looks {40 * sized[3]:.2f} mm")
