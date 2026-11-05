#!/usr/bin/env python3
"""5 Nov night: each point's outline as found, through CALIBRATE's camera, the cube free (x, y, turn, height): how
many edge points, how far they lie from its drawn edges, per edge, and how tall it looks. /tmp/c1105, no arm.

    nov5-edges.py
"""
from pathlib import Path

import numpy as np

from applib.camera import store
from applib.camera.sighting import in_frame, measure, misses, start

CONFIG = Path("/tmp/c1105")
fit = store.load_camera(CONFIG)
camera = fit.cube_fit().true_to_pixel
for point in store.load_refine_points(CONFIG, fit):
    s = point.sighting
    matrix = in_frame(camera, fit.frame_size, s)
    numbers = measure(matrix, s, start(matrix, s))
    off = misses(matrix, s, numbers)
    starts, ends, _ = s.edges
    per_edge = []
    for a, b in sorted({(int(x), int(y)) for x, y in zip(starts, ends)}):
        sel = (starts == a) & (ends == b)
        per_edge.append(f"{a}{b}:{np.mean(off[sel]):+.1f}")
    print(f"#{point.id} {point.cube or 'red*':6} {len(off)} edge points, {np.median(np.abs(off)):.2f} px typ, "
          f"{np.percentile(np.abs(off), 90):.2f} px 90%; top {numbers[3] - 40:+.2f} mm | mean miss per edge "
          + " ".join(per_edge))
