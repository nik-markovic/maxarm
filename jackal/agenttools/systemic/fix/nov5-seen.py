#!/usr/bin/env python3
"""5 Nov night: where CALIBRATE's camera sees each point's cube against its reading and where the arm was sent; the
pairs at one place compared (the arm sent to the same spot twice). No arm, no app: /tmp/c1105.

    nov5-seen.py
"""
from pathlib import Path

import numpy as np

from applib.camera import store
from camera import refine

CONFIG = Path("/tmp/c1105")
fit = store.load_camera(CONFIG)
alone = fit.cube_fit()
points = store.load_refine_points(CONFIG, fit)
seen = {}
for point in points:
    off = refine.off_by(alone, point.sighting)
    seen[point.id] = off.seen
    reading = np.array([point.x, point.y])
    print(f"#{point.id} {point.cube or 'red*':6} sent {point.sent[:2]} reading {reading.tolist()} seen "
          f"{np.round(off.seen, 2).tolist()}  seen-reading {np.round(off.seen - reading, 2).tolist()} "
          f"({off.across:.2f})  reading-sent {np.round(reading - point.sent[:2], 2).tolist()}  top {off.height:+.2f}"
          f"  turn {point.turn_deg:.1f}")
by_id = {point.id: point for point in points}
for first, second in ((3, 4), (2, 5), (1, 6)):
    a, b = by_id[first], by_id[second]
    readings = np.array([b.x - a.x, b.y - a.y])
    cubes = seen[second] - seen[first]
    print(f"#{first} -> #{second}: readings moved {np.round(readings, 2).tolist()} ({np.linalg.norm(readings):.2f}), "
          f"cube as seen moved {np.round(cubes, 2).tolist()} ({np.linalg.norm(cubes):.2f}); "
          f"difference {np.round(cubes - readings, 2).tolist()} ({np.linalg.norm(cubes - readings):.2f})")
