#!/usr/bin/env python3
"""10 Nov: today's CALIBRATE (3 cubes) and its 4th point, as the tests page shows them; and the same through
fully converged fits. No arm, no app: a frozen copy of config/ in /tmp/c1110.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/today.py [config-dir] [refine-file]
"""
import sys
from pathlib import Path

import numpy as np

from applib.camera import store
from applib.camera.model import least_squares
from applib.camera.fit import EDGE_HUBER_PX, HANDEDNESS
from camera import refine
from tests import refine as table

CONFIG = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/c1110")

fit = store.load_camera(CONFIG)
points = store.load_refine_points(CONFIG, fit)


def numbers(true_to_pixel):
    n = refine._numbers(true_to_pixel)
    p = refine._pinhole(n)
    return f"f {n[0]:.1f}, centre ({n[1]:.1f}, {n[2]:.1f}), at {np.round(p.position, 1).tolist()}"


print("saved camera.json:", numbers(fit.true_to_pixel), fit.refined)
print("cubes alone (CALIBRATE):", numbers(fit.cube_fit().true_to_pixel))
for count in range(3, len(points) + 1):
    shown = table.describe(fit, points[:count])
    print(f"\n{count} points: problem {shown['problem']}")
    for row in shown["points"]:
        print(f"  #{row['id']} {row['cube'] or 'red*':6} reading ({row['x']:7.2f}, {row['y']:7.2f}) z {row['z']:5.2f}"
              f"  XY calib {row['off']['calibrate']:5.2f} after {row['off']['after']}"
              f"  Z calib {row['height']['calibrate']} after {row['height']['after']}")

# Each point left out, the others' camera converged (not 12 steps from the all-points camera).
sightings = [p.sighting for p in points]
alone = fit.cube_fit()
full = refine._camera(alone, sightings)
print("\n4-point camera:", numbers(full))
for index in range(len(sightings)):
    others = [s for k, s in enumerate(sightings) if k != index]
    on_arm = refine._OnArm(fit.frame_size, others, full)
    for label, start, its in (("12 steps from all", refine._numbers(full), 12),
                              ("converged from all", refine._numbers(full), 300),
                              ("converged from CALIBRATE", refine._numbers(alone.true_to_pixel), 300)):
        on_arm = refine._OnArm(fit.frame_size, others, alone.true_to_pixel)
        n = least_squares(on_arm.residuals, start, refine.CAMERA_STEPS, huber=EDGE_HUBER_PX, iterations=its,
                          derivatives=on_arm.derivatives)[0]
        matrix = refine._pinhole(n).matrix
        seen = refine._seen(matrix, sightings[index])
        reading = HANDEDNESS @ np.array([sightings[index].x, sightings[index].y])
        print(f"  #{index + 1} left out, {label:26}: off {np.linalg.norm(seen - reading):5.2f} mm "
              f"(d {np.round(HANDEDNESS @ (seen - reading), 2).tolist()})  camera {numbers(matrix)}")
