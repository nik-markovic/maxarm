#!/usr/bin/env python3
"""5 Nov night: the owner's 6 points (CALIBRATE's three, then one near each) as the tests page showed them after each
was added -- XY and Z through CALIBRATE's camera and through SAVE's -- and the cameras. No arm, no app: a copy of
config/ in /tmp/c1105.

    nov5.py
"""
import dataclasses
from pathlib import Path

import numpy as np

from applib.camera import store
from camera import refine
from tests import refine as table

CONFIG = Path("/tmp/c1105")

fit = store.load_camera(CONFIG)
points = store.load_refine_points(CONFIG, fit)


def numbers(true_to_pixel):
    n = refine._numbers(true_to_pixel)
    p = refine._pinhole(n)
    return f"f {n[0]:.0f}, centre ({n[1]:.0f}, {n[2]:.0f}), at {np.round(p.position, 1).tolist()}"


print("saved camera.json:", numbers(fit.true_to_pixel), fit.refined)
print("cubes alone (CALIBRATE):", numbers(fit.cube_fit().true_to_pixel))
for count in range(3, len(points) + 1):
    shown = table.describe(fit, points[:count])
    print(f"\n{count} points: problem {shown['problem']}")
    if shown["fit"]:
        print("  fit", shown["fit"])
    for row in shown["points"]:
        print(f"  #{row['id']} {row['cube'] or 'red*':6} reading ({row['x']:7.2f}, {row['y']:7.2f}) z {row['z']:5.2f}"
              f"  XY calib {row['off']['calibrate']:5.2f} after {row['off']['after']}"
              f"  Z calib {row['height']['calibrate']} after {row['height']['after']}")
    try:
        trial = table.refined(fit, points[:count])
        print("  SAVE's camera:", numbers(trial.true_to_pixel))
    except Exception as reason:
        print("  no SAVE:", reason)
