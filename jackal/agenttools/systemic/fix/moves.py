#!/usr/bin/env python3
"""How far the refit calibration sends the arm from where the saved one does, for a tile seen on the desk (z 2):
the same pixel through both, out from the base / round it (+ toward +x), mm, over the work area.

    moves.py [config_dir]
"""
import math
import sys
from pathlib import Path

import numpy as np

JACKAL = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(JACKAL / "src"), str(JACKAL / "src" / "calibration")]
from applib.camera import store  # noqa: E402
from camera import refine  # noqa: E402

config = Path(sys.argv[1]) if len(sys.argv) > 1 else JACKAL.parent / "work" / "config-jackal-1104"
saved = store.load_camera(config)
new = refine.refine(saved, [p.sighting for p in store.load_refine_points(config, saved)])
print(f"scale {new.scale:.4f}; saved {saved.scale:.4f}")
Z = 2.0


def desk_at(matrix, pixel):
    a = matrix[:, [0, 1, 3]] + np.outer(matrix[:, 2] * 0, 0)
    a = np.column_stack((matrix[:, 0], matrix[:, 1], matrix[:, 2] * Z + matrix[:, 3]))
    x, y, w = np.linalg.solve(a, (*pixel, 1.0))
    return np.array([x / w, y / w])


print("reach \\ bearing " + " ".join(f"{b:+11d}" for b in range(-90, 91, 30)))
for reach in (120, 160, 200, 240, 270):
    cells = []
    for bearing in range(-90, 91, 30):
        b = math.radians(bearing)
        target = np.array([reach * math.sin(b), -reach * math.cos(b)])
        u, v, w = saved.desk_to_pixel @ (*target, Z, 1.0)
        d = desk_at(new.desk_to_pixel, (u / w, v / w)) - target
        cells.append(f"{d @ (math.sin(b), -math.cos(b)):+5.1f}/{d @ (math.cos(b), math.sin(b)):+4.1f}")
    print(f"{reach:5d}           " + " ".join(cells))
