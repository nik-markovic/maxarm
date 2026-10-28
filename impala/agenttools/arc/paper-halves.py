"""The squares' size on the near half of a sheet and on its far half (from the camera), each direction of line
apart: does the camera's scale change across the sheet? paper-lines.py's places; camera.json's camera. No arm.

python paper-halves.py FRAME-lines.npz
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
data = np.load(sys.argv[1])
P = np.array(json.loads((ROOT / "config" / "camera.json").read_text())["true_to_pixel"])
_, _, vt = np.linalg.svd(P)
camera = vt[-1][:3] / vt[-1][3]
middle = data["middle"]
for name in ("a", "b"):
    angle = float(data[f"{name}_angle"])
    normal = np.array([math.cos(angle), math.sin(angle)])
    along = np.array([-normal[1], normal[0]])
    rows = data[f"{name}_rows"]
    true = middle + rows[:, 2:3] * normal + rows[:, 1:2] * along
    distance = np.linalg.norm(np.column_stack((true, np.zeros(len(true)))) - camera, axis=1)
    cut = np.median(distance)
    parts = []
    for label, mine in (("near", distance <= cut), ("far", distance > cut)):
        k, band, place = rows[mine].T
        design = np.column_stack((np.ones_like(k), k, band))
        keep = np.ones(len(k), bool)
        for _ in range(4):
            solution, *_ = np.linalg.lstsq(design[keep], place[keep], rcond=None)
            left = place - design @ solution
            keep = np.abs(left) < max(4 * np.median(np.abs(left[keep])), 0.05)
        parts.append(f"{label} half ({distance[mine].min():.0f}-{distance[mine].max():.0f} mm from the camera): "
                     f"squares {solution[1]:.4f} mm, {keep.sum()} places")
    print(f"lines {name}: " + "; ".join(parts))
