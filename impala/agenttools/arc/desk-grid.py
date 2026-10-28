"""The desk grid the Fine Tune page draws (tests/view.js drawDeskGrid), drawn on a frame offline: desk space every
10 mm through camera.json, every 50 mm stronger, the lines through the arm's turning axis stronger still. No arm.

python desk-grid.py FRAME.png OUT.jpg [config-dir] [crop: x0,y0,x1,y1]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
frame = cv2.imread(sys.argv[1])
config = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "-" else ROOT / "config"
M = np.array(json.loads((config / "camera.json").read_text())["desk_to_pixel"])
crop = [int(value) for value in sys.argv[4].split(",")] if len(sys.argv) > 4 else None

scale = 4
big = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if crop else frame.copy()
factor = scale if crop else 1
overlay = big.copy()
steps = np.arange(-300, 301, 10)
for k in steps:
    width, alpha = (2, 0.85) if k == 0 else (1, 0.6) if k % 50 == 0 else (1, 0.4)
    for points in (np.column_stack((np.full_like(steps, k), steps)), np.column_stack((steps, np.full_like(steps, k)))):
        desk = np.column_stack((points, np.zeros(len(points)), np.ones(len(points)))) @ M.T
        pixels = (desk[:, :2] / desk[:, 2:]) * factor
        layer = big.copy()
        cv2.polylines(layer, [np.round(pixels * 16).astype(np.int32)], False, (255, 164, 201), width, cv2.LINE_AA,
                      shift=4)
        big = cv2.addWeighted(layer, alpha, big, 1 - alpha, 0)
if crop:
    x0, y0, x1, y1 = crop
    big = big[y0 * scale:y1 * scale, x0 * scale:x1 * scale]
cv2.imwrite(sys.argv[2], big, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("->", sys.argv[2], big.shape)
