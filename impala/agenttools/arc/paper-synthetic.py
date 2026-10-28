"""A drawn sheet of grid paper, squares exactly 5.000 mm, laid through camera.json's camera where the real sheet lies
in FRAME: to prove paper-lines.py finds 5.000 mm, square corners and straight lines when they are so. No arm.

python paper-synthetic.py FRAME.png OUT.png [turn_deg] [blur_px] [line_dark] [noise]
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
frame = cv2.imread(sys.argv[1])
out = sys.argv[2]
turn = math.radians(float(sys.argv[3])) if len(sys.argv) > 3 else math.radians(23.0)
blur = float(sys.argv[4]) if len(sys.argv) > 4 else 1.8
dark = float(sys.argv[5]) if len(sys.argv) > 5 else 18.0
noise = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0
H = np.array(json.loads((ROOT / "config" / "camera.json").read_text())["true_to_pixel"])[:, [0, 1, 3]]

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
white = ((hsv[..., 1] < 60) & (hsv[..., 2] > 150)).astype(np.uint8)
white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
count, labels, stats, _ = cv2.connectedComponentsWithStats(white)
sheet = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

# Supersampled 2x: each sample's true mm, its distance to the nearest line of each family, darkness of a 0.15 mm line.
height, width = sheet.shape
ys, xs = np.mgrid[0:height * 2, 0:width * 2].astype(np.float64)
pixels = np.stack(((xs + 0.5) / 2 - 0.5, (ys + 0.5) / 2 - 0.5, np.ones_like(xs)), axis=-1)
true = pixels @ np.linalg.inv(H).T
true = true[..., :2] / true[..., 2:]
u = true[..., 0] * math.cos(turn) + true[..., 1] * math.sin(turn) - 1.234
v = -true[..., 0] * math.sin(turn) + true[..., 1] * math.cos(turn) - 2.345
du = np.abs((u + 2.5) % 5.0 - 2.5)
dv = np.abs((v + 2.5) % 5.0 - 2.5)
half_line = 0.075
ink = np.maximum(np.clip(1 - du / half_line, 0, 1), np.clip(1 - dv / half_line, 0, 1))
ink = cv2.resize(ink.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
paper = 235.0 - dark * ink / max(ink.max(), 1e-9) * 1.0
paper = cv2.GaussianBlur(paper.astype(np.float32), (0, 0), blur)
image = frame.astype(np.float32).copy()
for channel in range(3):
    plane = image[..., channel]
    plane[sheet] = paper[sheet]
image += np.random.default_rng(1).normal(0, noise, image.shape)
cv2.imwrite(out, image.clip(0, 255).astype(np.uint8))
print(f"drawn: squares 5.000 mm, turned {math.degrees(turn):.1f} deg, blur {blur} px, lines {dark} darker -> {out}")
