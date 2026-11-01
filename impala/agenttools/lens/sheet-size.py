#!/usr/bin/env python3
"""The sheet's size through camera.json: are the camera's millimetres the paper's? Its four borders found as in
straight.py, each fitted with a straight line, the corners where they cross, taken to the desk (height 0) through
camera.json, and the sides measured. Also: has the camera moved since camera-frame.png (SIFT on what stays put)?

    .venv/bin/python impala/agenttools/lens/sheet-size.py FRAME.png
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parents[1] / "config"
spec = importlib.util.spec_from_file_location("straight", HERE / "straight.py")
straight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(straight)


def line_through(points):
    mean = points.mean(axis=0)
    along = np.linalg.svd(points - mean)[2][0]
    return mean, along


def crossing(first, second):
    (p, u), (q, v) = first, second
    t = np.linalg.solve(np.column_stack((u, -v)), q - p)[0]
    return p + t * u


def main():
    frame = cv2.imread(sys.argv[1])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    camera = json.loads((CONFIG / "camera.json").read_text())
    plane = np.array(camera["desk_to_pixel"])[:, [0, 1, 3]]
    corners = straight.paper_corners(gray)
    lines = [line_through(straight.border_points(gray, corners[i], corners[(i + 1) % 4])) for i in range(4)]
    crossings = [crossing(lines[i - 1], lines[i]) for i in range(4)]
    desk = []
    for pixel in crossings:
        x, y, w = np.linalg.solve(plane, (*pixel, 1.0))
        desk.append(np.array([x / w, y / w]))
    sides = [math.dist(desk[i], desk[(i + 1) % 4]) for i in range(4)]
    diagonals = [math.dist(desk[0], desk[2]), math.dist(desk[1], desk[3])]
    print("corners (px):", [np.round(c, 1).tolist() for c in crossings])
    print("corners (desk mm):", [np.round(d, 1).tolist() for d in desk])
    print("sides (mm):", [round(s, 2) for s in sides], " diagonals:", [round(d, 2) for d in diagonals])
    long_side, short_side = (sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2
    if long_side < short_side:
        long_side, short_side = short_side, long_side
    print(f"mean sides {long_side:.2f} x {short_side:.2f} mm, ratio {long_side / short_side:.4f}")
    for name, size in (("US Letter", (279.4, 215.9)), ("A4", (297.0, 210.0))):
        print(f"  against {name} {size[0]} x {size[1]}: {100 * (long_side / size[0] - 1):+.2f}% and "
              f"{100 * (short_side / size[1] - 1):+.2f}%  (ratio {size[0] / size[1]:.4f})")
    then = cv2.cvtColor(cv2.imread(str(CONFIG / "camera-frame.png")), cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(4000)
    mask = np.full(gray.shape, 255, np.uint8)
    mask[int(min(c[1] for c in crossings)) - 40:, int(min(c[0] for c in crossings)) - 40:] = 0   # not the sheet
    (a, da), (b, db) = sift.detectAndCompute(then, mask), sift.detectAndCompute(gray, mask)
    matches = [m for m, n in cv2.BFMatcher().knnMatch(da, db, k=2) if m.distance < 0.7 * n.distance]
    src = np.float32([a[m.queryIdx].pt for m in matches])
    dst = np.float32([b[m.trainIdx].pt for m in matches])
    shift = np.linalg.norm(dst - src, axis=1)
    still = shift < 2.0
    print(f"since camera-frame.png: {len(matches)} matches, {still.sum()} within 2 px; their median move "
          f"{np.median(shift[still]):.2f} px" if still.any() else "no still matches")


if __name__ == "__main__":
    main()
