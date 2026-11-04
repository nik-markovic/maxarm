#!/usr/bin/env python3
"""Cameras that draw the desk exactly as camera.json does but stand higher or lower: the desk plane (z = 0) alone
cannot tell them apart -- a camera higher up sees it with a longer lens, aimed a little otherwise. Only things above
the desk can. For each height, where the rim fit (rim-fit.py) puts the cup on the centre line at z 1 and z 42
(frames/c<y>.jpg, c<y>-z42.jpg), and how far the cubes' tops from CALIBRATE land from their own pixels.

    camera-height.py
"""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CAMERA = HERE.parents[1] / "config" / "camera.json"
spec = importlib.util.spec_from_file_location("rim_fit", HERE / "rim-fit.py")
rim_fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rim_fit)
# the centre line: sent y, the servos' z at 1 and at 42
ROWS = [(-120, 1.45, 42.65), (-150, 2.0, 42.81), (-180, 1.67, 43.22), (-210, 1.87, 43.8), (-240, 2.25, 42.87),
        (-265, 2.18, 43.43)]


def family(Q, cy):
    """The square-pixel camera, principal point row `cy`, with Q's desk plane: its 3x4 (Q's scale), centre, f, cx."""
    a, b, h = Q[:, 0], Q[:, 1], Q[:, 3]

    def row(u, v):          # u' M v, M = [[1, 0, -cx], [0, 1, -cy], [-cx, -cy, s]], as coefficients of cx, s, 1
        return np.array([-(u[0] * v[2] + u[2] * v[0]), u[2] * v[2],
                         u[0] * v[0] + u[1] * v[1] - cy * (u[1] * v[2] + u[2] * v[1])])
    ab, aa, bb = row(a, b), row(a, a), row(b, b)
    cx, s = np.linalg.solve(np.array([ab[:2], (aa - bb)[:2]]), -np.array([ab[2], (aa - bb)[2]]))
    f = np.sqrt(s - cx ** 2 - cy ** 2)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    Ki = np.linalg.inv(K)
    lam = 1 / np.linalg.norm(Ki @ a)
    r1, r2, t = lam * Ki @ a, lam * Ki @ b, lam * Ki @ h
    if t[2] < 0:
        r1, r2, t, lam = -r1, -r2, -t, -lam
    r3 = np.cross(r1, r2)
    centre = -np.c_[r1, r2, r3].T @ t
    if centre[2] < 0:
        r3 = -r3
        centre = -np.c_[r1, r2, r3].T @ t
    return np.c_[a, b, K @ r3 / lam, h], centre, f, cx


def at_height(Q, height):
    lo, hi = 700.0, 750.0                       # principal point rows spanning 153 to 278 mm up
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if family(Q, mid)[1][2] < height else (lo, mid)
    return family(Q, (lo + hi) / 2)


def main() -> None:
    cam = json.loads(CAMERA.read_text())
    Q = np.array(cam["true_to_pixel"]); D2T = np.array(cam["desk_to_true"]); T2D = np.linalg.inv(D2T)
    to_true = np.eye(4); to_true[:2, :2] = D2T
    frames = {(y, z): cv2.imread(str(HERE / "frames" / f"c{y}{'-z42' if z > 20 else ''}.jpg"))
              for y, z1, z42 in ROWS for z in (z1, z42)}
    saved_height = family(Q, 725.5)[1][2]
    for height in (210, 220, round(saved_height, 1), 245):
        Qa, centre, f, cx = at_height(Q, height)
        P = Qa @ to_true
        # the cubes' tops: where this camera draws the centre of each cube's top against the saved camera
        tops = []
        for colour, cube in cam["cubes"].items():
            p = np.array([cube["x"], cube["y"], 40.0, 1.0])
            here, saved = Qa @ to_true @ p, np.array(cam["desk_to_pixel"]) @ p
            tops.append(f"{colour} {np.linalg.norm(here[:2] / here[2] - saved[:2] / saved[2]):.1f}px")
        line = []
        for y, z1, z42 in ROWS:
            low = rim_fit.fit(frames[(y, z1)], P, T2D, 0.0, y, z1)[0][2]
            high = rim_fit.fit(frames[(y, z42)], P, T2D, 0.0, y, z42)[0][2]
            line.append(f"{y}: {low:.1f} / {high:.1f} / {high - low:+.1f}")
        print(f"{centre[2]:.0f} mm up, f {f:.0f}, principal point {cx - 1280:+.0f}px across from the frame's middle | "
              f"cube tops moved {', '.join(tops)}")
        print("    z1 / z42 / way down:  " + " | ".join(line))


if __name__ == "__main__":
    main()
