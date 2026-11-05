#!/usr/bin/env python3
"""The camera from reality alone, in the book frame: the two desk sheets on the desk (z 0) and the book's sheet on its
top (z = the book's height, 40 mm, the owner), each sheet free to lie anywhere on its plane, turned and maybe face
down; one square-pixel camera through all their lines. No cubes, no arm. Two parallel planes leave the camera's own
numbers to the book's height alone, so the fit is shown at 40 mm and a little either side, with the book's top free
to tilt, and with the optical centre held loosely to the frame's middle and not.

    book-camera.py        writes /tmp/book-camera.pkl
"""
import json
import math
import pickle
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
JACKAL = HERE.parents[2]
W, H = 2560, 1472
data = np.load(HERE / "out" / "book-sheets.npz")
NAMES = ("left", "right", "book")
SHEETS = [(data[f"{n}_pix"], data[f"{n}_mm"], data[f"{n}_is_x"]) for n in NAMES]
P_ROUGH = np.array([[1, 0, 1.6], [0, 1, -78.6], [0, 0, 1.0]]) @ np.array(
    json.loads((JACKAL / "config" / "camera.json").read_text())["desk_to_pixel"]) @ np.diag([-1.0, 1, 1, 1])
CAM, POSE = 9, 9                       # f, cx, cy, rotation 3, shift 3; then 3 per sheet: turn, x, y; then tilt a, b


def matrix(cam):
    K = np.array([[cam[0], 0, cam[1]], [0, cam[0], cam[2]], [0, 0, 1.0]])
    return K @ np.c_[cv2.Rodrigues(np.array(cam[3:6], float))[0], cam[6:9]]


def on_plane(T, pix, z, a=0.0, b=0.0):
    """Pixels to the plane z + a x + b y: x, y."""
    # T @ [x, y, z0 + a x + b y, 1] ~ pix  ->  columns for x, y and the constant
    M = np.c_[T[:, 0] + a * T[:, 2], T[:, 1] + b * T[:, 2], z * T[:, 2] + T[:, 3]]
    w = np.linalg.solve(M, np.c_[pix, np.ones(len(pix))].T)
    return (w[:2] / w[2]).T


def residuals(x, h, flips, prior):
    T = matrix(x[:CAM])
    out = []
    for k, (pix, mm, is_x) in enumerate(SHEETS):
        z, a, b = (h, x[POSE + 9], x[POSE + 10]) if NAMES[k] == "book" else (0.0, 0.0, 0.0)
        P = on_plane(T, pix, z, a, b) - x[POSE + 3 * k + 1: POSE + 3 * k + 3]
        c, s = math.cos(x[POSE + 3 * k]), math.sin(x[POSE + 3 * k])
        X, Y = c * P[:, 0] + s * P[:, 1], flips[k] * (-s * P[:, 0] + c * P[:, 1])
        out.append(np.where(is_x, X - mm, Y - mm))
    if prior:
        out.append(np.array([x[1] - W / 2, x[2] - H / 2]) / (0.1 * W))
    return np.concatenate(out)


def solve(x, free, h, flips, prior, iterations=80):
    x = x.astype(float).copy()
    for _ in range(iterations):
        r = residuals(x, h, flips, prior)
        w = np.where(np.abs(r) <= 0.3, 1.0, 0.3 / np.maximum(np.abs(r), 1e-12))
        J = np.empty((len(r), len(free)))
        for c, i in enumerate(free):
            d = 1e-6 * max(1.0, abs(x[i]))
            y = x.copy(); y[i] += d
            J[:, c] = (residuals(y, h, flips, prior) - r) / d
        A = J.T @ (J * w[:, None])
        step = np.linalg.lstsq(A + 1e-6 * np.diag(np.diag(A)), -(J.T @ (w * r)), rcond=None)[0]
        x[free] += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return x


def start():
    K, R, C = cv2.decomposeProjectionMatrix(P_ROUGH)[:3]
    K = K / K[2, 2]
    C = C[:3, 0] / C[3, 0]
    x = np.zeros(POSE + 11)
    x[:3] = (0.5 * (K[0, 0] + K[1, 1]), K[0, 2], K[1, 2])
    x[3:6] = cv2.Rodrigues(R)[0].ravel()
    x[6:9] = -R @ C
    T = matrix(x[:CAM])
    flips = []
    for k, (pix, mm, is_x) in enumerate(SHEETS):
        z = 40.0 if NAMES[k] == "book" else 0.0
        P = on_plane(T, pix, z)
        best = None
        for flip in (1.0, -1.0):
            for theta in np.radians(np.arange(0, 360, 5)):
                c, s = math.cos(theta), math.sin(theta)
                # the shift that best fits this turn: each sample gives one coordinate
                rows, rhs = [], []
                for (px, py), m, ix in zip(P, mm, is_x):
                    if ix:
                        rows.append((c, s)); rhs.append(c * px + s * py - m)
                    else:
                        rows.append((-flip * s, flip * c)); rhs.append(flip * (-s * px + c * py) - m)
                sh = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
                res = np.array(rhs) - np.array(rows) @ sh
                score = np.median(np.abs(res))
                if best is None or score < best[0]:
                    best = (score, theta, sh, flip)
        x[POSE + 3 * k: POSE + 3 * k + 3] = (best[1], *best[2])
        flips.append(best[3])
    return x, flips


def describe(x):
    R = cv2.Rodrigues(np.array(x[3:6], float))[0]
    lens = -R.T @ x[6:9]
    axis = R.T @ (0, 0, 1.0)
    return (f"f {x[0]:7.1f}, centre ({x[1]:6.1f}, {x[2]:6.1f}), lens {lens[2]:6.1f} up, looks "
            f"{math.degrees(math.asin(-axis[2])):5.2f} deg down")


def report(name, x, h, flips, prior):
    r = residuals(x, h, flips, False)
    parts, at = [], 0
    for k, (pix, _, _) in enumerate(SHEETS):
        rk = r[at: at + len(pix)]; at += len(pix)
        parts.append(f"{NAMES[k]} {np.median(np.abs(rk)):.3f}/{np.percentile(np.abs(rk), 95):.2f}")
    tilt = f", book tilt {math.degrees(math.atan(x[POSE + 9])):+.2f}/{math.degrees(math.atan(x[POSE + 10])):+.2f} deg" \
        if x[POSE + 9] or x[POSE + 10] else ""
    print(f"{name:34s} {describe(x)} | sheets mm typ/95%: {', '.join(parts)}{tilt}", flush=True)


def main():
    x0, flips = start()
    print("flips", flips)
    camera = list(range(CAM))
    poses = [POSE + 3 * k + j for k in (1, 2) for j in range(3)]           # the left sheet holds the frame
    fits = {}
    for h in (40.0, 39.8, 40.2, 40.1):
        for prior in (True, False):
            x = solve(x0, camera + poses, h, flips, prior)
            name = f"book {h:.1f} mm, {'middle held' if prior else 'middle free'}"
            report(name, x, h, flips, prior)
            fits[(h, prior, False)] = x
    for prior in (True, False):
        x = solve(fits[(40.0, prior, False)], camera + poses + [POSE + 9, POSE + 10], 40.0, flips, prior)
        report(f"book 40.0 mm, tilt free, {'held' if prior else 'free'}", x, 40.0, flips, prior)
        fits[(40.0, prior, True)] = x
    pickle.dump({"fits": fits, "flips": flips, "names": NAMES}, open("/tmp/book-camera.pkl", "wb"))


if __name__ == "__main__":
    main()
