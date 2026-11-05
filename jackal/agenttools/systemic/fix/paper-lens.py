#!/usr/bin/env python3
"""The lens, by the owner's 5 mm grid paper alone (no arm, no cubes): the paper's lines are straight and evenly
spaced, so the pixels of the line samples, undistorted, must lie on one flat sheet (a homography). Fitted with no
lens term, with k1, with k1 and k2, about the frame's middle; and k1 with the centre free.

    paper-lens.py
"""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("analyse", HERE.parent / "gridrun" / "analyse.py")
analyse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyse)

camera = json.loads((HERE.parents[2] / "config" / "camera-previous.json").read_text())   # 3 Nov, near the paper frame's pose
P = np.array(camera["desk_to_pixel"])
width, height = camera["frame_size"]
FOCAL = 1900.0
gray = cv2.cvtColor(cv2.imread(str(HERE.parent / "gridrun" / "frames" / "paper-home.jpg")),
                    cv2.COLOR_BGR2GRAY).astype(np.float32)
X, Y, rect = analyse.rectify(gray, P, -115.0, 115.0, -255.0, -60.0)
xs, ys = analyse.line_samples(rect, X, Y, "x"), analyse.line_samples(rect, X, Y, "y")
kx, rx, *_ = analyse.number_lines(xs)
ky, ry, *_ = analyse.number_lines(ys)
pix, mm, is_x = [], [], []
for (x, y), k, r in zip(xs, kx, rx):
    if abs(r) < 0.4:
        pix.append(analyse.project(P, np.array([[x, y, 0.0]]))[0]); mm.append(5.0 * k); is_x.append(True)
for (y, x), k, r in zip(ys, ky, ry):
    if abs(r) < 0.4:
        pix.append(analyse.project(P, np.array([[x, y, 0.0]]))[0]); mm.append(5.0 * k); is_x.append(False)
pix, mm, is_x = np.array(pix), np.array(mm), np.array(is_x)
r_px = np.linalg.norm(pix - (width / 2, height / 2), axis=1)
print(f"{len(pix)} samples; pixels x {pix[:, 0].min():.0f}-{pix[:, 0].max():.0f}, y {pix[:, 1].min():.0f}-"
      f"{pix[:, 1].max():.0f}; up to {r_px.max():.0f} px from the middle (r/f {r_px.max() / FOCAL:.2f})")


def undistort(p, k1, k2, cx, cy):
    d = (p - (cx, cy)) / FOCAL
    r2 = np.sum(d * d, axis=1)[:, None]
    return (cx, cy) + d * FOCAL * (1 + k1 * r2 + k2 * r2 * r2)


def homography_residuals(q):
    """DLT for paper = H q, then each sample's miss across its line, mm."""
    centre, spread = q.mean(0), q.std()
    c = np.c_[(q - centre) / spread, np.ones(len(q))]
    rows = [np.r_[v, np.zeros(3), -m * v] if ix else np.r_[np.zeros(3), v, -m * v] for v, m, ix in zip(c, mm, is_x)]
    H = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    w = c @ H.T
    paper = w[:, :2] / w[:, 2:]
    return np.where(is_x, paper[:, 0] - mm, paper[:, 1] - mm)


def solve(function, x, iterations=40):
    """Gauss-Newton, each sample weighted down past 0.3 mm (soft), finite differences."""
    for _ in range(iterations):
        r = function(x)
        w = 1.0 / np.sqrt(1.0 + (r / 0.3) ** 2)
        J = np.column_stack([(function(x + np.eye(len(x))[i] * h) - r) / h
                             for i, h in enumerate(np.where(np.abs(x) > 10, 0.01, 1e-5))])
        step = np.linalg.lstsq(J * w[:, None], -r * w, rcond=None)[0]
        x = x + step
        if np.all(np.abs(step) < np.where(np.abs(x) > 10, 1e-3, 1e-7)):
            break
    return x


def residuals(params, free):
    full = dict(k1=0.0, k2=0.0, cx=width / 2, cy=height / 2)
    full.update(zip(free, params))
    return homography_residuals(undistort(pix, full["k1"], full["k2"], full["cx"], full["cy"]))


for free, start in (([], []), (["k1"], [0.0]), (["k1", "k2"], [0.0, 0.0]), (["k1", "cx", "cy"],
                                                                        [0.0, width / 2, height / 2])):
    params = solve(lambda p: residuals(p, free), np.array(start, float)) if free else []
    r = residuals(params, free)
    shown = ", ".join(f"{n} {v:+.4f}" if n.startswith("k") else f"{n} {v:.0f}" for n, v in zip(free, params))
    print(f"{shown or 'no lens term':38s} {np.median(np.abs(r)):.3f} mm typically, {np.percentile(np.abs(r), 95):.3f} "
          f"at 95%, rms {np.sqrt(np.mean(np.minimum(np.abs(r), 1) ** 2)):.3f}")
for k1 in (-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10):
    r = residuals([k1], ["k1"])
    print(f"   undistort k1 {k1:+.2f}: {np.median(np.abs(r)):.3f} typ, {np.percentile(np.abs(r), 95):.3f} at 95%")


# --- one camera dimension: the pixels' aspect and focal length from the paper alone (Zhang, one view) -------------

def homography(q):
    """H: paper mm (X, Y, 1) -> pixels, from the samples (each fixes X or Y): the inverse of the DLT above."""
    centre, spread = q.mean(0), q.std()
    c = np.c_[(q - centre) / spread, np.ones(len(q))]
    rows = [np.r_[v, np.zeros(3), -m * v] if ix else np.r_[np.zeros(3), v, -m * v] for v, m, ix in zip(c, mm, is_x)]
    Hi = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    Hi = Hi @ np.array([[1 / spread, 0, -centre[0] / spread], [0, 1 / spread, -centre[1] / spread], [0, 0, 1]])
    return np.linalg.inv(Hi)


def intrinsics(H, cx, cy):
    """fx, fy with the principal point at (cx, cy): the paper's axes perpendicular and equally long in the camera."""
    h = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]]) @ H
    (h11, h12), (h21, h22), (h31, h32) = h[0, :2], h[1, :2], h[2, :2]
    A = np.array([[h11 * h12, h21 * h22], [h11 ** 2 - h12 ** 2, h21 ** 2 - h22 ** 2]])
    a, b = np.linalg.solve(A, [-h31 * h32, -(h31 ** 2 - h32 ** 2)])
    return 1 / np.sqrt(a), 1 / np.sqrt(b)


H = homography(pix)
cube_cx, cube_cy = refine_centre = (lambda n: (n[1], n[2]))(
    __import__("importlib").import_module("camera.refine")._numbers(np.array(camera["true_to_pixel"]))) \
    if False else (None, None)
print("\nthe paper alone: focal lengths along the frame's x and y, by where the lens's middle is taken")
for cx, cy in ((width / 2, height / 2), (width / 2 - 100, height / 2), (width / 2 + 100, height / 2),
               (width / 2, height / 2 - 100), (width / 2, height / 2 + 100)):
    fx, fy = intrinsics(H, cx, cy)
    print(f"   middle ({cx:.0f}, {cy:.0f}): fx {fx:.0f}, fy {fy:.0f}, fy/fx {fy / fx:.4f}")


def axis_residuals(params, which):
    """Undistorted along one axis of the frame only: x' = x (1 + k x^2), about the middle."""
    q = pix.copy()
    i = 0 if which == "x" else 1
    d = (q[:, i] - (width / 2, height / 2)[i]) / FOCAL
    q[:, i] = (width / 2, height / 2)[i] + d * FOCAL * (1 + params[0] * d * d)
    return homography_residuals(q)


for which in ("x", "y"):
    k = solve(lambda p: axis_residuals(p, which), np.array([0.0]))
    r = axis_residuals(k, which)
    print(f"one axis only ({which}): k {k[0]:+.4f}: {np.median(np.abs(r)):.3f} typ, {np.percentile(np.abs(r), 95):.3f} "
          f"at 95%")
