#!/usr/bin/env python3
"""The book frame (4 Nov, after the camera was moved): the grid sheet on the book (its top 40 mm, the owner) and the
two desk sheets, each one's lines found and numbered, kept as samples -- a pixel with one known sheet coordinate, mm
-- in out/book-sheets.npz. The old camera.json shifted by the move seen on the arm's base (79 px up) only helps find
them: rectify each sheet's plane, keep paper (light, colourless), find the thin dark lines. Each sheet then fitted as
a flat sheet (a homography) to say how straight and even its lines are.

    sheets2.py
"""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
JACKAL = HERE.parents[2]
spec = importlib.util.spec_from_file_location("analyse", HERE.parent / "gridrun" / "analyse.py")
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)

P = np.array([[1, 0, 1.6], [0, 1, -78.6], [0, 0, 1.0]]) @ np.array(
    json.loads((JACKAL / "config" / "camera.json").read_text())["desk_to_pixel"])
frame = cv2.imread(str(HERE / "out" / "book.png"))
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
light = cv2.GaussianBlur(gray, (0, 0), 25)
paper = ((chroma < 12) & (gray > 0.6 * light.max())).astype(np.float32)
paper = cv2.erode(paper, np.ones((9, 9), np.uint8))
# desk windows (desk mm, the old camera's frame): the paper's part of each sheet, the book's striped edge left out
SHEETS = {"book": (40.0, (-90.0, 148.0, -272.0, -124.0)),
          "left": (0.0, (180.0, 238.0, -255.0, -106.0)),
          "right": (0.0, (-252.0, -197.0, -260.0, -98.0))}


def plane(z):
    Q = P.copy()
    Q[:, 3] = P[:, 2] * z + P[:, 3]
    return Q


def samples(rect, mask, X, Y, across):
    out = []
    S = an.S
    if across == "x":
        for yb in np.arange(Y[0] + 3, Y[-1] - 3, 4.0):
            rows = slice(int((yb - 1.5 - Y[0]) * S), int((yb + 1.5 - Y[0]) * S))
            ok = mask[rows].mean(0) > 0.95
            prof = rect[rows].mean(0)
            for seg in an._segments(ok, 150):
                out += [(x, yb) for x in an.minima(prof[seg], X[seg])]
    else:
        for xb in np.arange(X[0] + 3, X[-1] - 3, 4.0):
            cols = slice(int((xb - 1.5 - X[0]) * S), int((xb + 1.5 - X[0]) * S))
            ok = mask[:, cols].mean(1) > 0.95
            prof = rect[:, cols].mean(1)
            for seg in an._segments(ok, 150):
                out += [(y, xb) for y in an.minima(prof[seg], Y[seg])]
    return np.array(out)


def number_lines(s):
    """Each sample's line number, the spacing found first (the rough camera may stretch it a few %): across =
    a + d k + t along, k whole."""
    best = None
    for d in np.arange(4.5, 5.5, 0.002):
        v = np.abs(np.mean(np.exp(2j * np.pi * s[:, 0] / d)))
        if best is None or v > best[0]:
            best = (v, d)
    d = best[1]
    a, t = d * np.angle(np.mean(np.exp(2j * np.pi * s[:, 0] / d))) / (2 * np.pi), 0.0
    for limit in (1.5, 1.0, 0.6, 0.4, 0.4, 0.4):
        k = np.round((s[:, 0] - a - t * s[:, 1]) / d)
        A = np.c_[np.ones(len(k)), k, s[:, 1]]
        r = s[:, 0] - A @ np.array([a, d, t])
        keep = np.abs(r) < limit
        a, d, t = np.linalg.lstsq(A[keep], s[keep, 0], rcond=None)[0]
    k = np.round((s[:, 0] - a - t * s[:, 1]) / d)
    return k.astype(int), s[:, 0] - (a + d * k + t * s[:, 1]), d / 5.0, a


def find(Q, window, keep_mm):
    """The lines in the plane Q maps (x, y) of, over the window: pixels, line mm, which family."""
    X, Y, rect = an.rectify(gray, Q, *window)
    _, _, mask = an.rectify(paper, Q, *window)
    pix, mm, is_x, counts = [], [], [], []
    for across in ("x", "y"):
        s = samples(rect, mask, X, Y, across)
        k, r, scale, _ = number_lines(s)
        good = np.abs(r) < keep_mm
        for (a, b), kk in zip(s[good], k[good]):
            x, y = (a, b) if across == "x" else (b, a)
            pix.append(an.project(Q, np.array([[x, y, 0.0]]))[0])
            mm.append(5.0 * kk)
            is_x.append(across == "x")
        counts.append(f"{across}: {len(s)} samples, {good.sum()} kept on {len(set(k[good]))} lines, "
                      f"spacing {5 * scale:.3f}")
    return np.array(pix), np.array(mm), np.array(is_x), "; ".join(counts)


def flat(pix, mm, is_x):
    """The sheet as a flat sheet: the 3x3 from sheet mm to pixels, and each sample's miss, mm."""
    centre, spread = pix.mean(0), pix.std()
    c = np.c_[(pix - centre) / spread, np.ones(len(pix))]
    rows = [np.r_[v, np.zeros(3), -m * v] if ix else np.r_[np.zeros(3), v, -m * v] for v, m, ix in zip(c, mm, is_x)]
    Hi = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    w = c @ Hi.T
    q = w[:, :2] / w[:, 2:]
    N = np.array([[1 / spread, 0, -centre[0] / spread], [0, 1 / spread, -centre[1] / spread], [0, 0, 1.0]])
    return np.linalg.inv(Hi @ N), np.where(is_x, q[:, 0] - mm, q[:, 1] - mm)


saved = {}
for name, (z, (x0, x1, y0, y1)) in SHEETS.items():
    Q = plane(z)
    pix, mm, is_x, counts = find(Q, (x0, x1, y0, y1), 0.4)
    G, r = flat(pix, mm, is_x)
    print(f"{name}, through the rough camera: {counts}; flat to {np.median(np.abs(r)):.3f} mm")
    # again, through the sheet's own plane: its lines now straight and parallel
    corners = an.project(Q, np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], float))
    w = np.linalg.solve(G, np.c_[corners, np.ones(4)].T)
    s = (w[:2] / w[2]).T
    Gp = np.c_[G[:, :2], np.zeros(3), G[:, 2]]
    pix, mm, is_x, counts = find(Gp, (s[:, 0].min(), s[:, 0].max(), s[:, 1].min(), s[:, 1].max()), 0.3)
    G, r = flat(pix, mm, is_x)
    print(f"{name}, through its own plane: {counts}\n   as a flat sheet: {len(r)} samples, "
          f"{np.median(np.abs(r)):.3f} mm typically, {np.percentile(np.abs(r), 95):.3f} at 95%; pixels x "
          f"{pix[:, 0].min():.0f}-{pix[:, 0].max():.0f}, y {pix[:, 1].min():.0f}-{pix[:, 1].max():.0f}")
    saved[f"{name}_pix"], saved[f"{name}_mm"], saved[f"{name}_is_x"] = pix, mm, is_x
np.savez(HERE / "out" / "book-sheets.npz", **saved)
