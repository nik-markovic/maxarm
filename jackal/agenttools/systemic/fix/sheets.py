#!/usr/bin/env python3
"""The two grid sheets in today's frame: each one's lines found (rectified through camera.json, which only helps find
them), numbered, and kept as samples -- a pixel with one known sheet coordinate, mm -- in out/sheets.npz. Each sheet
then fitted as a flat sheet (a homography) to say how straight and even its lines are.

    sheets.py [frame.png]
"""
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
JACKAL = HERE.parents[2]
spec = importlib.util.spec_from_file_location("analyse", HERE.parent / "gridrun" / "analyse.py")
analyse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyse)

frame_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "out" / "scene.png"
P = np.array(json.loads((JACKAL / "config" / "camera.json").read_text())["desk_to_pixel"])
gray = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2GRAY).astype(np.float32)
SHEETS = {"left": (-15.0, 245.0, -272.0, -86.0), "right": (-256.0, -4.0, -270.0, -78.0)}   # desk x0, x1, y0, y1
saved = {}
for name, (x0, x1, y0, y1) in SHEETS.items():
    X, Y, rect = analyse.rectify(gray, P, x0, x1, y0, y1)
    xs, ys = analyse.line_samples(rect, X, Y, "x"), analyse.line_samples(rect, X, Y, "y")
    pix, mm, is_x = [], [], []
    for samples, across in ((xs, "x"), (ys, "y")):
        k, r, scale, _ = analyse.number_lines(samples)
        for (a, b), kk, rr in zip(samples, k, r):
            if abs(rr) < 0.4:
                x, y = (a, b) if across == "x" else (b, a)
                pix.append(analyse.project(P, np.array([[x, y, 0.0]]))[0])
                mm.append(5.0 * kk)
                is_x.append(across == "x")
        print(f"{name} sheet, lines across desk {across}: {len(samples)} samples on {len(set(k))} lines, "
              f"spacing {5 * scale:.3f} desk mm through camera.json")
    pix, mm, is_x = np.array(pix), np.array(mm), np.array(is_x)
    saved[f"{name}_pix"], saved[f"{name}_mm"], saved[f"{name}_is_x"] = pix, mm, is_x
    # a flat sheet: the DLT of paper = H pixel, each sample fixing X or Y
    centre, spread = pix.mean(0), pix.std()
    c = np.c_[(pix - centre) / spread, np.ones(len(pix))]
    rows = [np.r_[v, np.zeros(3), -m * v] if ix else np.r_[np.zeros(3), v, -m * v] for v, m, ix in zip(c, mm, is_x)]
    Hi = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    w = c @ Hi.T
    q = w[:, :2] / w[:, 2:]
    r = np.where(is_x, q[:, 0] - mm, q[:, 1] - mm)
    print(f"{name} sheet as a flat sheet: {len(r)} samples, {np.median(np.abs(r)):.3f} mm typically, "
          f"{np.percentile(np.abs(r), 95):.3f} at 95%; pixels x {pix[:, 0].min():.0f}-{pix[:, 0].max():.0f}, "
          f"y {pix[:, 1].min():.0f}-{pix[:, 1].max():.0f}")
np.savez(HERE / "out" / "sheets.npz", **saved)
