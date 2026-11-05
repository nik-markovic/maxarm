#!/usr/bin/env python3
"""Has the camera moved since CALIBRATE's frame? Bare desk (wood grain) in two frames, patch by patch: phase
correlation of their gradients, then an affine map from the patches' moves. No cubes, no arm in the patches.

    nov5-moved.py then.png now.png
"""
import sys

import cv2
import numpy as np

PATCHES = [(100, 600, 600, 1000), (100, 1050, 700, 1450), (800, 600, 1300, 1000), (1700, 600, 2050, 1000),
           (800, 1100, 1250, 1450), (1750, 1050, 2050, 1330), (1300, 600, 1650, 950)]


def gradient(path):
    grey = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    grey = cv2.GaussianBlur(grey, (0, 0), 1.0)
    return np.hypot(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1))


then, now = gradient(sys.argv[1]), gradient(sys.argv[2])
src, dst = [], []
for x0, y0, x1, y1 in PATCHES:
    a, b = then[y0:y1, x0:x1], now[y0:y1, x0:x1]
    (dx, dy), peak = cv2.phaseCorrelate(a - a.mean(), b - b.mean(), cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
    c = ((x0 + x1) / 2, (y0 + y1) / 2)
    print(f"   patch at ({c[0]:.0f}, {c[1]:.0f}): moved ({dx:+.2f}, {dy:+.2f}) px, peak {peak:.2f}")
    if peak >= 0.1:
        src.append(c)
        dst.append((c[0] + dx, c[1] + dy))
src, dst = np.array(src), np.array(dst)
moves = dst - src
print(f"mean move ({moves[:, 0].mean():+.2f}, {moves[:, 1].mean():+.2f}) px, spread {moves.std(axis=0).round(2).tolist()}")
if len(src) >= 3:
    A = np.linalg.lstsq(np.c_[src, np.ones(len(src))], dst, rcond=None)[0].T
    left = np.linalg.norm(np.c_[src, np.ones(len(src))] @ A.T - dst, axis=1)
    print(f"affine from {len(src)} patches, each within {left.max():.2f} px:\n{np.round(A, 5)}")
