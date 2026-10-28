"""The focal length a plane's homography implies, square pixels, for a given principal point.

python focal-from-plane.py crossings.npz
Each homography gives two equations (the plane's axes at right angles, and equally long) for 1/f^2.
"""
import sys

import cv2
import numpy as np

data = np.load(sys.argv[1])
ij, px = data["ij"].astype(float), data["px"]
H, inl = cv2.findHomography(ij, px, cv2.RANSAC, 3.0)


def focal_for(H, cx, cy):
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    G = T @ H                                  # principal point moved to 0
    h1, h2 = G[:, 0], G[:, 1]
    # w = diag(1/f^2, 1/f^2, 1):  a/f^2 + b = 0 form
    # orthogonality: (h1x h2x + h1y h2y)/f^2 + h1z h2z = 0
    # equal length:  (h1x^2 + h1y^2 - h2x^2 - h2y^2)/f^2 + h1z^2 - h2z^2 = 0
    a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    b = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    each = [np.sqrt(-ai / bi) if -ai / bi > 0 else np.nan for ai, bi in zip(a, b)]
    s = -(a @ b) / (b @ b)
    return np.sqrt(1 / s) if s > 0 else np.nan, each


for cx, cy in ((1295.5, 971.5), (1278, 950), (1291.3, 983.4), (1316.9, 988.9)):
    f, each = focal_for(H, cx, cy)
    print(f"principal point ({cx:7.1f}, {cy:6.1f}): focal {f:7.1f} px  (right angles alone {each[0]:.1f}, equal sides alone {each[1]:.1f})")
