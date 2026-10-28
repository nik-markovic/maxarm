"""How far a frame has moved against another, on what does not move: matched features, a homography, then the
median shift of the desk round the arm and of the arm's base. No arm.

python camera-moved.py a.png b.png
"""
import sys

import cv2
import numpy as np

a = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
b = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create(8000)
ka, da = sift.detectAndCompute(a, None)
kb, db = sift.detectAndCompute(b, None)
matches = cv2.BFMatcher().knnMatch(da, db, k=2)
good = [m for m, n in matches if m.distance < 0.7 * n.distance]
pa = np.float32([ka[m.queryIdx].pt for m in good])
pb = np.float32([kb[m.trainIdx].pt for m in good])
H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 1.5)
inl = inl.ravel().astype(bool)
d = pb[inl] - pa[inl]
print(f"{inl.sum()} of {len(good)} matches agree; shift median ({np.median(d[:, 0]):+.2f}, {np.median(d[:, 1]):+.2f}) px, "
      f"spread {np.median(np.linalg.norm(d - np.median(d, 0), axis=1)):.2f} px")
for label, (x, y) in (("the base", (1000, 700)), ("front of the desk", (1300, 1500)), ("far desk", (1800, 400)),
                      ("left desk", (400, 1200))):
    q = cv2.perspectiveTransform(np.float32([[[x, y]]]), H)[0, 0]
    print(f"  {label:18s} ({x}, {y}) moved ({q[0] - x:+.2f}, {q[1] - y:+.2f}) px")
