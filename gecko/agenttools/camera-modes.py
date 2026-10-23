#!/usr/bin/env python3
"""Is each of the camera's sizes the whole picture, scaled? A still at each, matched against the largest.

    ../.venv/bin/python agenttools/camera-modes.py               # the camera the app would pick; no arm
    ../.venv/bin/python agenttools/camera-modes.py /dev/video2

The scene must hold still while it runs, about 20 s, and have some texture: the
desk's grain is enough. Each size's still is matched to the largest by its
features (SIFT), and the best affine map between them is compared with the plain
scaling the calibration assumes (pixel centres kept, as `model.rescaled()`).
The app needs the camera to itself: stop it first.
"""

import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera.devices import find_camera                        # noqa: E402

SIZES = [(2592, 1944), (2048, 1536), (1920, 1080), (1280, 720), (800, 600), (640, 480)]
SETTLE_FRAMES = 15


def still(device: str, size: Tuple[int, int]) -> np.ndarray:
    capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    frame = None
    for _ in range(SETTLE_FRAMES):
        is_read, latest = capture.read()
        frame = latest if is_read else frame
    capture.release()
    if frame is None:
        raise RuntimeError(f"{device} gave no frame at {size[0]} x {size[1]}")
    return frame


def main(argv) -> int:
    device = argv[1] if len(argv) > 1 else find_camera()
    sift = cv2.SIFT_create(8000)
    frames = {size: still(device, size) for size in SIZES}
    largest = frames[SIZES[0]]
    width, height = largest.shape[1], largest.shape[0]
    points, described = sift.detectAndCompute(cv2.cvtColor(largest, cv2.COLOR_BGR2GRAY), None)
    for size, frame in frames.items():
        if frame.shape[:2] != (size[1], size[0]):
            print(f"{size[0]} x {size[1]}: the camera gave {frame.shape[1]} x {frame.shape[0]}")
            continue
        found, features = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
        pairs = [first for first, second in cv2.BFMatcher().knnMatch(features, described, k=2)
                 if first.distance < 0.7 * second.distance]
        small = np.float32([found[pair.queryIdx].pt for pair in pairs])
        large = np.float32([points[pair.trainIdx].pt for pair in pairs])
        affine, kept = cv2.estimateAffine2D(small, large, ransacReprojThreshold=2.0, maxIters=5000)
        kept = kept.ravel().astype(bool)
        sx, sy = width / size[0], height / size[1]
        scaled = np.column_stack(((small[:, 0] + 0.5) * sx - 0.5, (small[:, 1] + 0.5) * sy - 0.5))
        apart = np.linalg.norm(large[kept] - scaled[kept], axis=1)
        corners = np.float32([[-0.5, -0.5], [size[0] - 0.5, size[1] - 0.5]]) @ affine[:, :2].T + affine[:, 2]
        print(f"{size[0]:4d} x {size[1]:4d}: {kept.sum():3d} matches; from plain scaling {np.median(apart):6.2f} px "
              f"of the largest (median); it covers ({corners[0, 0]:.0f}, {corners[0, 1]:.0f}) to "
              f"({corners[1, 0]:.0f}, {corners[1, 1]:.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
