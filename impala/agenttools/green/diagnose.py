#!/usr/bin/env python3
"""Why the cube finder misses a cube in a frame: the desk's chroma, the Otsu cut, and each cube face's chroma.

    .venv/bin/python impala/agenttools/green/diagnose.py FRAME.png x,y [x,y ...]

Each x,y is a pixel on a face to sample (a 15 px square round it). Prints the numbers find_blobs() decides by.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera import cubes   # noqa: E402


def main():
    frame = cv2.imread(sys.argv[1])
    desk = cubes.desk_chroma(frame)
    chroma = cubes.chroma_from_desk(frame, desk)
    distance = np.hypot(chroma[..., 0], chroma[..., 1])
    scaled = np.clip(distance * 4.0, 0, 255).astype(np.uint8)
    cut, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    print(f"frame {frame.shape[1]}x{frame.shape[0]}; desk a*,b* = {desk[0]:+.1f}, {desk[1]:+.1f} "
          f"(from grey {np.hypot(*desk):.1f}); Otsu cut {cut / 4:.1f} chroma from the desk")
    for spot in sys.argv[2:]:
        x, y = map(int, spot.split(","))
        patch = (slice(y - 7, y + 8), slice(x - 7, x + 8))
        a, b = chroma[patch][..., 0].mean() + desk[0], chroma[patch][..., 1].mean() + desk[1]
        bgr = frame[patch].reshape(-1, 3).mean(axis=0)
        print(f"  ({x}, {y}): BGR {bgr.round().astype(int)}, L {lab[patch][..., 0].mean() * 100 / 255:.0f}, "
              f"a*,b* {a:+.1f}, {b:+.1f}; from the desk {distance[patch].mean():.1f}, from grey {np.hypot(a, b):.1f}, "
              f"hue {np.degrees(np.arctan2(b, a)) % 360:.0f} deg")


if __name__ == "__main__":
    main()
