#!/usr/bin/env python3
"""Snapshots straight from the camera, opened as the app opens it (focus held at 0): QUALITY and FPS, five frames
averaged after a few are let go. Only while the app is not running; no arm.

    .venv/bin/python impala/agenttools/green/take.py OUT-PREFIX [quality|fps ...]
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

IMPALA = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(IMPALA), str(IMPALA / "calibration")]

from camera.devices import FPS, QUALITY, find_camera, view_modes   # noqa: E402
from camera.feed import decode, open_view                          # noqa: E402

SKIP, COUNT = 8, 5


def main():
    prefix = sys.argv[1]
    device = find_camera()
    modes = view_modes(device)
    for choice in sys.argv[2:] or [QUALITY, FPS]:
        mode = modes[choice]
        camera = open_view(device, mode, 0)
        started = time.monotonic()
        images = []
        for index in range(SKIP + COUNT):
            is_read, raw = camera.read()
            if not is_read:
                raise RuntimeError("no frame")
            if index >= SKIP:
                images.append(decode(raw.tobytes()) if mode.fourcc == "MJPG" else raw)
        camera.release()
        average = (sum(image.astype(np.float64) for image in images) / len(images)).round().astype(np.uint8)
        out = f"{prefix}-{choice}.png"
        cv2.imwrite(out, average)
        print(f"{choice}: {mode.describe()} -> {out} ({time.monotonic() - started:.1f} s)")


if __name__ == "__main__":
    main()
