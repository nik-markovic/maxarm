#!/usr/bin/env python3
"""Fit camera.json to the frame CALIBRATE kept, as CALIBRATE does after its snapshot. No arm, no camera.

    ../.venv/bin/python agenttools/refit-frame.py              # config/camera-frame.png and camera-points.json
    ../.venv/bin/python agenttools/refit-frame.py --dry-run    # say what it would write, write nothing

For when the fit has changed and the cubes, the camera and the arm have not:
the frame and the points on file are what CALIBRATE would take again. It
writes config/camera.json and the check image, as CALIBRATE does.
"""

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera import check, store                     # noqa: E402
from camera.fit import fit_camera                   # noqa: E402

CONFIG = ROOT / "config"


def main(argv) -> int:
    frame = cv2.imread(str(CONFIG / store.FRAME_FILE))
    points = store.load_points(CONFIG)
    if frame is None or None in points.values():
        print(f"needs config/{store.FRAME_FILE} and all three cubes in config/{store.POINTS_FILE}")
        return 1
    started = time.monotonic()
    fit = fit_camera(frame, {colour: (point.x, point.y) for colour, point in points.items()})
    summary = store.describe_fit(fit)
    print(f"fitted in {time.monotonic() - started:.1f} s: {summary}")
    print(f"camera at desk {', '.join(f'{v:.0f}' for v in fit.camera_position)} mm, focal {fit.focal:.0f} px")
    if "--dry-run" in argv:
        return 0
    store.save_camera(CONFIG, fit)
    store.save_check(CONFIG, check.draw(frame, fit, ["camera.json, fitted to the saved frame", summary]))
    print(f"wrote config/{store.CAMERA_FILE} and config/{store.CHECK_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
