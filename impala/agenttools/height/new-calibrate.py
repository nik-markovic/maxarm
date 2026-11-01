#!/usr/bin/env python3
"""The new CALIBRATE and SAVE on the owner's files: CALIBRATE again on camera-frame.png (the arm tied by turn, shift
and stretch), the three cubes as Fine Tune points through it, and SAVE on them alone (the camera from their
outlines, the arm laid by least squares). Nothing is written.

    .venv/bin/python impala/agenttools/height/new-calibrate.py [config-dir]
"""

import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera import check, store                                     # noqa: E402
from camera.fit import CameraFit, fit_camera, stretch_of            # noqa: E402
from camera.refine import _numbers, off_by, refine                  # noqa: E402


def main():
    config = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "config"
    frame = cv2.imread(str(config / "camera-frame.png"))
    saved = CameraFit.from_json(json.loads((config / "camera.json").read_text()))
    arm = {colour: (cube["x"], cube["y"])
           for colour, cube in json.loads((config / "camera-points.json").read_text())["cubes"].items()}
    started = time.monotonic()
    fit = fit_camera(frame, arm)
    print(f"CALIBRATE {time.monotonic() - started:.1f} s: {store.describe_fit(fit)}")
    low, high, skew = stretch_of(fit.desk_to_true)
    print(f"  stretch {low:.4f} to {high:.4f}, {skew:.2f} deg off square; focal {fit.focal:.1f} (saved "
          f"{saved.focal:.1f}); the desk {check.difference_mm(saved, fit)} mm from the saved camera.json")
    points = store.cube_refine_points(fit, {}, "now")
    for point in points:
        off = off_by(fit, point.sighting)
        print(f"  {point.cube:5s} XY {off.across:.3f} mm  Z {off.height:+.2f} mm")
    started = time.monotonic()
    refined = refine(fit, [point.sighting for point in points])
    print(f"SAVE on the three alone {time.monotonic() - started:.1f} s: {refined.refined}; focal "
          f"{_numbers(refined.true_to_pixel)[0]:.1f}; the desk {check.difference_mm(fit, refined)} mm from CALIBRATE's")
    for point in points:
        off = off_by(refined, point.sighting)
        print(f"  {point.cube:5s} XY {off.across:.3f} mm  Z {off.height:+.2f} mm")


if __name__ == "__main__":
    main()
