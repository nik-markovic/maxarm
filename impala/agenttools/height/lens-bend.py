#!/usr/bin/env python3
"""Does a bending lens make CALIBRATE's three cubes miss the arm's readings? CALIBRATE again on the frame with a trial
radial bend k1 undone (round the frame's centre, at the cubes' focal length), for a range of k1.

    .venv/bin/python impala/agenttools/height/lens-bend.py [config-dir]

Prints, for each k1: each cube from its arm reading (XY, mm) and its height from 40 (Z), the cubes' edges (px, as
CALIBRATE gives them), and the focal length. Nothing is written.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera import store                                        # noqa: E402
from camera.fit import CameraFit, fit_camera                     # noqa: E402
from camera.refine import _numbers, off_by                       # noqa: E402

BENDS = (-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16)


def main():
    config = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "config"
    frame = cv2.imread(str(config / "camera-frame.png"))
    saved = CameraFit.from_json(json.loads((config / "camera.json").read_text()))
    arm = {colour: (cube["x"], cube["y"])
           for colour, cube in json.loads((config / "camera-points.json").read_text())["cubes"].items()}
    height, width = frame.shape[:2]
    lens = np.array([[saved.focal, 0, width / 2], [0, saved.focal, height / 2], [0, 0, 1]])
    print("   k1    blue XY/Z      green XY/Z     red XY/Z    edges px    focal")
    for bend in BENDS:
        bends = np.array([bend, 0, 0, 0])
        whole = cv2.getOptimalNewCameraMatrix(lens, bends, (width, height), 1)[0]   # every pixel kept in the frame
        try:
            fit = fit_camera(cv2.undistort(frame, lens, bends, None, whole), arm)
        except Exception as reason:                                             # noqa: BLE001
            print(f"  {bend:+.2f}  {str(reason)[:90]}")
            continue
        offs = {point.cube: off_by(fit, point.sighting) for point in store.cube_refine_points(fit, {}, "now")}
        cells = "  ".join(f"{offs[c].across:5.2f} {offs[c].height:+5.2f}" for c in ("blue", "green", "red"))
        print(f"  {bend:+.2f}  {cells}   {fit.edge_px[0]:.2f}/{fit.edge_px[1]:.2f}   {_numbers(fit.true_to_pixel)[0]:.0f}")


if __name__ == "__main__":
    main()
