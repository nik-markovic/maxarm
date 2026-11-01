#!/usr/bin/env python3
"""CALIBRATE on frames made harder: smaller (a camera further off), paler (a camera that renders less colour),
cast (another white balance: a channel's gain), darker and lighter. Where does it stop finding the cubes?

    .venv/bin/python impala/agenttools/green/stress.py [calibration-dir]

The owner's frame of today (green/now-quality.png, today's points) and the C920 frame of 28 Oct with its points.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

IMPALA = Path(__file__).resolve().parents[2]
CALIBRATION = Path(sys.argv[1]) if len(sys.argv) > 1 else IMPALA / "calibration"
sys.path[:0] = [str(IMPALA), str(CALIBRATION)]

from camera.fit import fit_camera      # noqa: E402

FRAMES = [("today", IMPALA / "agenttools/green/now-quality.png", IMPALA / "config/camera-points.json"),
          ("28 Oct", IMPALA / "config/camera-frame-previous.png", IMPALA / "config/camera-points-previous.json")]


def lab_changed(frame, chroma=1.0, shift=(0.0, 0.0), lightness=1.0):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] = np.clip(lab[..., 0] * lightness, 0, 255)
    lab[..., 1] = np.clip((lab[..., 1] - 128) * chroma + 128 + shift[0], 0, 255)
    lab[..., 2] = np.clip((lab[..., 2] - 128) * chroma + 128 + shift[1], 0, 255)
    return cv2.cvtColor(lab.round().astype(np.uint8), cv2.COLOR_LAB2BGR)


def variants(frame):
    height, width = frame.shape[:2]
    for share in (0.5, 0.35, 0.25, 0.18):
        yield f"{share:.0%} size", cv2.resize(frame, (round(width * share), round(height * share)),
                                              interpolation=cv2.INTER_AREA)
    for chroma in (0.8, 0.65, 0.5):
        yield f"colour x{chroma}", lab_changed(frame, chroma=chroma)
    for gains, label in (((1.15, 1, 1), "blue x1.15"), ((0.87, 1, 1), "blue x0.87"), ((1, 1, 1.15), "red x1.15"),
                         ((1, 1, 0.87), "red x0.87"), ((1, 1.1, 1), "green x1.1"), ((1, 0.9, 1), "green x0.9")):
        yield label, np.clip(frame.astype(np.float32) * gains, 0, 255).round().astype(np.uint8)
    for light in (0.6, 1.25):
        yield f"lightness x{light}", lab_changed(frame, lightness=light)


def main():
    for name, frame_path, points_path in FRAMES:
        frame = cv2.imread(str(frame_path))
        points = {colour: (point["x"], point["y"])
                  for colour, point in json.loads(points_path.read_text())["cubes"].items()}
        for label, changed in variants(frame):
            try:
                fit = fit_camera(changed, points)
                result = ", ".join(f"{colour} {fit.cube_off_mm(colour):.1f}" for colour in fit.cubes) + " mm"
            except Exception as error:
                result = f"REFUSED: {str(error)[:90]}"
            print(f"{name:7s} {label:15s} {result}")


if __name__ == "__main__":
    main()
