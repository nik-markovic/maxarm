"""YUYV against MJPG at the camera's full size: same picture (crop, shift), how sharp, how long.

    ../.venv/bin/python agenttools/live-res/compare-formats.py     # writes yuyv.png, mjpg.png beside this file
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "calibration"))
from camera.capture import device_index, request_mode            # noqa: E402
from camera.devices import find_camera                           # noqa: E402

HERE = Path(__file__).resolve().parent


def grab(fourcc: str, count: int):
    camera = cv2.VideoCapture(device_index(find_camera()), cv2.CAP_V4L2)
    request_mode(camera, fourcc, 2592, 1944)
    started = time.monotonic()
    for _ in range(8):
        camera.read()
    settled = time.monotonic()
    frames = [camera.read()[1].astype(np.float64) for _ in range(count)]
    done = time.monotonic()
    camera.release()
    print(f"{fourcc}: open+settle {settled - started:.1f} s, {count} frames {done - settled:.1f} s")
    return np.mean(frames, axis=0).round().astype(np.uint8), frames


def sharpness(image):
    return cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


yuyv, _ = grab("YUYV", 5)
mjpg, mjpg_frames = grab("MJPG", 5)
cv2.imwrite(str(HERE / "yuyv.png"), yuyv)
cv2.imwrite(str(HERE / "mjpg.png"), mjpg)
grey = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) for image in (yuyv, mjpg)]
shift, response = cv2.phaseCorrelate(grey[0], grey[1])
print(f"shift MJPG vs YUYV: {shift[0]:+.2f}, {shift[1]:+.2f} px (response {response:.2f})")
scales = []
for scale in (0.995, 0.998, 1.0, 1.002, 1.005):
    warped = cv2.warpAffine(grey[1], cv2.getRotationMatrix2D((1296, 972), 0, scale), (2592, 1944))
    scales.append((np.abs(warped - grey[0])[200:-200, 200:-200].mean(), scale))
print("mean |diff| by MJPG scale:", ", ".join(f"{s}: {d:.2f}" for d, s in scales))
print(f"sharpness (Laplacian var): YUYV avg {sharpness(yuyv):.1f}, MJPG avg {sharpness(mjpg):.1f}, "
      f"MJPG single {sharpness(mjpg_frames[0].round().astype(np.uint8)):.1f}")
print(f"mean colour B,G,R: YUYV {yuyv.reshape(-1, 3).mean(0).round(1)}, MJPG {mjpg.reshape(-1, 3).mean(0).round(1)}")
