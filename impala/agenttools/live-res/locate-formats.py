"""The red cube found in a YUYV full frame, an MJPG full frame and the 800x600 live view, as FIND RED CUBE does.

    ../.venv/bin/python agenttools/live-res/locate-formats.py      # the arm out of the picture, the red cube in view
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "calibration"))
sys.path.insert(0, str(ROOT))
from camera import store                                         # noqa: E402
from camera.capture import device_index, request_mode            # noqa: E402
from camera.devices import find_camera                           # noqa: E402
from camera.locate import find_red_cube                          # noqa: E402


def frames(fourcc: str, width: int, height: int, count: int, settle_s: float):
    camera = cv2.VideoCapture(device_index(find_camera()), cv2.CAP_V4L2)
    request_mode(camera, fourcc, width, height)
    started = time.monotonic()
    while time.monotonic() - started < settle_s:
        camera.read()
    taken = [camera.read()[1] for _ in range(count)]
    camera.release()
    return taken


fit = store.load_camera(ROOT / "config")
for name, fourcc, size, settle in (("YUYV 2592", "YUYV", (2592, 1944), 4.0), ("MJPG 2592", "MJPG", (2592, 1944), 1.0),
                                   ("MJPG 800 ", "MJPG", (800, 600), 1.0)):
    found = []
    for frame in frames(fourcc, *size, 5, settle):
        started = time.monotonic()
        cube = find_red_cube(frame, fit)
        found.append((cube.x, cube.y, cube.turn_deg, cube.edge_px, time.monotonic() - started))
    found = np.array(found)
    mean = found.mean(0)
    print(f"{name}: x {mean[0]:7.2f} y {mean[1]:7.2f} turn {mean[2]:5.1f}  spread x {np.ptp(found[:, 0]):.2f} "
          f"y {np.ptp(found[:, 1]):.2f} mm  edge {mean[3]:.2f} px  {mean[4]:.2f} s")
