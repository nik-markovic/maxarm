"""Is the picture soft for want of focus? The camera's manual focus stepped through its range, the paper's cut edge
measured at each step (10-90% rise, px), then the focus put back as it was and the picture compared with the one
taken before. MJPG at full size. No arm.

    ../.venv/bin/python agenttools/live-res/focus-sweep.py OUT_DIR
"""
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "calibration"))
from camera.devices import find_camera  # noqa: E402
from camera.feed import device_index  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("sharp_edge", Path(__file__).resolve().parent / "sharp-edge.py")
sharp_edge = importlib.util.module_from_spec(spec)
sys.argv, argv = [sys.argv[0]], sys.argv
spec.loader.exec_module(sharp_edge)
sys.argv = argv
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
device = find_camera()


def control(name: str, value=None):
    if value is None:
        text = subprocess.run(["v4l2-ctl", "-d", device, "-C", name], capture_output=True, text=True).stdout
        return int(text.split(":")[1])
    subprocess.run(["v4l2-ctl", "-d", device, "-c", f"{name}={value}"], check=True)


camera = cv2.VideoCapture(device_index(device), cv2.CAP_V4L2)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def frame_after(seconds: float):
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        camera.read()
    return np.mean([camera.read()[1].astype(np.float64) for _ in range(3)], axis=0).round().astype(np.uint8)


original = control("focus_absolute")
print(f"focus_absolute was {original}; autofocus {control('focus_automatic_continuous')}")
before = frame_after(2.0)
cv2.imwrite(str(out / "focus-before.png"), before)
try:
    for value in (0, 40, 80, 120, 160, 200, 260, 320, 400, 500, 650, 800, 1023):
        control("focus_absolute", value)
        frame = frame_after(1.2)
        width, rows = sharp_edge.paper_edge_width(frame)
        print(f"focus {value:4d}: paper edge {width:5.2f} px")
        cv2.imwrite(str(out / f"focus-{value:04d}.png"), frame)
finally:
    control("focus_absolute", original)
after = frame_after(2.0)
camera.release()
cv2.imwrite(str(out / "focus-after.png"), after)
grey = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) for image in (before, after)]
shift, response = cv2.phaseCorrelate(grey[0], grey[1])
print(f"focus put back to {control('focus_absolute')}: picture shift {shift[0]:+.2f}, {shift[1]:+.2f} px against "
      f"before; paper edge before {sharp_edge.paper_edge_width(before)[0]:.2f}, after "
      f"{sharp_edge.paper_edge_width(after)[0]:.2f} px")
