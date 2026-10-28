"""The camera's two full-size formats, side by side: YUYV (uncompressed) and MJPG (the camera's own JPEGs), 2592 x
1944, one frame and five averaged of each, written as PNG (lossless) to OUT. The arm is not touched.

    ../.venv/bin/python agenttools/live-res/formats.py /tmp/formats
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "calibration"))
from camera.devices import find_camera  # noqa: E402
from camera.feed import device_index  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)


def grab(fourcc: str, settle_s: float, count: int):
    camera = cv2.VideoCapture(device_index(find_camera()), cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    got = int(camera.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode()
    started = time.monotonic()
    while time.monotonic() - started < settle_s:
        camera.read()
    frames, times = [], []
    for _ in range(count):
        ok, frame = camera.read()
        times.append(time.monotonic())
        frames.append(frame)
    camera.release()
    rate = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0
    print(f"{fourcc} (camera says {got}): {frames[0].shape[1]}x{frames[0].shape[0]}, settled {settle_s:.0f} s, "
          f"{rate:.1f} frames/s")
    average = (sum(frame.astype(np.float64) for frame in frames) / len(frames)).round().astype(np.uint8)
    return frames[0], average


for name, fourcc, settle in (("yuyv", "YUYV", 6.0), ("mjpg", "MJPG", 2.0)):
    single, average = grab(fourcc, settle, 5)
    cv2.imwrite(str(out / f"{name}-single.png"), single)
    cv2.imwrite(str(out / f"{name}-avg5.png"), average)
print("written:", ", ".join(sorted(path.name for path in out.glob("*.png"))))
