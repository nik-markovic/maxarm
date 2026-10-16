"""One still frame from the camera, at the largest size it will actually hand over.

Lifted from `condor/calibration/calibrate.py` unchanged in behaviour: both
formats are tried and the bigger frame wins, the camera's auto-exposure is let
settle, and several frames of the still scene are averaged. The frame size is
judged by what comes back, never by what the device agreed to.
"""

import time
from typing import Tuple

import cv2
import numpy as np

DEFAULT_DEVICE = "/dev/video1"

CAPTURE_FORMATS = ("YUYV", "MJPG")
LARGER_THAN_ANY_SENSOR = 10000
SMALLEST_USEFUL_FRAME = 640 * 480
SETTLE_FRAMES = 8
AVERAGED_FRAMES = 5


def capture(device: str = DEFAULT_DEVICE) -> Tuple[np.ndarray, str]:
    camera = cv2.VideoCapture(_device_index(device), cv2.CAP_V4L2)
    if not camera.isOpened():
        raise RuntimeError(f"cannot open {device}; `v4l2-ctl --list-devices` says which is which")
    try:
        started = time.monotonic()
        fourcc, width, height = _largest_frame(camera)
        if width * height < SMALLEST_USEFUL_FRAME:
            raise RuntimeError(f"{device} offers nothing bigger than {width}x{height}")
        _request(camera, fourcc, width, height)
        for _ in range(SETTLE_FRAMES):
            camera.read()
        image = _average_frames(camera)
        elapsed = time.monotonic() - started
        return image, f"{device} ({width}x{height} {fourcc}, {AVERAGED_FRAMES} averaged, {elapsed:.1f} s)"
    finally:
        camera.release()


def _average_frames(camera: cv2.VideoCapture) -> np.ndarray:
    total = None
    for index in range(AVERAGED_FRAMES):
        is_read, frame = camera.read()
        if not is_read:
            raise RuntimeError(f"the camera stopped after {index} frames")
        total = frame.astype(np.float64) if total is None else total + frame
    return (total / AVERAGED_FRAMES).round().astype(np.uint8)


def _largest_frame(camera: cv2.VideoCapture) -> Tuple[str, int, int]:
    best = ("", 0, 0)
    for fourcc in CAPTURE_FORMATS:
        _request(camera, fourcc, LARGER_THAN_ANY_SENSOR, LARGER_THAN_ANY_SENSOR)
        is_read, frame = camera.read()
        if is_read and frame.shape[1] * frame.shape[0] > best[1] * best[2]:
            best = (fourcc, frame.shape[1], frame.shape[0])
    return best


def _request(camera: cv2.VideoCapture, fourcc: str, width: int, height: int) -> None:
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def _device_index(device: str) -> int:
    """OpenCV here will not open a camera by name, and with V4L2 the index is N of /dev/videoN."""
    digits = "".join(character for character in device if character.isdigit())
    if not digits:
        raise RuntimeError(f"no camera number in {device!r}; give /dev/videoN or N")
    return int(digits)
