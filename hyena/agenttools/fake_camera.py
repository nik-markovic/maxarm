#!/usr/bin/env python3
"""A camera that exists only in RAM: one still frame, for the camera page with no camera plugged in.

**It is just fake**, like `fake_arm.py`. The frame never changes, but for a
clock drawn on the live view so the page can be seen to be live. Use it to
check what the app does with frames -- streams them, pauses for a snapshot,
writes the files -- never to learn anything about a camera.

    with fake_camera(frame) as fake:     # every Feed made inside sees `frame`
        ...
        fake.snapshots                   # how many snapshots were taken
        fake.opened, fake.released       # how often the live view opened and let go of the camera
        fake.open_at_snapshots           # how many views had it open as each snapshot was taken: 0
"""

import contextlib
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera import feed as feed_module                       # noqa: E402

FAKE_DEVICE = "/dev/video-fake"
FAKE_NAME = "fake camera (a still frame)"


class FakeCamera:
    """The frame, and a count of what was asked of it."""

    def __init__(self, frame: np.ndarray, fps: float = 30.0) -> None:
        self.frame = frame
        self.fps = fps
        self.snapshots = 0
        self.opened = 0
        self.released = 0
        self.open_at_snapshots: List[int] = []
        self.is_broken = False          # True: the camera will not open, as if unplugged

    def open_view(self, device: str) -> "StillCapture":
        if self.is_broken:
            raise RuntimeError(f"cannot open {FAKE_NAME}: it is unplugged")
        self.opened += 1
        return StillCapture(self, feed_module.VIEW_SIZE)

    def capture(self, device: str) -> Tuple[np.ndarray, str]:
        if self.is_broken:
            raise RuntimeError(f"cannot open {FAKE_NAME}: it is unplugged")
        self.snapshots += 1
        self.open_at_snapshots.append(self.opened - self.released)
        height, width = self.frame.shape[:2]
        return self.frame.copy(), f"{FAKE_NAME} ({width}x{height})"


class StillCapture:
    """Reads like the live view's `cv2.VideoCapture`: the frame at the view's size, at a camera's pace."""

    def __init__(self, camera: FakeCamera, size: Tuple[int, int]) -> None:
        self.camera = camera
        self.size = size

    def isOpened(self) -> bool:
        return True

    def grab(self) -> bool:
        time.sleep(1.0 / self.camera.fps)
        return not self.camera.is_broken

    def retrieve(self) -> Tuple[bool, np.ndarray]:
        image = cv2.resize(self.camera.frame, self.size, interpolation=cv2.INTER_AREA)
        cv2.putText(image, time.strftime("%H:%M:%S") + f".{int(time.time() * 10) % 10}", (12, self.size[1] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return True, image

    def read(self) -> Tuple[bool, np.ndarray]:
        return self.retrieve() if self.grab() else (False, None)

    def release(self) -> None:
        self.camera.released += 1


@contextlib.contextmanager
def fake_camera(frame: np.ndarray):
    """Every `Feed` used inside this sees `frame`: the live view scaled down, the snapshot as it is."""
    camera = FakeCamera(frame)
    patches = {"open_view": camera.open_view, "capture": camera.capture,
               "find_camera": lambda: FAKE_DEVICE, "describe": lambda device: FAKE_NAME}
    saved = {name: getattr(feed_module, name) for name in patches}
    for name, value in patches.items():
        setattr(feed_module, name, value)
    try:
        yield camera
    finally:
        for name, value in saved.items():
            setattr(feed_module, name, value)
