#!/usr/bin/env python3
"""A camera that is one still frame: for the pages and the app with no camera plugged in.

**It is just fake**, like `fake_arm.py`. The frame never changes, but for a
clock drawn on it so the page can be seen to be live. Use it to check what the
app does with frames -- streams them, averages a snapshot, writes the files --
never to learn anything about a camera.

    with fake_camera(frame) as fake:     # every Feed opened inside sees `frame`, as MJPG
        ...
        fake.opened                      # how often the camera's pipeline was built

The feed's pipeline starts from the frame as a JPEG file, frozen into a live
picture at 10 a second with a clock over it, instead of the camera (`feed.camera_source`).
"""

import contextlib
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path[:0] = [str(SRC), str(SRC / "calibration")]

from applib.camera import feed as feed_module                      # noqa: E402
from applib.camera.devices import FPS, QUALITY, Mode              # noqa: E402

FAKE_DEVICE = "/dev/video-fake"
FAKE_NAME = "fake camera (a still frame)"
FAKE_FPS = 10


class FakeCamera:
    def __init__(self, still: Path) -> None:
        self.still = still
        self.opened = 0

    def source(self, device: str, mode: Mode) -> str:
        self.opened += 1
        return (f"filesrc name=camera location={self.still} ! jpegdec ! imagefreeze is-live=true ! "
                f"video/x-raw,framerate={FAKE_FPS}/1 ! clockoverlay halignment=left valignment=bottom ! "
                f"jpegenc quality=95 ! image/jpeg,width={mode.width},height={mode.height}")


class NoControls:
    """A camera with no focus, zoom, pan or tilt."""

    def __init__(self, device: str) -> None:
        pass

    def close(self) -> None:
        pass

    def get(self, control: int) -> None:
        return None

    def set(self, control: int, value: int) -> bool:
        return False

    def least(self, control: int) -> None:
        return None


@contextlib.contextmanager
def fake_camera(frame: np.ndarray):
    """Every `Feed` used inside this sees `frame`, clock and all."""
    with tempfile.TemporaryDirectory() as folder:
        still = Path(folder) / "still.jpg"
        cv2.imwrite(str(still), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        camera = FakeCamera(still)
        mode = Mode("MJPG", frame.shape[1], frame.shape[0], Fraction(FAKE_FPS))
        patches = {"camera_source": camera.source, "Controls": NoControls,
                   "view_modes": lambda device: {QUALITY: mode, FPS: mode}, "find_camera": lambda: FAKE_DEVICE,
                   "describe": lambda device: FAKE_NAME, "query": lambda device: None}
        saved = {name: getattr(feed_module, name) for name in patches}
        for name, value in patches.items():
            setattr(feed_module, name, value)
        try:
            yield camera
        finally:
            for name, value in saved.items():
                setattr(feed_module, name, value)
