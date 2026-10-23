"""The live view: the camera at a small size, JPEG after JPEG, for aiming the camera by.

    feed = Feed()                  # the camera is found when a page first watches
    feed.start()
    for jpeg in feed.frames(): ... # what the page's <img> is sent, for as long as it watches
    frame = feed.still()           # a frame of the view, as the camera gave it, taken after asking
    frame, described = feed.snapshot()   # the still frame a calibration takes, full size
    feed.close()

One thread owns the camera for the view. It opens it when the first page
starts watching and closes it when the last one goes; every page gets the
newest frame and skips any it was too slow for. A snapshot needs the camera
to itself at its largest size (`capture.py`), so the view pauses for it and
picks up again after. A camera that will not open, or stops giving frames, is
tried again every few seconds while a page watches; `state()` says why.
"""

import threading
import time
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from .capture import capture, device_index, request_mode
from .devices import describe, find_camera

# What the view asks the camera for -- it picks its nearest -- and what it sends on:
# enough to aim the camera by, and light on the board and the network.
VIEW_FORMAT = "MJPG"
VIEW_SIZE = (800, 600)
VIEW_FPS = 10.0
VIEW_QUALITY = 70
RETRY_S = 2.0
# still() hands over a frame that came this many after it was asked for: the newest one may have been taken
# before the arm stopped. More if the view had to start for it, so that the camera's exposure settles first.
FRESH_FRAMES = 3
STARTING_FRAMES = 15
STILL_WAIT_S = 15.0


def open_view(device: str):
    """The camera, asked for the view's size: a `cv2.VideoCapture`."""
    camera = cv2.VideoCapture(device_index(device), cv2.CAP_V4L2)
    if not camera.isOpened():
        raise RuntimeError(f"cannot open {describe(device)}: is another program using it?")
    request_mode(camera, VIEW_FORMAT, *VIEW_SIZE)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)            # the newest frame, not one queued earlier
    return camera


class Feed:
    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device                # None: found afresh each time the camera is opened
        self.condition = threading.Condition()
        self.viewers = 0
        self.jpeg: Optional[bytes] = None   # the newest frame while the view runs
        self.image: Optional[np.ndarray] = None    # the same, as the camera gave it
        self.size: Optional[Tuple[int, int]] = None   # the view's frame size, once a frame has come
        self.count = 0                      # frames so far: a viewer waits for the next number
        self.is_streaming = False           # the view has the camera open
        self.is_paused = False              # a snapshot has the camera
        self.is_closed = False
        self.status = {"status": "idle", "detail": ""}
        self.thread = threading.Thread(target=self._run, name="camera", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        with self.condition:
            self.is_closed = True
            self.condition.notify_all()
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)

    def frames(self) -> Iterator[bytes]:
        """Every new frame as a JPEG, for one page, until it stops watching."""
        with self.condition:
            self.viewers += 1
            self.condition.notify_all()
        try:
            seen = -1
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.is_closed or (self.jpeg and self.count != seen))
                    if self.is_closed:
                        return
                    seen, jpeg = self.count, self.jpeg
                yield jpeg
        finally:
            with self.condition:
                self.viewers -= 1
                self.condition.notify_all()

    def still(self) -> np.ndarray:
        """A frame of the view as the camera gave it, taken after this was asked. The view runs for it if no page
        watches."""
        with self.condition:
            wanted = self.count + (FRESH_FRAMES if self.is_streaming else STARTING_FRAMES)
            self.viewers += 1
            self.condition.notify_all()
            try:
                self.condition.wait_for(lambda: self.is_closed or self.count >= wanted, timeout=STILL_WAIT_S)
                image = self.image if self.count >= wanted else None
            finally:
                self.viewers -= 1
                self.condition.notify_all()
        if image is None:
            raise RuntimeError(f"the camera gave no frame: {self.status['detail'] or self.status['status']}")
        return image

    def snapshot(self) -> Tuple[np.ndarray, str]:
        """The still frame, at the camera's largest size, and a line saying how it was taken."""
        with self.condition:
            self.is_paused = True
            self.condition.notify_all()
            self.condition.wait_for(lambda: not self.is_streaming)
            self.status = {"status": "snapshot", "detail": "taking the snapshot, at full size"}
        try:
            return capture(self.device or find_camera())
        finally:
            with self.condition:
                self.is_paused = False
                self.status = {"status": "idle", "detail": ""}
                self.condition.notify_all()

    def state(self) -> dict:
        with self.condition:
            return {**self.status, "size": list(self.size) if self.size else None}

    # --- the view's thread ------------------------------------------------

    def _run(self) -> None:
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.is_closed or (self.viewers and not self.is_paused))
                if self.is_closed:
                    return
            try:
                self._stream()
            except Exception as error:               # will not open, or stopped: the page says why
                with self.condition:
                    self.status = {"status": "failed", "detail": str(error)}
                    self.condition.wait_for(lambda: self.is_closed or self.is_paused, timeout=RETRY_S)

    def _stream(self) -> None:
        device = self.device or find_camera()
        name = describe(device)
        camera = open_view(device)
        with self.condition:
            self.is_streaming = True
            self.status = {"status": "starting", "detail": name}
        try:
            sent = 0.0
            while self._is_wanted():
                if not camera.grab():
                    raise RuntimeError(f"{name} stopped giving frames")
                if time.monotonic() - sent < 1.0 / VIEW_FPS:
                    continue
                is_read, image = camera.retrieve()
                if not is_read:
                    raise RuntimeError(f"{name} gave a frame that would not decode")
                jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, VIEW_QUALITY])[1].tobytes()
                sent = time.monotonic()
                with self.condition:
                    self.jpeg, self.image, self.count = jpeg, image, self.count + 1
                    self.size = (image.shape[1], image.shape[0])
                    self.status = {"status": "live", "detail": f"{name}, {image.shape[1]}x{image.shape[0]}"}
                    self.condition.notify_all()
        finally:
            camera.release()
            with self.condition:
                self.is_streaming, self.jpeg, self.image = False, None, None
                if self.status["status"] in ("starting", "live"):
                    self.status = {"status": "idle", "detail": ""}
                self.condition.notify_all()

    def _is_wanted(self) -> bool:
        with self.condition:
            return bool(self.viewers) and not self.is_paused and not self.is_closed
