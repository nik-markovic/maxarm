"""The live view: the camera at its full size, its own JPEG after JPEG, and the frames the calibrations take.

    feed = Feed()                  # the camera is found when a page first watches
    feed.start()
    for jpeg in feed.frames(): ... # what the page's <img> is sent, for as long as it watches
    frame = feed.still()           # a frame of the view, taken after asking
    frame, described = feed.snapshot()   # the still frame a calibration takes: fresh frames averaged
    feed.close()

One thread owns the camera. It opens it when the first page starts watching
and closes it when the last one goes; every page gets the newest frame and
skips any it was too slow for. The camera is asked for MJPG at its largest
size, and its own JPEGs are passed on as they come: a frame is decoded only
when a still or a snapshot wants it (STATUS-impala §1: why MJPG). A camera
that will not open, or stops giving frames, is tried again every few seconds while a
page watches; `state()` says why.
"""

import threading
import time
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .devices import describe, find_camera

VIEW_FORMAT = "MJPG"
LARGER_THAN_ANY_SENSOR = 10000          # asked for, the camera gives its largest
VIEW_FPS = 5.0
RETRY_S = 2.0
# A still or a snapshot is made of frames that came this many after it was asked for: the newest one may have been
# taken before the arm stopped. More if the view had to start for it, so that the camera's exposure settles first.
FRESH_FRAMES = 2
STARTING_FRAMES = 10
AVERAGED_FRAMES = 5
STILL_WAIT_S = 15.0


def open_view(device: str):
    """The camera, asked for MJPG at its largest size, handing over its JPEGs undecoded: a `cv2.VideoCapture`."""
    camera = cv2.VideoCapture(device_index(device), cv2.CAP_V4L2)
    if not camera.isOpened():
        raise RuntimeError(f"cannot open {describe(device)}: is another program using it?")
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*VIEW_FORMAT))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, LARGER_THAN_ANY_SENSOR)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, LARGER_THAN_ANY_SENSOR)
    camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)            # the newest frame, not one queued earlier
    return camera


def device_index(device: str) -> int:
    """OpenCV here will not open a camera by name, and with V4L2 the index is N of /dev/videoN."""
    digits = "".join(character for character in device if character.isdigit())
    if not digits:
        raise RuntimeError(f"no camera number in {device!r}; give /dev/videoN or N")
    return int(digits)


def decode(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


class Feed:
    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device                # None: found afresh each time the camera is opened
        self.condition = threading.Condition()
        self.viewers = 0
        self.jpeg: Optional[bytes] = None   # the newest frame while the view runs, as the camera gave it
        self.name = ""                      # the camera the view has open
        self.size: Optional[Tuple[int, int]] = None   # the view's frame size, once a frame has come
        self.count = 0                      # frames so far: a viewer waits for the next number
        self.is_streaming = False           # the view has the camera open
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
        """A frame of the view, taken after this was asked. The view runs for it if no page watches."""
        return decode(self._fresh(1)[0])

    def snapshot(self) -> Tuple[np.ndarray, str]:
        """The still frame: fresh frames of the view averaged, and a line saying how it was taken."""
        started = time.monotonic()
        jpegs = self._fresh(AVERAGED_FRAMES)
        total = sum(decode(jpeg).astype(np.float64) for jpeg in jpegs)
        image = (total / len(jpegs)).round().astype(np.uint8)
        return image, (f"{self.name} ({image.shape[1]}x{image.shape[0]} {VIEW_FORMAT}, {len(jpegs)} averaged, "
                       f"{time.monotonic() - started:.1f} s)")

    def state(self) -> dict:
        with self.condition:
            return {**self.status, "size": list(self.size) if self.size else None}

    def _fresh(self, count: int) -> List[bytes]:
        """`count` frames in a row, the first taken some frames after this was asked."""
        jpegs: List[bytes] = []
        with self.condition:
            wanted = self.count + (FRESH_FRAMES if self.is_streaming else STARTING_FRAMES)
            self.viewers += 1
            self.condition.notify_all()
            try:
                while len(jpegs) < count:
                    if not self.condition.wait_for(lambda: self.is_closed or self.count >= wanted,
                                                   timeout=STILL_WAIT_S) or not self.jpeg:
                        raise RuntimeError("the camera gave no frame: "
                                           f"{self.status['detail'] or self.status['status']}")
                    jpegs.append(self.jpeg)
                    wanted = self.count + 1
            finally:
                self.viewers -= 1
                self.condition.notify_all()
        return jpegs

    # --- the view's thread ------------------------------------------------

    def _run(self) -> None:
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.is_closed or self.viewers)
                if self.is_closed:
                    return
            try:
                self._stream()
            except Exception as error:               # will not open, or stopped: the page says why
                with self.condition:
                    self.status = {"status": "failed", "detail": str(error)}
                    self.condition.wait_for(lambda: self.is_closed, timeout=RETRY_S)

    def _stream(self) -> None:
        device = self.device or find_camera()
        name = describe(device)
        camera = open_view(device)
        with self.condition:
            self.is_streaming, self.name = True, name
            self.status = {"status": "starting", "detail": name}
        try:
            sent, size = 0.0, None
            while self._is_wanted():
                if not camera.grab():
                    raise RuntimeError(f"{name} stopped giving frames")
                if time.monotonic() - sent < 1.0 / VIEW_FPS:
                    continue
                is_read, raw = camera.retrieve()
                jpeg = raw.tobytes() if is_read else b""
                if not jpeg.startswith(b"\xff\xd8"):
                    raise RuntimeError(f"{name} gave no {VIEW_FORMAT} frame")
                sent = time.monotonic()
                with self.condition:
                    if not size:
                        size = decode(jpeg).shape[1::-1]
                    self.jpeg, self.size, self.count = jpeg, size, self.count + 1
                    self.status = {"status": "live", "detail": f"{name}, {size[0]}x{size[1]}"}
                    self.condition.notify_all()
        finally:
            camera.release()
            with self.condition:
                self.is_streaming, self.jpeg = False, None
                if self.status["status"] in ("starting", "live"):
                    self.status = {"status": "idle", "detail": ""}
                self.condition.notify_all()

    def _is_wanted(self) -> bool:
        with self.condition:
            return bool(self.viewers) and not self.is_closed
