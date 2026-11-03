"""The live view: the camera at its full size, JPEG after JPEG, and the frames the calibrations take.

    feed = Feed(config_dir=CONFIG) # the camera is found when a page first watches; its mode and focus from config/
    feed.start()
    for jpeg in feed.frames(): ... # what the page's <img> is sent, for as long as it watches
    frame = feed.still()           # a frame of the view, taken after asking
    frame, described = feed.snapshot()   # the still frame a calibration takes: fresh frames averaged
    feed.reopen()                  # the camera opened again: after the camera page changes the mode
    feed.close()

One thread owns the camera. It opens it when the first page starts watching
and closes it when the last one goes; every page gets the newest frame and
skips any it was too slow for. The camera is opened in the mode the camera
page chose, QUALITY or FPS (`devices.view_modes()`, `camera-view.json`):
YUYV frames are kept as they come and made JPEGs for the pages; MJPG frames,
the camera's own JPEGs, are passed on and decoded only when a still or a
snapshot wants them; one the camera cut short, with no end, is skipped. Zoom, pan and tilt are put to the widest, straight view. The focus is held still: a
camera focusing moves its picture, the C920's by up to 4%, desk points up to
17 mm (STATUS-impala §11). It is the focus `camera.json` was calibrated at, for
the same camera; with none, the camera focuses itself once, watched until its
picture keeps still, and is held there -- `lens` says at what, for CALIBRATE
to keep. Nothing is shown until it is held. A camera that will not open, or
stops giving frames, is tried again every few seconds while a page watches;
`state()` says why.
"""

import threading
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from . import store
from .devices import QUALITY, Mode, describe, find_camera, query, view_modes

VIEW_FPS = 5.0
# A YUYV frame made a JPEG for the pages: about 6 ms at 2304 x 1536 (STATUS-impala §11).
VIEW_QUALITY = 90
RETRY_S = 2.0
# A still or a snapshot is made of frames that came this many after it was asked for: the newest one may have been
# taken before the arm stopped. More if the view had to start for it, so that the camera's exposure settles first.
FRESH_FRAMES = 2
STARTING_FRAMES = 10
AVERAGED_FRAMES = 5
STILL_WAIT_S = 15.0
# MJPG frames cut short in a row before the camera counts as having stopped giving frames.
CUT_FRAMES_MOST = 30
# The camera focusing itself is watched until its picture has kept still this long, at least this long after it
# starts, at most this long: then its focus is held. The C920 settled in 0.6 s from as far off as its focus goes
# (STATUS-impala §11); another camera may take longer. Still: a frame differing from the one before by no more than
# so many times the least difference seen, the camera's own noise, or by so many grey levels.
FOCUS_STILL_S = 0.5
FOCUS_LEAST_S = 1.0
FOCUS_MOST_S = 8.0
FOCUS_NOISE_TIMES = 2.0
FOCUS_STILL_GREY = 1.0


def open_view(device: str, mode: Mode, focus: Optional[int]):
    """The camera in `mode`, at its widest and straight, held at `focus` -- or, None, focusing itself: a
    `cv2.VideoCapture` handing over MJPG's JPEGs undecoded, YUYV's frames as BGR images."""
    camera = cv2.VideoCapture(device_index(device), cv2.CAP_V4L2)
    if not camera.isOpened():
        raise RuntimeError(f"cannot open {describe(device)}: is another program using it?")
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*mode.fourcc))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, mode.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, mode.height)
    camera.set(cv2.CAP_PROP_CONVERT_RGB, 0 if mode.fourcc == "MJPG" else 1)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)            # the newest frame, not one queued earlier
    camera.set(cv2.CAP_PROP_ZOOM, 0)                  # held to its least: the widest view
    camera.set(cv2.CAP_PROP_PAN, 0)
    camera.set(cv2.CAP_PROP_TILT, 0)
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 1 if focus is None else 0)
    if focus is not None:
        camera.set(cv2.CAP_PROP_FOCUS, focus)
    return camera


def settle_focus(camera, name: str, is_jpeg: bool) -> None:
    """Until the camera focusing itself has stopped moving its picture, or has had long enough."""
    started = time.monotonic()
    still_since, last, least = started, None, np.inf
    while time.monotonic() - started < FOCUS_MOST_S:
        is_read, raw = camera.read()
        if not is_read:
            raise RuntimeError(f"{name} stopped giving frames")
        if is_jpeg and not is_whole_jpeg(raw.tobytes()):
            continue
        grey = (cv2.imdecode(raw.reshape(-1), cv2.IMREAD_REDUCED_GRAYSCALE_8) if is_jpeg
                else cv2.resize(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), None, fx=1 / 8, fy=1 / 8,
                                interpolation=cv2.INTER_AREA)).astype(np.float32)
        now = time.monotonic()
        if last is not None:
            difference = float(np.mean(np.abs(grey - last)))
            least = min(least, difference)
            if difference > max(FOCUS_NOISE_TIMES * least, FOCUS_STILL_GREY):
                still_since = now
            elif now - started >= FOCUS_LEAST_S and now - still_since >= FOCUS_STILL_S:
                return
        last = grey


def hold_focus(camera) -> Optional[int]:
    """The camera's own focus stopped where it is and set to that again, so that it can be set the same later; None
    for a camera with no focus to set."""
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    focus = camera.get(cv2.CAP_PROP_FOCUS)
    if focus < 0 or not camera.set(cv2.CAP_PROP_FOCUS, focus):
        return None
    return int(round(focus))


def device_index(device: str) -> int:
    """OpenCV here will not open a camera by name, and with V4L2 the index is N of /dev/videoN."""
    digits = "".join(character for character in device if character.isdigit())
    if not digits:
        raise RuntimeError(f"no camera number in {device!r}; give /dev/videoN or N")
    return int(digits)


def is_whole_jpeg(data: bytes) -> bool:
    """A JPEG from its start marker to its end marker: a frame cut short has no end."""
    return data.startswith(b"\xff\xd8") and data.rfind(b"\xff\xd9") > 0


def decode(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


class Feed:
    def __init__(self, device: Optional[str] = None, config_dir: Optional[Path] = None) -> None:
        self.device = device                # None: found afresh each time the camera is opened
        self.config_dir = config_dir        # where camera-view.json says the mode, and camera.json the focus
        self.condition = threading.Condition()
        self.viewers = 0
        self.jpeg: Optional[bytes] = None   # the newest frame while the view runs, as the pages are sent it
        self.image: Optional[np.ndarray] = None   # the same, as the camera gave it, if not a JPEG
        self.name = ""                      # the camera the view has open
        self.modes: dict = {}               # the camera's QUALITY and FPS modes, once it has been opened
        self.mode: Optional[Mode] = None    # the mode the view has open
        self.lens: Optional[dict] = None    # which camera, and the focus it is held at: camera.json keeps it
        self.size: Optional[Tuple[int, int]] = None   # the view's frame size, once a frame has come
        self.count = 0                      # frames so far: a viewer waits for the next number
        self.is_streaming = False           # the view has the camera open
        self.is_closed = False
        self.is_reopening = False           # the view stops, to open the camera again
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

    def reopen(self) -> None:
        """The camera closed and opened again, as the files in config/ now say, if a page watches."""
        with self.condition:
            self.is_reopening = True
            self.condition.notify_all()

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
        return self._fresh(1)[0]

    def snapshot(self) -> Tuple[np.ndarray, str]:
        """The still frame: fresh frames of the view averaged, and a line saying how it was taken."""
        started = time.monotonic()
        images = self._fresh(AVERAGED_FRAMES)
        image = (sum(image.astype(np.float64) for image in images) / len(images)).round().astype(np.uint8)
        fourcc = self.mode.fourcc if self.mode else "?"
        return image, (f"{self.name} ({image.shape[1]}x{image.shape[0]} {fourcc}, {len(images)} averaged, "
                       f"{time.monotonic() - started:.1f} s)")

    def state(self) -> dict:
        with self.condition:
            return {**self.status, "size": list(self.size) if self.size else None,
                    "mode": self.mode.to_json() if self.mode else None,
                    "modes": {choice: mode.to_json() for choice, mode in self.modes.items()}}

    def _fresh(self, count: int) -> List[np.ndarray]:
        """`count` frames in a row, as images, the first taken some frames after this was asked."""
        frames: List[Tuple[bytes, Optional[np.ndarray]]] = []
        with self.condition:
            wanted = self.count + (FRESH_FRAMES if self.is_streaming else STARTING_FRAMES)
            self.viewers += 1
            self.condition.notify_all()
            try:
                while len(frames) < count:
                    if not self.condition.wait_for(lambda: self.is_closed or self.count >= wanted,
                                                   timeout=STILL_WAIT_S) or not self.jpeg:
                        raise RuntimeError("the camera gave no frame: "
                                           f"{self.status['detail'] or self.status['status']}")
                    frames.append((self.jpeg, self.image))
                    wanted = self.count + 1
            finally:
                self.viewers -= 1
                self.condition.notify_all()
        return [decode(jpeg) if image is None else image for jpeg, image in frames]

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
        node = query(device)
        card = node.card if node else name
        modes = view_modes(device)
        mode = modes[store.load_view_choice(self.config_dir) if self.config_dir else QUALITY]
        saved = store.load_lens(self.config_dir) if self.config_dir else None
        focus = saved["focus"] if saved and saved["camera"] == card else None
        with self.condition:
            self.is_reopening = False
        camera = open_view(device, mode, focus)
        is_jpeg = mode.fourcc == "MJPG"
        with self.condition:
            self.is_streaming, self.name, self.modes, self.mode = True, name, modes, mode
            self.status = {"status": "starting", "detail": f"{name}, {mode.describe()}"}
        try:
            if focus is None:
                with self.condition:
                    self.status = {"status": "starting", "detail": f"{name}: focusing"}
                settle_focus(camera, name, is_jpeg)
                focus = hold_focus(camera)
            with self.condition:
                self.lens = {"camera": card, "focus": focus, "mode": mode.to_json()}
            sent, size, cut = 0.0, None, 0
            while self._is_wanted():
                if not camera.grab():
                    raise RuntimeError(f"{name} stopped giving frames")
                if time.monotonic() - sent < 1.0 / VIEW_FPS:
                    continue
                is_read, raw = camera.retrieve()
                if not is_read:
                    raise RuntimeError(f"{name} gave no {mode.fourcc} frame")
                image = None if is_jpeg else raw
                jpeg = (raw.tobytes() if is_jpeg
                        else cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, VIEW_QUALITY])[1].tobytes())
                if is_jpeg and not is_whole_jpeg(jpeg):
                    cut += 1
                    if cut >= CUT_FRAMES_MOST:
                        raise RuntimeError(f"{name} gave {cut} MJPG frames in a row cut short")
                    continue
                cut = 0
                sent = time.monotonic()
                size = size or (decode(jpeg) if is_jpeg else raw).shape[1::-1]
                with self.condition:
                    self.jpeg, self.image, self.size, self.count = jpeg, image, size, self.count + 1
                    self.status = {"status": "live", "detail": f"{name}, {size[0]}x{size[1]} {mode.fourcc}"}
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
            return bool(self.viewers) and not self.is_closed and not self.is_reopening
