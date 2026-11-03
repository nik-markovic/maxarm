"""The camera: one GStreamer pipeline, giving the live view, the frames the calibrations take, and the stream.

    feed = Feed(config_dir=CONFIG)  # the camera found when a page first watches; its mode and focus from config/
    feed.add_stream(stream)         # the app: the camera open the whole run, the stream a branch of its pipeline
    feed.start()
    for jpeg in feed.frames(): ...  # what the page's <img> is sent, for as long as it watches
    frame = feed.still()            # a frame of the view, taken after asking
    frame, described = feed.snapshot()   # the still frame a calibration takes: fresh frames averaged
    feed.reopen()                   # the camera opened again: after the camera page changes the mode
    feed.close()

    camera -> damaged frames dropped -> at most VIEW_FPS -> tee -+-> the newest frame, for Python
                                                                 +-> the stream (`stream.py`), when there is one

One thread owns the pipeline. Without a stream it runs while a page watches or
a still is wanted, and stops when the last one goes; every page gets the newest
frame and skips any it was too slow for. The camera is opened in the mode the
camera page chose, QUALITY or FPS (`devices.view_modes()`, `camera-view.json`).
Python keeps each frame as the camera gave it -- the camera's own JPEG, or YUYV
-- and makes a JPEG for the pages or an image for a still only when one is
wanted. A frame the camera's driver marks as damaged (a JPEG cut short) is
dropped as it comes.

Zoom, pan and tilt are put to the widest, straight view (`controls.py`). The
focus is held still: a camera focusing moves its picture, the C920's by up to
4%, desk points up to 17 mm (STATUS-impala §11). It is the focus `camera.json`
was calibrated at, for the same camera; with none, the camera focuses itself
once, watched until its picture keeps still, and is held there -- `lens` says
at what, for CALIBRATE to keep. Nothing is shown until it is held. A camera that
will not open, or stops giving frames, is tried again every few seconds while
it is wanted; `state()` says why.
"""

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, List, NamedTuple, Optional, Tuple

import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402  (gi wants the version first)

from . import store  # noqa: E402
from .controls import FOCUS, FOCUS_AUTO, PAN, TILT, ZOOM, Controls  # noqa: E402
from .devices import QUALITY, Mode, describe, find_camera, query, view_modes  # noqa: E402

if TYPE_CHECKING:
    from applib.stream import Stream

Gst.init(None)

VIEW_FPS = 5
# A YUYV frame made a JPEG for the pages: about 6 ms at 2304 x 1536 (STATUS-impala §11).
VIEW_QUALITY = 90
RETRY_S = 2.0
# The camera has stopped giving frames when none has come for this long; the pipeline is asked this often meanwhile.
FRAME_WAIT_S = 5.0
POLL_S = 0.5
# A still or a snapshot is made of frames that came this many after it was asked for: the newest one may have been
# taken before the arm stopped. More if the view had to start for it, so that the camera's exposure settles first.
FRESH_FRAMES = 2
STARTING_FRAMES = 10
AVERAGED_FRAMES = 5
STILL_WAIT_S = 15.0
# The camera focusing itself is watched until its picture has kept still this long, at least this long after it
# starts, at most this long: then its focus is held. The C920 settled in 0.6 s from as far off as its focus goes
# (STATUS-impala §11); another camera may take longer. Still: a frame differing from the one before by no more than
# so many times the least difference seen, the camera's own noise, or by so many grey levels.
FOCUS_STILL_S = 0.5
FOCUS_LEAST_S = 1.0
FOCUS_MOST_S = 8.0
FOCUS_NOISE_TIMES = 2.0
FOCUS_STILL_GREY = 1.0


class Raw(NamedTuple):
    """A frame as the camera gave it: its own JPEG, or YUYV."""
    data: bytes
    mode: Mode


def camera_source(device: str, mode: Mode) -> str:
    """The pipeline's start: the camera, named `camera`, in `mode`, at its fastest. The rate is pinned: left open,
    the pipeline's `videorate` has the camera itself slow down, and a slowed camera can damage most frames
    (STATUS-jackal §9)."""
    caps = (f"image/jpeg,width={mode.width},height={mode.height}" if mode.fourcc == "MJPG"
            else f"video/x-raw,format=YUY2,width={mode.width},height={mode.height}")
    if mode.fps:
        caps += f",framerate={mode.fps.numerator}/{mode.fps.denominator}"
    return f"v4l2src name=camera device={device} ! {caps}"


def describe_pipeline(source: str, stream_branch: Optional[str]) -> str:
    """Leaky one-frame queues on both branches: each takes the newest frame, and neither holds the other up."""
    pipeline = (f"{source} ! videorate drop-only=true max-rate={VIEW_FPS} ! tee name=split "
                "split. ! queue max-size-buffers=1 leaky=downstream ! "
                "appsink name=frames max-buffers=1 drop=true sync=false")
    if stream_branch:
        pipeline += f" split. ! queue max-size-buffers=1 leaky=downstream ! {stream_branch}"
    return pipeline


def point_camera(controls: Controls, focus: Optional[int]) -> None:
    """Zoom at its least, the widest view; pan and tilt straight; the focus held at `focus` or, None, the camera
    focusing itself. A control the camera does not have is left alone."""
    zoom = controls.least(ZOOM)
    if zoom is not None:
        controls.set(ZOOM, zoom)
    controls.set(PAN, 0)
    controls.set(TILT, 0)
    controls.set(FOCUS_AUTO, 1 if focus is None else 0)
    if focus is not None:
        controls.set(FOCUS, focus)


def hold_focus(controls: Controls) -> Optional[int]:
    """The camera's own focus stopped where it is and set to that again, so that it can be set the same later; None
    for a camera with no focus to set."""
    controls.set(FOCUS_AUTO, 0)
    focus = controls.get(FOCUS)
    if focus is None or not controls.set(FOCUS, focus):
        return None
    return focus


def settle_focus(next_frame: Callable[[], Raw]) -> None:
    """Until the camera focusing itself has stopped moving its picture, or has had long enough."""
    started = time.monotonic()
    still_since, last, least = started, None, np.inf
    while time.monotonic() - started < FOCUS_MOST_S:
        grey = small_grey(next_frame())
        now = time.monotonic()
        if last is not None:
            difference = float(np.mean(np.abs(grey - last)))
            least = min(least, difference)
            if difference > max(FOCUS_NOISE_TIMES * least, FOCUS_STILL_GREY):
                still_since = now
            elif now - started >= FOCUS_LEAST_S and now - still_since >= FOCUS_STILL_S:
                return
        last = grey


def small_grey(raw: Raw) -> np.ndarray:
    """The frame's lightness at an eighth of its size."""
    if raw.mode.fourcc == "MJPG":
        grey = cv2.imdecode(np.frombuffer(raw.data, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    else:
        luma = np.ascontiguousarray(_yuyv(raw)[:, :, 0])
        grey = cv2.resize(luma, None, fx=1 / 8, fy=1 / 8, interpolation=cv2.INTER_AREA)
    return grey.astype(np.float32)


def to_image(raw: Raw) -> np.ndarray:
    """The frame as BGR."""
    if raw.mode.fourcc == "MJPG":
        return decode(raw.data)
    return cv2.cvtColor(_yuyv(raw), cv2.COLOR_YUV2BGR_YUYV)


def decode(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def _yuyv(raw: Raw) -> np.ndarray:
    return np.frombuffer(raw.data, np.uint8).reshape(raw.mode.height, raw.mode.width, 2)


def _drop_damaged(pad, info) -> Gst.PadProbeReturn:
    """The driver marks a frame that did not come whole; the camera source passes the mark on."""
    is_damaged = info.get_buffer().has_flags(Gst.BufferFlags.CORRUPTED)
    return Gst.PadProbeReturn.DROP if is_damaged else Gst.PadProbeReturn.OK


def _raise_problem(pipeline, name: str) -> None:
    """What went wrong in the pipeline, if anything has, raised."""
    message = pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
    if message is None:
        return
    if message.type == Gst.MessageType.EOS:
        raise RuntimeError(f"{name} stopped giving frames")
    error, _ = message.parse_error()
    raise RuntimeError(f"{name}: {error.message}")


class Feed:
    def __init__(self, device: Optional[str] = None, config_dir: Optional[Path] = None) -> None:
        self.device = device                # None: found afresh each time the camera is opened
        self.config_dir = config_dir        # where camera-view.json says the mode, and camera.json the focus
        self.stream: "Optional[Stream]" = None   # fed from the pipeline; with it, the camera is open the whole run
        self.condition = threading.Condition()
        self.viewers = 0
        self.raw: Optional[Raw] = None      # the newest frame while the view runs
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
        self.page_jpeg: Tuple[int, bytes] = (-1, b"")   # the newest frame as the pages are sent it, made once
        self.page_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="camera", daemon=True)

    def add_stream(self, stream: "Stream") -> None:
        """Before `start()`."""
        self.stream = stream

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        with self.condition:
            self.is_closed = True
            self.condition.notify_all()
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)

    def reopen(self) -> None:
        """The camera closed and opened again, as the files in config/ now say, if it is open."""
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
                    self.condition.wait_for(lambda: self.is_closed or (self.raw and self.count != seen))
                    if self.is_closed:
                        return
                    seen, raw = self.count, self.raw
                yield self._page_jpeg(seen, raw)
        finally:
            with self.condition:
                self.viewers -= 1
                self.condition.notify_all()

    def still(self) -> np.ndarray:
        """A frame of the view, taken after this was asked. The view runs for it if nothing else keeps it open."""
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

    def _page_jpeg(self, count: int, raw: Raw) -> bytes:
        """The frame as a JPEG: the camera's own, or YUYV encoded once, however many pages watch."""
        if raw.mode.fourcc == "MJPG":
            return raw.data
        with self.page_lock:
            if self.page_jpeg[0] != count:
                jpeg = cv2.imencode(".jpg", to_image(raw), [cv2.IMWRITE_JPEG_QUALITY, VIEW_QUALITY])[1].tobytes()
                self.page_jpeg = (count, jpeg)
            return self.page_jpeg[1]

    def _fresh(self, count: int) -> List[np.ndarray]:
        """`count` frames in a row, as images, the first taken some frames after this was asked."""
        frames: List[Raw] = []
        with self.condition:
            wanted = self.count + (FRESH_FRAMES if self.is_streaming else STARTING_FRAMES)
            self.viewers += 1
            self.condition.notify_all()
            try:
                while len(frames) < count:
                    if not self.condition.wait_for(lambda: self.is_closed or self.count >= wanted,
                                                   timeout=STILL_WAIT_S) or not self.raw:
                        raise RuntimeError("the camera gave no frame: "
                                           f"{self.status['detail'] or self.status['status']}")
                    frames.append(self.raw)
                    wanted = self.count + 1
            finally:
                self.viewers -= 1
                self.condition.notify_all()
        return [to_image(raw) for raw in frames]

    # --- the view's thread ------------------------------------------------

    def _run(self) -> None:
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.is_closed or self.viewers or self.stream)
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
        controls = Controls(device)
        pipeline = None
        try:
            point_camera(controls, focus)
            pipeline = self._play(camera_source(device, mode), mode, name)
            frames = pipeline.get_by_name("frames")
            with self.condition:
                self.is_streaming, self.name, self.modes, self.mode = True, name, modes, mode
                self.status = {"status": "starting", "detail": f"{name}, {mode.describe()}"}
            if focus is None:
                with self.condition:
                    self.status = {"status": "starting", "detail": f"{name}: focusing"}
                settle_focus(lambda: self._wait_frame(pipeline, frames, name, mode))
                focus = hold_focus(controls)
            with self.condition:
                self.lens = {"camera": card, "focus": focus, "mode": mode.to_json()}
            last = time.monotonic()
            while self._is_wanted():
                raw = self._next_frame(pipeline, frames, name, mode, last)
                if raw is None:
                    continue
                last = time.monotonic()
                with self.condition:
                    self.raw, self.size, self.count = raw, (mode.width, mode.height), self.count + 1
                    self.status = {"status": "live", "detail": f"{name}, {mode.width}x{mode.height} {mode.fourcc}"}
                    self.condition.notify_all()
        finally:
            if pipeline is not None:
                pipeline.set_state(Gst.State.NULL)
                if self.stream:
                    self.stream.detach()
            controls.close()
            with self.condition:
                self.is_streaming, self.raw = False, None
                if self.status["status"] in ("starting", "live"):
                    self.status = {"status": "idle", "detail": ""}
                self.condition.notify_all()

    def _play(self, source: str, mode: Mode, name: str):
        """The pipeline built, the stream joined to it, and playing; or what stopped it, raised."""
        pipeline = Gst.parse_launch(describe_pipeline(source, self.stream.branch(mode) if self.stream else None))
        pipeline.get_by_name("camera").get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, _drop_damaged)
        if self.stream:
            self.stream.attach(pipeline)
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            try:
                _raise_problem(pipeline, f"cannot open {name}")
                raise RuntimeError(f"cannot open {name}: is another program using it?")
            finally:
                pipeline.set_state(Gst.State.NULL)
                if self.stream:
                    self.stream.detach()
        return pipeline

    def _next_frame(self, pipeline, frames, name: str, mode: Mode, last: float) -> Optional[Raw]:
        """The next frame, or None after a short wait for it; raised, what went wrong, or a camera silent since
        `last` for too long."""
        _raise_problem(pipeline, name)
        sample = frames.emit("try-pull-sample", int(POLL_S * Gst.SECOND))
        if sample is None:
            if time.monotonic() - last >= FRAME_WAIT_S:
                raise RuntimeError(f"{name} stopped giving frames")
            return None
        buffer = sample.get_buffer()
        return Raw(buffer.extract_dup(0, buffer.get_size()), mode)

    def _wait_frame(self, pipeline, frames, name: str, mode: Mode) -> Raw:
        last = time.monotonic()
        while True:
            raw = self._next_frame(pipeline, frames, name, mode, last)
            if raw is not None:
                return raw

    def _is_wanted(self) -> bool:
        with self.condition:
            return bool(self.viewers or self.stream) and not self.is_closed and not self.is_reopening
