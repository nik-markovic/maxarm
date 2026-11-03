"""The stream: the camera's live view with the overlay drawn in, frame after frame, for whoever watches over WebRTC.

    stream = Stream(feed, overlay, streamer.on_frame, streamer.is_watched)
    stream.start()
    ...
    stream.stop()

One thread. It keeps the camera open for the whole run -- the live view opens
the camera only while someone takes its frames, and this always does -- so a
command's snapshot never waits for the camera to start and focus. While someone
watches, each new frame is decoded at the stream's size, the overlay is burnt in
(`overlay.stream`), and the frame is handed on as BGR; nobody watching, it is
dropped as it comes.

The stream's size is the camera's picture made as large as H.264 level 3.1
holds, 3600 macroblocks, 1280 x 720 at 16:9: aiortc's encoder is held to that
level, and a viewer's decoder may hold it to it too.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from .camera.feed import Feed
from .overlay.stream import StreamOverlay

MAX_MACROBLOCKS = 3600
# The JPEG decoder's own reductions: a frame decoded at a half, a quarter or an eighth of its size, much faster.
REDUCED = ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4), (2, cv2.IMREAD_REDUCED_COLOR_2))


def stream_size(frame_size: Tuple[int, int]) -> Tuple[int, int]:
    """The camera's picture as large as fits in `MAX_MACROBLOCKS`: its shape kept, the width a multiple of 16, the
    height even."""
    width, height = frame_size
    for columns in range(min(width, 4096) // 16, 0, -1):
        size = (16 * columns, 2 * round(8 * columns * height / width))
        if size[0] <= width and columns * -(-size[1] // 16) <= MAX_MACROBLOCKS:
            return size
    return 16, 16


class Stream:
    def __init__(self, feed: Feed, overlay: StreamOverlay, on_frame: Callable[[np.ndarray], None],
                 is_watched: Callable[[], bool]) -> None:
        self.feed = feed
        self.overlay = overlay
        self.on_frame = on_frame
        self.is_watched = is_watched
        self.size = overlay.size
        self.is_stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name="stream", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        """Stop taking frames; the camera is let go once the live view is closed too (`Feed.close()`)."""
        self.is_stopping.set()

    def join(self, timeout: float) -> None:
        self.thread.join(timeout)

    def _run(self) -> None:
        while not self.is_stopping.is_set():
            frames = self.feed.frames()
            try:
                for jpeg in frames:
                    if self.is_stopping.is_set():
                        return
                    frame = self._decode(jpeg) if self.is_watched() else None
                    if frame is not None:
                        self.overlay.burn(frame)
                        self.on_frame(frame)
            finally:
                frames.close()
            self.is_stopping.wait(1.0)          # the live view closed under us: it comes back with the camera

    def _decode(self, jpeg: bytes) -> Optional[np.ndarray]:
        """A frame at the stream's size: decoded as small as still covers it, then made exactly that. None for one
        that will not decode: skipped."""
        data = np.frombuffer(jpeg, np.uint8)
        width = self.feed.state()["size"][0]
        flag = next((flag for share, flag in REDUCED if width / share >= self.size[0]), cv2.IMREAD_COLOR)
        frame = cv2.imdecode(data, flag)
        if frame is None:
            return None
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        return frame
