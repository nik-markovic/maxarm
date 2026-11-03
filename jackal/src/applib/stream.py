"""The stream: the camera's live view with the overlay drawn in, H.264, for whoever watches over WebRTC.

    stream = Stream(overlay)
    feed = Feed(None, CONFIG_DIR, stream=stream)   # the stream is a branch of the camera's pipeline
    stream.connect_encoded(streamer.on_encoded)    # each H.264 access unit, on GStreamer's thread
    stream.set_watched(True)                       # a viewer came: frames flow, a keyframe first
    stream.request_keyframe()

    tee -> valve -> [jpegdec] -> scale -> overlay -+-> hardware H.264 ------------> access units
                                                   +-> I420 -> x264 (PyAV) -------> access units

The branch is closed at its valve while nobody watches, so nothing is decoded,
scaled or encoded for nobody; its sink does not hold the pipeline up for a first
frame (`async=false`), which a closed valve never lets through. The overlay is the `overlay.stream` layer, laid
over each frame by GStreamer's `overlaycomposition`; Python redraws it only when
what it shows changes. H.264 comes from the board's hardware encoder when
GStreamer has one (`v4l2h264enc`, the i.MX95's, as the property-alarm reference
uses it); else libx264 in PyAV, at its fastest preset, once per frame whoever
watches. Constrained Baseline, every viewer's decoder takes it, with a keyframe
every two seconds and whenever a viewer joins or falls behind.

The stream's size is the camera's picture made as large as H.264 level 3.1
holds, 3600 macroblocks, 1280 x 720 at 16:9: a viewer's decoder may be held to
that level.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Callable, Optional, Tuple

import av
import gi
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo  # noqa: E402  (gi wants the version first)

from .camera.devices import Mode  # noqa: E402
from .camera.feed import VIEW_FPS  # noqa: E402
from .overlay.canvas import Layer  # noqa: E402
from .overlay.stream import StreamOverlay  # noqa: E402

MAX_MACROBLOCKS = 3600
KEYFRAME_S = 2
BITRATE_BPS = 1_500_000
# x264's frames held to this much of the bitrate, by its VBV buffer: unheld, a keyframe takes over a second on the
# wire, in more RTP packets than aiortc's own receiver holds for a frame (STATUS-jackal §9).
FRAME_MOST_S = 0.4
HARDWARE_ENCODER = "v4l2h264enc"
# The hardware encoder's controls: profile 4 is Constrained Baseline.
HARDWARE_CONTROLS = "controls,h264_profile=4,video_bitrate={bitrate},h264_i_frame_period={period}"


def stream_size(frame_size: Tuple[int, int]) -> Tuple[int, int]:
    """The camera's picture as large as fits in `MAX_MACROBLOCKS`: its shape kept, the width a multiple of 16, the
    height even."""
    width, height = frame_size
    for columns in range(min(width, 4096) // 16, 0, -1):
        size = (16 * columns, 2 * round(8 * columns * height / width))
        if size[0] <= width and columns * -(-size[1] // 16) <= MAX_MACROBLOCKS:
            return size
    return 16, 16


def has_hardware_encoder() -> bool:
    return all(Gst.ElementFactory.find(name) for name in (HARDWARE_ENCODER, "h264parse"))


class Stream:
    def __init__(self, overlay: StreamOverlay) -> None:
        self.overlay = overlay
        self.size = overlay.size
        self.is_hardware = has_hardware_encoder()
        self.on_encoded: Callable[[bytes], None] = lambda unit: None
        self.is_watched = False
        self.is_keyframe_wanted = True
        self.valve = None
        self.encoder = None                     # the hardware encoder, while the pipeline runs
        self.software = None if self.is_hardware else SoftwareEncoder(self.size)
        self.layer: Optional[Layer] = None      # the overlay as GStreamer last drew it
        self.composition = None

    def connect_encoded(self, on_encoded: Callable[[bytes], None]) -> None:
        self.on_encoded = on_encoded

    def set_watched(self, is_watched: bool) -> None:
        """Frames through the branch only while someone watches; a keyframe first, for the newcomer."""
        self.is_watched = is_watched
        if is_watched:
            self.request_keyframe()
        if self.valve is not None:
            self.valve.set_property("drop", not is_watched)

    def request_keyframe(self) -> None:
        self.is_keyframe_wanted = True
        if self.encoder is not None:
            self.encoder.send_event(GstVideo.video_event_new_upstream_force_key_unit(Gst.CLOCK_TIME_NONE, True, 0))

    # --- the camera's pipeline, `feed.py` --------------------------------------------------------------------

    def branch(self, mode: Mode) -> str:
        """The stream's part of the pipeline, from the camera's tee."""
        width, height = self.size
        start = (f"valve name=stream_valve drop={str(not self.is_watched).lower()} ! "
                 + ("jpegdec ! " if mode.fourcc == "MJPG" else "")
                 + "videoscale add-borders=false ! videoconvert ! ")
        if self.is_hardware:
            controls = HARDWARE_CONTROLS.format(bitrate=BITRATE_BPS, period=KEYFRAME_S * VIEW_FPS)
            return (f"{start}video/x-raw,width={width},height={height} ! overlaycomposition name=stream_overlay ! "
                    f"{HARDWARE_ENCODER} name=stream_encoder extra-controls=\"{controls}\" ! "
                    "h264parse config-interval=-1 ! video/x-h264,stream-format=byte-stream,alignment=au ! "
                    "appsink name=stream_sink emit-signals=true max-buffers=2 drop=true sync=false async=false")
        return (f"{start}video/x-raw,format=I420,width={width},height={height} ! "
                "overlaycomposition name=stream_overlay ! "
                "appsink name=stream_sink emit-signals=true max-buffers=1 drop=true sync=false async=false")

    def attach(self, pipeline) -> None:
        """Joined to a pipeline built with `branch()`, before it plays."""
        self.valve = pipeline.get_by_name("stream_valve")
        self.encoder = pipeline.get_by_name("stream_encoder")
        pipeline.get_by_name("stream_overlay").connect("draw", self._draw)
        pipeline.get_by_name("stream_sink").connect("new-sample", self._on_sample)
        self.is_keyframe_wanted = True          # a new encoder starts on one anyway; x264's goes on from the last

    def detach(self) -> None:
        self.valve = self.encoder = None

    # --- on GStreamer's threads ------------------------------------------------------------------------------

    def _draw(self, element, sample):
        """The overlay for this frame: made again only when the layer was drawn again."""
        layer = self.overlay.current()
        if layer is not self.layer:
            self.layer, self.composition = layer, composition(layer)
        return self.composition

    def _on_sample(self, sink) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR
        buffer = sample.get_buffer()
        data = buffer.extract_dup(0, buffer.get_size())
        if self.software is not None:
            is_keyframe, self.is_keyframe_wanted = self.is_keyframe_wanted, False
            data = self.software.encode(data, is_keyframe)
        if data:
            self.on_encoded(data)
        return Gst.FlowReturn.OK


class SoftwareEncoder:
    """libx264 in PyAV, I420 in, one access unit out per frame: SPS and PPS before every keyframe, so a viewer
    joining later can start at the next one."""

    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size
        self.count = 0
        self.codec = av.CodecContext.create("libx264", "w")
        self.codec.width, self.codec.height = size
        self.codec.pix_fmt = "yuv420p"
        self.codec.framerate = Fraction(VIEW_FPS, 1)
        self.codec.time_base = Fraction(1, VIEW_FPS)
        self.codec.bit_rate = BITRATE_BPS
        self.codec.gop_size = KEYFRAME_S * VIEW_FPS
        self.codec.profile = "Baseline"
        self.codec.options = {"preset": "ultrafast", "tune": "zerolatency", "level": "31",
                              "maxrate": str(BITRATE_BPS), "bufsize": str(int(BITRATE_BPS * FRAME_MOST_S))}

    def encode(self, i420: bytes, is_keyframe: bool) -> bytes:
        width, height = self.size
        planes = np.frombuffer(i420, np.uint8)
        if planes.size != width * height * 3 // 2:
            return b""
        frame = av.VideoFrame.from_ndarray(planes.reshape(height * 3 // 2, width), format="yuv420p")
        frame.pts = self.count
        self.count += 1
        frame.pict_type = av.video.frame.PictureType.I if is_keyframe else av.video.frame.PictureType.NONE
        return b"".join(bytes(packet) for packet in self.codec.encode(frame))


def composition(layer: Layer):
    """The layer as GStreamer lays it over a frame: its drawn box only, BGRA, straight alpha; None for an empty one."""
    if layer.box is None:
        return None
    rows, columns = layer.box
    patch = np.ascontiguousarray(layer.image[layer.box])
    height, width = patch.shape[:2]
    buffer = Gst.Buffer.new_wrapped(patch.tobytes())
    GstVideo.buffer_add_video_meta(buffer, GstVideo.VideoFrameFlags.NONE, GstVideo.VideoFormat.BGRA, width, height)
    rectangle = GstVideo.VideoOverlayRectangle.new_raw(buffer, columns.start, rows.start, width, height,
                                                       GstVideo.VideoOverlayFormatFlags.NONE)
    return GstVideo.VideoOverlayComposition.new(rectangle)
