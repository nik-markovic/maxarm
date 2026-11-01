"""Which camera to use: the owner's if it is plugged in, else the first real one; and which of its modes.

Every /dev/videoN is asked what it is (V4L2's QUERYCAP). Nodes that cannot
capture video -- a USB camera's metadata node, say -- are skipped, and so are
virtual cameras: OBS's v4l2loopback, or anything calling itself virtual.

The camera is asked its formats and sizes (ENUM_FMT, ENUM_FRAMESIZES). Two of
them can be the live view: QUALITY, its largest YUYV, uncompressed, and FPS, its
largest MJPG, faster. The choice is there only when the YUYV is at least as
large as the MJPG; otherwise the MJPG is both.
"""

import fcntl
import glob
import os
import re
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# The owner's document camera, by the USB port it hangs off (PILOT-gecko).
PREFERRED_BUS = "usb-0000:0b:00.0-3.1"

_VIDIOC_QUERYCAP = 0x80685600            # _IOR('V', 0, struct v4l2_capability), 104 bytes
_CAPABILITY_SIZE = 104
_CAP_VIDEO_CAPTURE = 0x00000001
_CAP_DEVICE_CAPS = 0x80000000            # device_caps is filled in
_VIRTUAL_DRIVERS = ("v4l2 loopback",)
_VIDIOC_ENUM_FMT = 0xC0405602            # _IOWR('V', 2, struct v4l2_fmtdesc), 64 bytes
_VIDIOC_ENUM_FRAMESIZES = 0xC02C564A     # _IOWR('V', 74, struct v4l2_frmsizeenum), 44 bytes
_BUF_TYPE_VIDEO_CAPTURE = 1
_DISCRETE = 1
VIEW_FORMATS = ("YUYV", "MJPG")          # what the live view can take: raw, or the camera's own JPEGs
QUALITY, FPS = "quality", "fps"


@dataclass(frozen=True)
class CameraNode:
    path: str             # /dev/videoN
    driver: str           # uvcvideo, v4l2 loopback, ...
    card: str             # what the camera calls itself
    bus: str              # where it is plugged in: usb-0000:0b:00.0-3.1, platform:...
    is_capture: bool      # gives video frames, as opposed to metadata or output

    @property
    def is_virtual(self) -> bool:
        return (self.driver in _VIRTUAL_DRIVERS or "loopback" in self.bus
                or "virtual" in self.card.lower())

    def describe(self) -> str:
        return f"{self.card} on {self.path}"


@dataclass(frozen=True)
class Mode:
    fourcc: str           # YUYV or MJPG
    width: int
    height: int

    def describe(self) -> str:
        return f"{self.width} x {self.height} {self.fourcc}"

    def to_json(self) -> dict:
        return {"format": self.fourcc, "size": [self.width, self.height]}


def view_modes(device: str) -> Dict[str, Mode]:
    """{QUALITY: ..., FPS: ...}, the same mode for both when there is no choice. Raises if the camera has neither
    YUYV nor MJPG."""
    largest = {}
    for mode in list_modes(device):
        best = largest.get(mode.fourcc)
        if best is None or mode.width * mode.height > best.width * best.height:
            largest[mode.fourcc] = mode
    raw, jpeg = largest.get("YUYV"), largest.get("MJPG")
    if not raw and not jpeg:
        raise RuntimeError(f"{describe(device)} offers neither YUYV nor MJPG")
    if not jpeg or (raw and raw.width * raw.height >= jpeg.width * jpeg.height):
        return {QUALITY: raw, FPS: jpeg or raw}
    return {QUALITY: jpeg, FPS: jpeg}


def list_modes(device: str) -> List[Mode]:
    """Each YUYV and MJPG size the camera gives; a stepwise size at its largest."""
    modes = []
    handle = os.open(device, os.O_RDWR | os.O_NONBLOCK)
    try:
        for fourcc in _formats(handle):
            if fourcc not in VIEW_FORMATS:
                continue
            for width, height in _sizes(handle, fourcc):
                modes.append(Mode(fourcc, width, height))
    finally:
        os.close(handle)
    return modes


def _formats(handle: int) -> List[str]:
    found = []
    for index in range(64):
        reply = bytearray(struct.pack("<II", index, _BUF_TYPE_VIDEO_CAPTURE) + bytes(56))
        if not _asked(handle, _VIDIOC_ENUM_FMT, reply):
            break
        found.append(_fourcc(struct.unpack_from("<I", reply, 44)[0]))
    return found


def _sizes(handle: int, fourcc: str) -> List[Tuple[int, int]]:
    found = []
    for index in range(256):
        reply = bytearray(struct.pack("<II", index, _code(fourcc)) + bytes(36))
        if not _asked(handle, _VIDIOC_ENUM_FRAMESIZES, reply):
            break
        kind, *numbers = struct.unpack_from("<I6I", reply, 8)
        found.append((numbers[0], numbers[1]) if kind == _DISCRETE else (numbers[1], numbers[4]))
        if kind != _DISCRETE:
            break
    return found


def _asked(handle: int, request: int, reply: bytearray) -> bool:
    try:
        fcntl.ioctl(handle, request, reply)
        return True
    except OSError:
        return False


def _code(fourcc: str) -> int:
    return struct.unpack("<I", fourcc.encode())[0]


def _fourcc(code: int) -> str:
    return struct.pack("<I", code).decode("ascii", "replace")


def find_camera() -> str:
    """The /dev/videoN to use. Raises, saying what is plugged in, if there is none."""
    nodes = list_nodes()
    cameras = [node for node in nodes if node.is_capture and not node.is_virtual]
    if not cameras:
        seen = ", ".join(f"{node.path} ({node.card})" for node in nodes) or "no /dev/video* at all"
        raise RuntimeError(f"no camera found: {seen}")
    return min(cameras, key=lambda node: (node.bus != PREFERRED_BUS, not node.bus.startswith("usb-"),
                                          _number(node.path))).path


def describe(device: str) -> str:
    """'USB Camera: USB Camera on /dev/video1', or the path alone if it will not say."""
    node = query(device)
    return node.describe() if node else device


def list_nodes() -> List[CameraNode]:
    nodes = (query(path) for path in sorted(glob.glob("/dev/video*"), key=_number))
    return [node for node in nodes if node is not None]


def query(path: str) -> Optional[CameraNode]:
    """What the node says it is; None if it cannot be opened or asked. Opening it starts nothing."""
    try:
        handle = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        reply = bytearray(_CAPABILITY_SIZE)
        fcntl.ioctl(handle, _VIDIOC_QUERYCAP, reply)
    except OSError:
        return None
    finally:
        os.close(handle)
    driver, card, bus = (_text(reply[start:end]) for start, end in ((0, 16), (16, 48), (48, 80)))
    capabilities, device_caps = struct.unpack_from("<II", reply, 84)
    caps = device_caps if capabilities & _CAP_DEVICE_CAPS else capabilities
    return CameraNode(path, driver, card, bus, bool(caps & _CAP_VIDEO_CAPTURE))


def _text(field: bytes) -> str:
    return bytes(field).split(b"\0", 1)[0].decode("utf-8", "replace")


def _number(path: str) -> int:
    digits = re.findall(r"\d+", path)
    return int(digits[-1]) if digits else -1
