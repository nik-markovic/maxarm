"""Which camera to use: the owner's if it is plugged in, else the first real one.

Every /dev/videoN is asked what it is (V4L2's QUERYCAP). Nodes that cannot
capture video -- a USB camera's metadata node, say -- are skipped, and so are
virtual cameras: OBS's v4l2loopback, or anything calling itself virtual.
"""

import fcntl
import glob
import os
import re
import struct
from dataclasses import dataclass
from typing import List, Optional

# The owner's document camera, by the USB port it hangs off (PILOT-gecko).
PREFERRED_BUS = "usb-0000:0b:00.0-3.1"

_VIDIOC_QUERYCAP = 0x80685600            # _IOR('V', 0, struct v4l2_capability), 104 bytes
_CAPABILITY_SIZE = 104
_CAP_VIDEO_CAPTURE = 0x00000001
_CAP_DEVICE_CAPS = 0x80000000            # device_caps is filled in
_VIRTUAL_DRIVERS = ("v4l2 loopback",)


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
