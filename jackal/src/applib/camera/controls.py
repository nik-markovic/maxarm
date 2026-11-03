"""The camera's own controls -- focus, zoom, pan, tilt -- read and set on its /dev/videoN while GStreamer streams.

    controls = Controls("/dev/video1")
    controls.set(FOCUS_AUTO, 0)
    focus = controls.get(FOCUS)          # None: the camera has no such control
    controls.close()

V4L2's G_CTRL, S_CTRL and QUERYCTRL, as `devices.py` asks its questions. A
camera's controls belong to the camera, not to whoever opened it, so these work
beside the pipeline's own open of the same node.
"""

import fcntl
import os
import struct
from typing import Optional

_VIDIOC_G_CTRL = 0xC008561B             # _IOWR('V', 27, struct v4l2_control), 8 bytes
_VIDIOC_S_CTRL = 0xC008561C             # _IOWR('V', 28, struct v4l2_control)
_VIDIOC_QUERYCTRL = 0xC0445624          # _IOWR('V', 36, struct v4l2_queryctrl), 68 bytes
_CAMERA_CLASS = 0x009A0900

PAN = _CAMERA_CLASS + 8
TILT = _CAMERA_CLASS + 9
FOCUS = _CAMERA_CLASS + 10
FOCUS_AUTO = _CAMERA_CLASS + 12
ZOOM = _CAMERA_CLASS + 13


class Controls:
    def __init__(self, device: str) -> None:
        self.handle = os.open(device, os.O_RDWR | os.O_NONBLOCK)

    def close(self) -> None:
        os.close(self.handle)

    def get(self, control: int) -> Optional[int]:
        reply = bytearray(struct.pack("<Ii", control, 0))
        if not self._asked(_VIDIOC_G_CTRL, reply):
            return None
        return struct.unpack("<Ii", reply)[1]

    def set(self, control: int, value: int) -> bool:
        """False for a control the camera does not have, or a value it refuses."""
        return self._asked(_VIDIOC_S_CTRL, bytearray(struct.pack("<Ii", control, int(value))))

    def least(self, control: int) -> Optional[int]:
        reply = bytearray(struct.pack("<I", control) + bytes(64))
        if not self._asked(_VIDIOC_QUERYCTRL, reply):
            return None
        return struct.unpack_from("<i", reply, 40)[0]

    def _asked(self, request: int, reply: bytearray) -> bool:
        try:
            fcntl.ioctl(self.handle, request, reply)
            return True
        except OSError:
            return False
