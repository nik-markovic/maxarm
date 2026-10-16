#!/usr/bin/env python3
"""The serial port to the board. Nothing here knows what the bytes mean.

The board's micro-USB bridge carries a MicroPython REPL at 115200, and on stock
firmware that is the only way in: the 0xAA 0x55 binary protocol has no parser
on the board (`work/PROTOCOL-bin.md`), and BLE is a one-way jog protocol.

**Opening the port resets the ESP32.** DTR and RTS land on EN and BOOT through
the bridge, and holding them low does not prevent it -- measured, five opens
out of five. So every connect re-runs the board's `main.py`, which homes the
arm and centres the cup. Budget ~4 s, sometimes 15 s (the board's BLE init).
"""

import errno
import glob
import os
import time
from typing import Callable, Optional

DEVICE = os.environ.get("MAXARM_DEVICE", "/dev/ttyUSB0")
BAUD = 115200

# The bridge is a CH340-class part, so a MaxArm is always `ttyUSB`. Only used to
# say what *is* plugged in when the asked-for port is not there.
PORT_PATTERNS = ("/dev/ttyUSB*",)


class TransportError(RuntimeError):
    """The port could not be opened, or is not open."""


class SerialTransport:
    """pyserial on the board's USB bridge.

    `port_factory` lets the offline tests put a fake board on the end of the
    wire and still exercise the real framing; nothing else passes it.
    """

    def __init__(self, device: str = DEVICE, read_timeout_s: float = 1.0,
                 port_factory: Optional[Callable[[], object]] = None) -> None:
        self.device = device
        self.read_timeout_s = read_timeout_s
        self._port_factory = port_factory
        self._port = None

    def open(self) -> None:
        if self._port is not None:
            return
        port = (self._port_factory or _load_pyserial())()
        port.port = self.device
        port.baudrate = BAUD
        port.timeout = self.read_timeout_s
        # Both sides of open(): it does not stop the reset, but asserting them
        # would hold the chip in reset or drop it into its bootloader.
        port.dtr = False
        port.rts = False
        try:
            port.open()
        except Exception as error:
            raise TransportError(_why_it_would_not_open(self.device, error)) from error
        port.dtr = False
        port.rts = False
        self._port = port

    def close(self) -> None:
        if self._port is None:
            return
        try:
            self._port.close()
        finally:
            self._port = None

    def write(self, payload: bytes) -> None:
        port = self._require()
        port.write(payload)
        port.flush()

    def read_until(self, needle: bytes, timeout_s: float) -> bytes:
        """Read until `needle` arrives or the time is up. Never raises on a timeout."""
        port = self._require()
        buffer = b""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            # `in_waiting or 1`: an empty port blocks on one byte instead of spinning.
            pending = port.read(port.in_waiting or 1)
            if pending:
                buffer += pending
                if needle in buffer:
                    break
            else:
                time.sleep(0.02)
        return buffer

    def reset_input(self) -> None:
        self._require().reset_input_buffer()

    @property
    def is_open(self) -> bool:
        return self._port is not None

    @property
    def port(self):
        """The pyserial port, or whatever `port_factory` made."""
        return self._port

    def _require(self):
        if self._port is None:
            raise TransportError("the serial port is not open")
        return self._port


def _why_it_would_not_open(device: str, error: Exception) -> str:
    """Unplugged, wrong group, or already open: same traceback, different remedy."""
    code = getattr(error, "errno", None)
    if code == errno.ENOENT:
        return (f"{device} is not there -- the arm is unplugged, unpowered, or it "
                f"enumerated somewhere else.{_ports_that_do_exist()}")
    if code in (errno.EACCES, errno.EPERM):
        return (f"{device} exists but this user may not open it -- usually the `dialout` "
                f"group: `sudo usermod -aG dialout $USER`, then log out and back in.")
    if code == errno.EBUSY:
        return f"{device} is already open -- a serial monitor, or an earlier run still going."
    return f"cannot open {device}: {error}"


def _ports_that_do_exist() -> str:
    found = sorted(path for pattern in PORT_PATTERNS for path in glob.glob(pattern))
    if not found:
        return f" Nothing matches {' or '.join(PORT_PATTERNS)} either."
    return f" What is there: {', '.join(found)} -- point $MAXARM_DEVICE at the right one."


def _load_pyserial() -> Callable[[], object]:
    try:
        import serial
    except ImportError as error:      # pragma: no cover - environment problem
        raise TransportError("pyserial is needed to talk to the arm") from error
    return serial.Serial
