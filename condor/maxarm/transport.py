#!/usr/bin/env python3
"""Moving bytes to the board. Nothing here knows what the bytes mean.

Only USB is implemented, and that is a firmware fact rather than a shortcut:

  * **USB** is the board's own micro-USB bridge, carrying a MicroPython REPL at
    115200. Stock firmware answers on it.
  * **UART** over the 4-pin header would carry Hiwonder's 0xAA 0x55 binary
    protocol -- except that stock firmware has no parser for it. The parser
    (`MaxArm_ctl.py`) and the five board methods it calls are simply absent, so
    using it means replacing `main.py`, `espmax.py` and `SuctionNozzle.py`.
    That is a firmware swap, and `work/GUIDELINES.md` forbids touching the
    board. Hence `NotImplementedError`, not a missing feature.
  * **BLE** is live on stock firmware but is a one-way 3 mm jog protocol with
    no readback at all -- see `work/PROTOCOL-ble.md`.

The one thing worth knowing about this transport: **opening the port resets the
ESP32.** DTR and RTS land on EN and BOOT through the USB bridge, and holding
them low before and after `open()` does not prevent it -- measured, five opens
out of five. So every connect re-runs the board's `main.py`, which homes the
arm and rotates the nozzle. There is no way to attach quietly. Budget ~3.3 s,
and about one connect in five stalls a further 10 s inside the board's BLE
init, which is why nothing here sleeps a fixed interval.
"""

import errno
import glob
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional

from .config import ArmConfig, ConnectionMethod

# Where a MaxArm shows up: the board's bridge is a CH340-class part, so it is
# always `ttyUSB`, never `ttyACM`. Only used to tell someone what *is* plugged
# in when the port they asked for is not. A Windows `COM*` probe would go here.
PORT_PATTERNS = ("/dev/ttyUSB*",)


class TransportError(RuntimeError):
    """The link could not be opened, or stopped answering."""


class Transport(ABC):
    """A byte pipe with a deadline-aware read. Framing lives in `protocol.py`."""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def write(self, payload: bytes) -> None: ...

    @abstractmethod
    def read_until(self, needle: bytes, timeout_s: float) -> bytes:
        """Read until `needle` appears or the timeout expires. Never raises."""

    @abstractmethod
    def reset_input(self) -> None: ...

    @property
    @abstractmethod
    def is_open(self) -> bool: ...


class SerialTransport(Transport):
    """pyserial against the board's USB bridge, with the reset caveat above.

    `port_factory` exists so tests can drop in a fake port and still exercise
    the real framing above it; production never passes it.
    """

    def __init__(self, device: str, baud: int, read_timeout_s: float = 1.0,
                 port_factory: Optional[Callable[[], object]] = None) -> None:
        self.device = device
        self.baud = baud
        self.read_timeout_s = read_timeout_s
        self._port_factory = port_factory
        self._port = None

    def open(self) -> None:
        if self._port is not None:
            return
        factory = self._port_factory or _load_pyserial()
        port = factory()
        port.port = self.device
        port.baudrate = self.baud
        port.timeout = self.read_timeout_s
        # Set on both sides of open(): it does not stop the reset, but asserting
        # them would hold the chip in reset or drop it into the bootloader.
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
        port = self._require()
        buffer = b""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            # `in_waiting or 1` so an empty port still blocks on one byte
            # rather than spinning: the read returns the instant one arrives.
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
        """The underlying pyserial port, or whatever `port_factory` made."""
        return self._port

    def _require(self):
        if self._port is None:
            raise TransportError("transport is not open")
        return self._port


def create_transport(config: ArmConfig,
                     port_factory: Optional[Callable[[], object]] = None) -> Transport:
    """Build the transport a resolved `ArmConfig` asks for."""
    method = config.connection or ConnectionMethod.USB
    if method is ConnectionMethod.USB:
        return SerialTransport(config.device, config.baud, port_factory=port_factory)
    if method is ConnectionMethod.UART:
        raise NotImplementedError(
            "UART needs the 0xAA 0x55 slave firmware, which is not on this board; "
            "flashing it is an owner decision -- see work/PROTOCOL-bin.md")
    if method is ConnectionMethod.BLE:
        raise NotImplementedError(
            "BLE is jog-only and has no position readback -- see work/PROTOCOL-ble.md")
    raise NotImplementedError(f"unknown connection method {method!r}")


def _why_it_would_not_open(device: str, error: Exception) -> str:
    """Name which of the three usual things went wrong, and what fixes it.

    They look the same in a traceback and have nothing in common as remedies:
    plug the arm in, join a group, or close the other program. pyserial hands
    the OS `errno` straight through on its `SerialException`, so there is
    nothing to guess at.
    """
    code = getattr(error, "errno", None)
    if code == errno.ENOENT:
        return (f"{device} is not there -- the arm is unplugged, unpowered, or it "
                f"enumerated somewhere else.{_ports_that_do_exist()}")
    if code in (errno.EACCES, errno.EPERM):
        return (f"{device} exists but this user may not open it -- it is owned by a "
                f"group you are not in, usually `dialout`. "
                f"`sudo usermod -aG dialout $USER`, then log out and back in.")
    if code == errno.EBUSY:
        return (f"{device} is already open -- a serial monitor, or an earlier run "
                f"of this that has not exited yet.")
    return f"cannot open {device}: {error}"


def _ports_that_do_exist() -> str:
    found = sorted(path for pattern in PORT_PATTERNS for path in glob.glob(pattern))
    if not found:
        return f" Nothing matches {' or '.join(PORT_PATTERNS)} either."
    return (f" What is there: {', '.join(found)}"
            f" -- point $MAXARM_DEVICE or ArmConfig(device=...) at the right one.")


def _load_pyserial() -> Callable[[], object]:
    try:
        import serial
    except ImportError as error:      # pragma: no cover - environment problem
        raise TransportError("pyserial is required for the USB transport") from error
    return serial.Serial
