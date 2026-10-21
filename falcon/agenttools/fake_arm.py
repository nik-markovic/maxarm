#!/usr/bin/env python3
"""A MaxArm that exists only in RAM: the board, its firmware's habits, and the serial port.

**It is just fake.** Use it to check what the library sends, in what order, and
what it waits for. Never draw conclusions from it about how the arm behaves --
where it goes, how it lands, how long it takes: the real arm often does not
behave the same.

It replaces pyserial and nothing else, so the transport, the raw-REPL framing,
the result marker and the board's stdout chatter are all exercised for real.
What it reproduces is how this board misleads its driver:

  1. `set_position()` returns None inside the 50 mm blind cylinder,
  2. False when the IK will not solve -- printing to stdout first,
  3. True while servo 2 sits silently clamped and the arm does not move,
  4. Ctrl-D on an empty buffer is a soft reboot that re-homes the arm,
  5. and, when asked (`deadband_mm`, `stop_short_mm`), the servos' deadband: a
     target closer than the deadband is ignored, a farther one stopped short of.

Reachability uses the library's `geometry`. Everything the board is told lands in
`FakeSerial.log`, in order: ("move", (x, y, z), ms), ("turn", angle, ms),
("suction", is_on).
"""

import contextlib
import io
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from maxarm import geometry                                   # noqa: E402
from maxarm import arm as arm_module                          # noqa: E402
from maxarm import protocol as protocol_module                # noqa: E402
from maxarm import transport as transport_module              # noqa: E402
from maxarm.arm import MaxArm                                 # noqa: E402
from maxarm.protocol import ReplProtocol                      # noqa: E402
from maxarm.spaces import CONFIG_DIR                          # noqa: E402
from maxarm.transport import SerialTransport                  # noqa: E402

Position = Tuple[float, float, float]

# How much of the remaining distance the servos close between one read and the
# next: what lets a stop catch the arm part of the way through a leg.
TRACKING_GAIN = 0.6


class FakeBoardArm:
    """`arm` in the board's namespace: ESPMax's semantics."""

    def __init__(self, log: list) -> None:
        self.log = log
        self.position: Position = geometry.HOME      # last accepted command
        self.actual: Position = geometry.HOME        # where the servos are
        self.is_frozen = False                       # accepts and ignores, like a clamped servo
        self.deadband_mm = 0.0                       # targets nearer than this are not moved to
        self.stop_short_mm = 0.0                     # and farther ones stopped this far short of

    def set_position(self, position: Position, duration_ms: int):
        x, y, z = position
        self.log.append(("move", (x, y, z), int(duration_ms)))
        z = min(z, geometry.Z_CLAMP)
        if math.hypot(x, y) < geometry.BLIND_RADIUS:
            return None
        reason = geometry.limit_reason((x, y, z))
        if reason in ("out_of_reach", "base_angle_out_of_range",
                      "servo2_angle_out_of_range", "servo3_angle_out_of_range"):
            print(f"{math.dist((x, y, z), (0.0, 0.0, geometry.L0)):.4f}")
            print(f"{geometry.L2} {geometry.L3}")
            return False
        if self.is_frozen or reason in ("servo2_clamped", "servo3_clamped"):
            return True
        self.position = self._settles_at((x, y, z))
        return True

    def _settles_at(self, target: Position) -> Position:
        gap = math.dist(target, self.actual)
        if self.deadband_mm and gap <= self.deadband_mm:
            return self.position
        if not self.stop_short_mm:
            return target
        return tuple(goal - self.stop_short_mm * (goal - now) / gap for goal, now in zip(target, self.actual))

    def read_position(self):
        self.track()
        return tuple(int(round(value)) for value in self.actual)

    def track(self) -> None:
        """The servos close part of the distance between one read and the next."""
        self.actual = tuple(self.actual[i] + TRACKING_GAIN * (self.position[i] - self.actual[i])
                            for i in range(3))

    def go_home(self, duration_ms: int = 2000) -> None:
        self.position = self.actual = geometry.HOME


class FakeBusServo:
    FITTED_IDS = (1, 2, 3)

    def __init__(self, arm: FakeBoardArm) -> None:
        self.arm = arm
        self.is_answering = True
        self.voltage_mv = 12300
        self.is_bus_echoing = False

    def get_position(self, servo_id: int):
        if not self.is_answering or (servo_id not in self.FITTED_IDS and not self.is_bus_echoing):
            return False
        if servo_id == 1:
            self.arm.track()                     # a read of all three servos is one step closer
        pulses = geometry.deg_to_pulse(geometry.inverse(self.arm.actual))
        return int(round(pulses[min(servo_id, 3) - 1]))

    def get_vin(self, servo_id: int):
        if not self.is_answering or servo_id not in self.FITTED_IDS:
            return False
        return self.voltage_mv


class FakeNozzle:
    def __init__(self, log: list) -> None:
        self.log = log

    def on(self) -> None:
        self.log.append(("suction", True))

    def off(self) -> None:
        self.log.append(("suction", False))

    def set_angle(self, angle: float = 0.0, duration: int = 1000) -> None:
        self.log.append(("turn", float(angle), int(duration)))


class FakeSerial:
    """A MicroPython v1.12 raw REPL on a wire. Executes whatever it is sent."""

    def __init__(self, *args, **kwargs) -> None:
        self.port = self.baudrate = self.timeout = None
        self.dtr = self.rts = None
        self.outbox = b""
        self.buffered_code = b""
        self.is_raw = False
        self.soft_reboots = 0
        self.executions = 0        # round trips
        self.log: list = []
        self.arm = FakeBoardArm(self.log)
        self.bus_servo = FakeBusServo(self.arm)
        self.namespace = {"arm": self.arm, "nozzle": FakeNozzle(self.log),
                          "bus_servo": self.bus_servo, "time": object(), "gc": _FakeGc()}

    def open(self) -> None:
        self.outbox += (b"\r\nrst:0x1 (POWERON_RESET)\r\nMicroPython v1.12 on 2019-12-20; "
                        b"ESP32 module\r\n>>> ")
        self.arm.go_home()

    def close(self) -> None:
        pass

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        self.outbox = b""

    @property
    def in_waiting(self) -> int:
        return len(self.outbox)

    def read(self, size: int = 1) -> bytes:
        taken, self.outbox = self.outbox[:size], self.outbox[size:]
        return taken

    def write(self, data: bytes) -> int:
        for byte in bytes(data):
            if byte == 0x01:
                self.is_raw = True
                self.outbox += b"raw REPL; CTRL-B to exit\r\n>"
            elif byte == 0x02:
                self.is_raw = False
                self.outbox += b"\r\n>>> "
            elif byte == 0x03:
                self.buffered_code = b""
            elif byte == 0x04 and self.is_raw:
                self._execute()
            elif byte in (0x0A, 0x0D) and not self.is_raw:
                self.outbox += b"\r\n>>> "
            else:
                self.buffered_code += bytes([byte])
        return len(data)

    def _execute(self) -> None:
        source, self.buffered_code = self.buffered_code.decode(), b""
        if not source:
            self.soft_reboots += 1
            self.arm.go_home()
            self.outbox += b"OK\r\nMPY: soft reboot\r\nraw REPL; CTRL-B to exit\r\n>"
            return
        self.executions += 1
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                exec(source, self.namespace)      # noqa: S102 - that is the point
            self.outbox += b"OK" + captured.getvalue().encode() + b"\x04\x04>"
        except Exception as error:
            self.outbox += b"OK\x04" + repr(error).encode() + b"\x04>"

    def moves(self) -> List[Position]:
        return [entry[1] for entry in self.log if entry[0] == "move"]


class _FakeGc:
    def collect(self) -> None:
        pass


@contextlib.contextmanager
def instant_waits(waits: Optional[list] = None):
    """No real sleeping: every wait the library asks for is recorded, in ms, and skipped.

    The constants stay as the library ships them, so the durations it commands
    and the waits it asks for are the real ones. A pending stop() still ends a wait.
    """
    record = waits if waits is not None else []
    original_wait = MaxArm._wait
    patches = [(protocol_module, "INTERRUPT_SETTLE_S", 0.0),
               (protocol_module, "BOARD_CHECK_RETRY_S", 0.0),
               (protocol_module, "SETTLE_MS", 0)]
    saved = [(module, name, getattr(module, name)) for module, name, _ in patches]

    def wait(self, milliseconds: float) -> bool:
        record.append(round(milliseconds))
        return not self._abort.is_set()

    MaxArm._wait = wait
    for module, name, value in patches:
        setattr(module, name, value)
    try:
        yield record
    finally:
        MaxArm._wait = original_wait
        for module, name, value in saved:
            setattr(module, name, value)


@contextlib.contextmanager
def fake_pyserial():
    """Every `SerialTransport` opened inside this gets a fake board. Yields the boards made."""
    boards: List[FakeSerial] = []

    def make() -> FakeSerial:
        board = FakeSerial()
        boards.append(board)
        return board

    original = transport_module._load_pyserial
    transport_module._load_pyserial = lambda: make
    try:
        yield boards
    finally:
        transport_module._load_pyserial = original


@contextlib.contextmanager
def fake_arm(config_dir: Path = CONFIG_DIR, connect: bool = True):
    """A `MaxArm` on a fake board, waits instant. Yields (arm, board, waits)."""
    board = FakeSerial()
    protocol = ReplProtocol(SerialTransport("/dev/fake", port_factory=lambda: board))
    with instant_waits() as waits:
        arm = MaxArm(config_dir=config_dir, protocol=protocol)
        if connect:
            arm.connect()
            board.log.clear()
        try:
            yield arm, board, waits
        finally:
            arm.disconnect()


__all__ = ["FakeSerial", "fake_arm", "fake_pyserial", "instant_waits", "arm_module"]
