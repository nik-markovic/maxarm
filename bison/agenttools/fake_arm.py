#!/usr/bin/env python3
"""A MaxArm that exists only in RAM: board, firmware quirks, serial port.

The point is not to simulate a robot. It is to reproduce the four ways this
particular board misleads its driver, so that the library's defences can be
tested without risking a real nozzle:

  1. `set_position()` returns `None` inside the 50 mm blind cylinder,
  2. returns `False` when the IK will not solve,
  3. returns **`True` while servo 2 sits silently clamped** and the arm does
     not move -- the one real lie,
  4. droops: a commanded z near the desk is not the z the arm reaches, and
     below the surface it simply stops.

Reachability uses the library's own `geometry`, which was validated against the
live board over 336 points, so "the fake arm refused" means the real one would
have. The desk height, the sag coefficient and the tracking lag are invented;
they are shapes of behaviour, not measurements.

Everything runs through the real `SerialTransport` and `ReplProtocol`, so the
raw-REPL framing, the result marker and the board's stdout chatter are all
exercised too -- not stubbed out.
"""

import contextlib
import io
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maxarm import geometry, tuning                           # noqa: E402
from maxarm import maxarm as maxarm_module                    # noqa: E402
from maxarm import protocol as protocol_module                # noqa: E402
from maxarm.config import ArmConfig                           # noqa: E402
from maxarm.maxarm import MaxArm                              # noqa: E402
from maxarm.protocol import ReplProtocol                      # noqa: E402
from maxarm.transport import SerialTransport                  # noqa: E402

Position = Tuple[float, float, float]

# Desk contact as the owner observed it: command z=46 at long reach and the
# cup presses into the surface while the readback shows 42. The lie is in the
# same direction as the sag, which is exactly why one look is not enough to
# tell them apart.
DESK_CONTACT_Z = 47.0
DESK_CONTACT_R = 170.0
DESK_PRESS_MM = 4.0
SAG_START_R = 150.0      # gravity droop begins to matter past this reach
SAG_PER_MM = 0.02        # ~2 mm of droop at full extension
TRACKING_GAIN = 0.85     # fraction of the remaining distance covered per command


class FakeBusServo:
    """Pulse-level feedback, including the board's False-on-no-answer habit."""

    FITTED_IDS = (1, 2, 3)

    def __init__(self, arm: "FakeBoardArm") -> None:
        self.arm = arm
        self.is_answering = True
        self.loaded = [True, True, True]
        self.voltage_mv = 7400          # off the barrel jack; USB does not power these
        self.is_bus_echoing = False     # when true, even absent servos "answer"

    def get_position(self, servo_id: int):
        if not self.is_answering:
            return False
        if servo_id not in self.FITTED_IDS and not self.is_bus_echoing:
            return False
        try:
            pulses = geometry.position_to_pulses(self.arm.actual)
        except geometry.Unreachable:
            return False
        return int(round(pulses[min(servo_id, 3) - 1]))

    def get_vin(self, servo_id: int):
        if not self.is_answering or servo_id not in self.FITTED_IDS:
            return False
        return self.voltage_mv

    def load(self, servo_id: int) -> None:
        self.loaded[servo_id - 1] = True

    def unload(self, servo_id: int) -> None:
        self.loaded[servo_id - 1] = False


class FakePwmPin:
    """Enough of `machine.PWM` for the vent pulse to run as written."""

    def __init__(self, nozzle: "FakeNozzle", name: str) -> None:
        self.nozzle, self.name, self.value = nozzle, name, 0

    def duty(self, value: int) -> None:
        self.value = int(value)
        self.nozzle.events.append(f"{self.name}:{self.value}")


class FakeNozzle:
    def __init__(self, hz: int = 1000) -> None:
        self.is_on = False
        self.angle = 0.0
        self.events: List[str] = []
        # The board exposes the pump and valve H-bridges as plain attributes,
        # and `release()` drives the valve directly because `off()` cannot be
        # called twice. Fake them so that expression executes here too.
        self.hz = hz
        self.valve_f = FakePwmPin(self, "valve_f")
        self.valve_b = FakePwmPin(self, "valve_b")

    def on(self) -> None:
        self.is_on = True
        self.events.append("on")

    def off(self) -> None:
        self.is_on = False
        self.events.append("off")

    def set_angle(self, angle: float = 0.0, duration: int = 1000) -> None:
        self.angle = angle
        self.events.append(f"angle:{angle}")


class FakeBoardArm:
    """What `arm` looks like in the board's namespace -- ESPMax's semantics."""

    def __init__(self) -> None:
        self.position: Position = geometry.HOME_COMMAND   # last ACCEPTED command
        self.actual: Position = geometry.HOME_COMMAND     # where the servos are
        self.move_count = 0
        self.commands: List[Position] = []
        self.durations: List[int] = []
        # Accept commands and do not move, as a silently clamped servo does.
        self.is_frozen = False
        # Per-instance so a check can dial them: how far the arm droops at
        # reach, and how much of the remaining distance it closes between one
        # look and the next. Turning the gain down is how a test gets to see a
        # move that is genuinely still in flight.
        self.sag_per_mm = SAG_PER_MM
        self.tracking_gain = TRACKING_GAIN
        # Stand in for the time real servos take, so a test can watch a move
        # from another thread while it is still running.
        self.command_delay_s = 0.0

    def set_position(self, position: Position, duration_ms: int):
        x, y, z = position
        self.commands.append((x, y, z))
        self.durations.append(int(duration_ms))
        if self.command_delay_s:
            time.sleep(self.command_delay_s)
        z = min(z, geometry.Z_CLAMP)          # silently pinned, reported as honoured
        if math.hypot(x, y) < geometry.BLIND_RADIUS:
            return None

        reason = geometry.limit_reason((x, y, z))
        if reason in ("out_of_reach", "base_angle_out_of_range",
                      "servo2_angle_out_of_range", "servo3_angle_out_of_range"):
            # The compiled IK prints the required reach and the link lengths to
            # stdout before failing, which lands in the middle of our reply.
            print(f"{math.dist((x, y, z), (0.0, 0.0, geometry.L0)):.4f}")
            print(f"{geometry.L2} {geometry.L3}")
            return False

        self.move_count += 1
        if self.is_frozen or reason in ("servo2_clamped", "servo3_clamped"):
            return True                       # pinned, and still reports success

        self.position = (x, y, z)
        return True

    def _goal(self) -> Position:
        """Where the servos are actually heading, sag and desk included."""
        x, y, z = self.position
        reach = math.hypot(x, y)
        z -= self.sag_per_mm * max(0.0, reach - SAG_START_R)
        if self.position[2] <= DESK_CONTACT_Z and reach >= DESK_CONTACT_R:
            z -= DESK_PRESS_MM                # the cup is loaded against the desk
        return (x, y, z)

    def read_position(self):
        # The arm closes part of the remaining distance between one look and
        # the next, which is what makes it possible to watch a move in flight
        # and to see an in-between pose rather than only the endpoints.
        if not self.is_frozen:
            goal = self._goal()
            self.actual = tuple(self.actual[i] + self.tracking_gain * (goal[i] - self.actual[i])
                                for i in range(3))
        try:
            geometry.inverse(self.actual)
        except geometry.Unreachable:
            return False
        return tuple(int(round(value)) for value in self.actual)   # 1 mm quantisation

    def verify_position(self, x: float, y: float, z: float) -> bool:
        # The board feeds x straight in here while set_position negates it, so
        # the caller is expected to pre-negate. Undo that to reuse geometry.
        try:
            geometry.position_to_pulses((-x, y, z))
            return True
        except geometry.Unreachable:
            print(f"{math.dist((-x, y, z), (0.0, 0.0, geometry.L0)):.4f}")
            return False

    def position_to_pulses(self, position: Position):
        x, y, z = position
        return geometry.position_to_pulses((-x, y, z))

    def go_home(self, duration_ms: int = 2000) -> None:
        self.position = self.actual = geometry.HOME_COMMAND


class FakeBoardTime:
    def __init__(self) -> None:
        self.slept_ms: List[int] = []

    def sleep_ms(self, milliseconds: int) -> None:
        self.slept_ms.append(milliseconds)


class FakeGarbageCollector:
    def collect(self) -> None:
        pass

    def mem_free(self) -> int:
        return 51_200


class FakeSerial:
    """A MicroPython v1.12 raw REPL on a wire. Executes whatever it is sent."""

    def __init__(self, *args, **kwargs) -> None:
        self.port = self.baudrate = self.timeout = None
        self.dtr = self.rts = None
        self.outbox = b""
        self.buffered_code = b""
        self.is_raw = False
        self.executions = 0        # round trips, for asserting batching works
        self.broken_by = None      # a test's hook to spoil the board at open()
        self.soft_reboots = 0      # Ctrl-D on an empty buffer: a real trap, see below
        self.arm = FakeBoardArm()
        self.nozzle = FakeNozzle()
        self.bus_servo = FakeBusServo(self.arm)
        self.board_time = FakeBoardTime()
        self.namespace = {"arm": self.arm, "time": self.board_time,
                          "nozzle": self.nozzle, "bus_servo": self.bus_servo,
                          "gc": FakeGarbageCollector()}

    def open(self) -> None:
        if self.broken_by is not None:
            self.broken_by(self)
        # The port open resets the ESP32, so the board announces itself again.
        self.outbox += (b"\r\nrst:0x1 (POWERON_RESET)\r\nPlease wait...\r\nStart\r\n"
                        b"MicroPython v1.12 on 2019-12-20; ESP32 module\r\n>>> ")
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
            # Ctrl-D on an empty buffer is a soft reboot, not a no-op: the board
            # re-runs main.py and re-homes the arm. Measured on the hardware
            # after the library did exactly this by accident.
            self.soft_reboots += 1
            self.arm.go_home()
            self.outbox += (b"OK\r\nMPY: soft reboot\r\n"
                            b"raw REPL; CTRL-B to exit\r\n>")
            return
        self.executions += 1
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                exec(source, self.namespace)      # noqa: S102 - that is the point
            self.outbox += b"OK" + captured.getvalue().encode() + b"\x04" + b"\x04>"
        except Exception as error:
            self.outbox += b"OK\x04" + repr(error).encode() + b"\x04>"


@contextlib.contextmanager
def brisk_tuning():
    """Shrink the waits for the duration of a test, and put them back after.

    The fake board has no servos to settle and no inertia to build up, so the
    real settle times and speeds would buy nothing but twenty minutes of
    staring at a terminal. Everything else -- margins, step sizes, thresholds,
    the retarget budget -- is left exactly as the library ships, because those
    are what the checks are about.
    """
    brisk = dict(SETTLE_MS=10, SETTLE_READS=3, POLL_MS=2, MIN_MOVE_MS=4,
                 TRAVEL_SPEED_MM_S=1500.0, DESCENT_SPEED_MM_S=1500.0, PICK_DWELL_S=0.0,
                 SUCTION_SETTLE_MS=0, RELEASE_DWELL_MS=0, NOZZLE_SETTLE_MS=0,
                 NOZZLE_MS_PER_DEG=0.0, NOZZLE_MIN_TURN_MS=1)
    # `RELEASE_HOLD_MS` is a board fact rather than a tuning value -- the real
    # firmware holds its vent valve open for a second on a thread -- so it does
    # not live in `tuning.py` and has to be shrunk where it is bound. A fake
    # board has no valve to wait for, and waiting anyway cost a full second per
    # release across the whole suite.
    patches = [(tuning, name, value) for name, value in brisk.items()]
    patches += [(protocol_module, "RELEASE_HOLD_MS", 0),
                (maxarm_module, "RELEASE_HOLD_MS", 0),
                # Nothing is looping on a fake board, so there is no interrupt
                # to wait for. Two of these per connect, and the suite connects
                # for almost every check.
                (protocol_module, "INTERRUPT_SETTLE_S", 0.0),
                # The connect check re-reads the servo bus before believing a
                # bad answer. A fake fault stays broken, so the retries are
                # certain to be spent -- just not slowly.
                (protocol_module, "BOARD_CHECK_RETRY_S", 0.0)]
    original = [(module, name, getattr(module, name)) for module, name, _ in patches]
    for module, name, value in patches:
        setattr(module, name, value)
    try:
        yield
    finally:
        for module, name, value in original:
            setattr(module, name, value)


@contextlib.contextmanager
def fake_arm(config: Optional[ArmConfig] = None, break_board=None):
    """A connected `MaxArm` on a fake board. Yields (arm, wire).

    `break_board` is called with the wire the moment the port opens and before
    anything is asked of the board -- the hook for testing a board that is
    reachable but not fit to drive the arm.
    """
    resolved = (config or ArmConfig()).resolved()

    def make_port():
        port = FakeSerial()
        if break_board is not None:
            port.broken_by = break_board
        return port

    transport = SerialTransport(resolved.device, resolved.baud, port_factory=make_port)
    protocol = ReplProtocol(transport)
    arm = MaxArm(resolved, transport=transport, protocol=protocol)
    with brisk_tuning():
        arm.connect()
        try:
            yield arm, transport.port
        finally:
            arm.disconnect()
