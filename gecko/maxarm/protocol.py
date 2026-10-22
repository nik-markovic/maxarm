#!/usr/bin/env python3
"""Everything ever said to the board, in one place.

`ReplProtocol` drives stock firmware over its MicroPython REPL, using the
`arm`, `nozzle` and `bus_servo` objects the board's `main.py` leaves live:

  * **It writes nothing and binds nothing.** Every command is a throwaway
    expression in the raw REPL. Unplug the board and it is as it shipped.
  * **It reads answers by marker.** The board's compiled IK prints to stdout
    when a target will not solve, so stdout is not ours alone.

`MAXARM_TRACE=1` echoes every expression sent, with its answer and timing, on
stderr. `run()` is the only way anything reaches the board, so the trace is
complete: a command that is not in it was not sent, and a gap in the
timestamps is proof that nothing was sent then.

**Positions here are true z** (`raw.py`): `set_position()` sends the raw z,
and the reads give back true z. Only the commands themselves carry raw z, so
the trace notes each move and read with both.
"""

import ast
import os
import sys
import time
from statistics import median
from typing import Optional, Tuple

from . import geometry, raw
from .geometry import Position
from .transport import SerialTransport, TransportError

CTRL_A = b"\x01"
CTRL_B = b"\x02"
CTRL_C = b"\x03"
CTRL_D = b"\x04"
RAW_REPL_BANNER = b"raw REPL"
PROMPT = b">>>"
RESULT_MARKER = "#R#"

# The port-open reset re-runs main.py: 4.7 s once, 15.1 s twice running, the
# long ones being the board's BLE init timing out. This is only the give-up point.
CONNECT_TIMEOUT_S = 25.0
COMMAND_TIMEOUT_S = 5.0
# How long a Ctrl-C needs to land before the next byte. If main.py is in a loop,
# a Ctrl-A sent too soon is line-edited into garbage by the friendly REPL.
INTERRUPT_SETTLE_S = 0.2

# The connect check reads the servo bus, and bus reads drop: one connect in
# twenty failed on a healthy arm that had just homed. So a bad reading is read
# again, after a pause, before it is believed.
BOARD_CHECK_ATTEMPTS = 3
BOARD_CHECK_RETRY_S = 0.3
# Asked of a servo id that is not fitted: if it answers, the bus is echoing.
ABSENT_SERVO_ID = 5
# The servos run off the barrel jack; USB powers only the ESP32. Below this the
# supply is off, and the servos still answer reads while being unable to move.
SERVO_VOLTAGE_MIN_MV = 6000

# A read waits for the arm to stop creeping after a move, then takes the median of
# several: a servo's read flickers by a pulse.
SETTLE_MS = 400
SETTLE_READS = 5
READ_PULSES = "[bus_servo.get_position(i) for i in (1, 2, 3)]"

BOARD_CHECK = (
    "([n in globals() for n in ('arm', 'bus_servo', 'nozzle', 'time', 'gc')],"
    " arm.read_position(), bus_servo.get_position({absent}), bus_servo.get_vin(1))"
)

IS_TRACED = bool(os.environ.get("MAXARM_TRACE"))
_TRACE_START = time.monotonic()
_TRACE_WIDTH = 96


class ProtocolError(RuntimeError):
    """The board answered with a traceback, or with nothing we can parse."""


class BoardNotReadyError(ProtocolError):
    """The board answers, but is in no state to drive the arm. The message says why."""


def trace_note(text: str) -> None:
    """A trace line that is not a board command, marked `#`."""
    _trace("#", text)


def _trace(marker: str, text: str, started: Optional[float] = None) -> float:
    """`>` sent, `<` answered, `!` refused. Seconds since import, and round-trip time."""
    if not IS_TRACED:
        return 0.0
    now = time.monotonic()
    took = f" [{(now - started) * 1000:.0f} ms]" if started is not None else ""
    body = text if len(text) <= _TRACE_WIDTH else text[:_TRACE_WIDTH - 1] + "…"
    print(f"  {now - _TRACE_START:8.3f}s {marker} {body}{took}", file=sys.stderr)
    return now


class ReplProtocol:
    """Stock firmware over the raw MicroPython REPL."""

    def __init__(self, transport: SerialTransport) -> None:
        self.transport = transport

    # --- session ----------------------------------------------------------

    def connect(self) -> None:
        self.transport.open()
        self._wait_for_prompt()
        self._enter_raw_repl()
        # One statement that does nothing, which clears any half-typed line and
        # proves the framing. Not an empty program: in the v1.12 raw REPL,
        # Ctrl-D on an empty buffer is a soft reboot that re-homes the arm.
        self.run("pass")

    def check_board(self) -> None:
        """Prove the board can drive the arm, or raise `BoardNotReadyError`. Reads only."""
        for attempt in range(1, BOARD_CHECK_ATTEMPTS + 1):
            reason, is_transient = self._board_problem()
            if reason is None:
                return
            if not is_transient:
                raise BoardNotReadyError(reason)
            if attempt == BOARD_CHECK_ATTEMPTS:
                raise BoardNotReadyError(f"{reason} (unchanged over {attempt} reads)")
            time.sleep(BOARD_CHECK_RETRY_S)

    def _board_problem(self) -> Tuple[Optional[str], bool]:
        """What is wrong, and whether reading again could change the answer."""
        globals_present, position, absent, voltage = self.evaluate(
            BOARD_CHECK.format(absent=ABSENT_SERVO_ID))
        if not all(globals_present):
            return ("the board's main.py did not finish: arm/bus_servo/nozzle are not all "
                    "defined. Power-cycle the arm and reconnect.", False)
        if absent is not False:
            return (f"servo id {ABSENT_SERVO_ID} is not fitted but answered {absent!r}: the bus "
                    "is echoing, so no readback can be trusted. Power-cycle the arm.", False)
        if not position:
            return ("the servo bus did not answer a position read. Check the arm's DC supply "
                    "and the servo cabling.", True)
        if not isinstance(voltage, int) or voltage < SERVO_VOLTAGE_MIN_MV:
            reading = f"{voltage / 1000.0:.2f} V" if isinstance(voltage, int) else repr(voltage)
            return (f"servo supply reads {reading}, below {SERVO_VOLTAGE_MIN_MV / 1000:.1f} V. "
                    "USB powers the ESP32 but not the servos -- check the barrel jack and its "
                    "switch.", True)
        return (None, False)

    def disconnect(self) -> None:
        if not self.transport.is_open:
            return
        try:
            self.transport.write(CTRL_B)    # leave a friendly REPL for the next user
        except TransportError:
            pass
        finally:
            self.transport.close()

    def _wait_for_prompt(self) -> None:
        """Wait out the port-open reset before sending anything.

        Writing into a booting board once left main.py half-built. So no newline
        first and no fixed sleep: one connect in five takes 13 s rather than 3.
        """
        banner = self.transport.read_until(PROMPT, CONNECT_TIMEOUT_S)
        if PROMPT not in banner:
            # Attached after main.py had already settled: nudge it.
            self.transport.write(b"\r\n")
            banner = self.transport.read_until(PROMPT, 5.0)
            if PROMPT not in banner:
                raise ProtocolError(f"no prompt within {CONNECT_TIMEOUT_S}s of opening the "
                                    f"port; last reply: {banner[-200:]!r}")
        self.transport.reset_input()

    def _enter_raw_repl(self, attempts: int = 3) -> None:
        """Ctrl-A, and prove it took: otherwise everything after it is line-edited."""
        banner = b""
        for _ in range(attempts):
            self.transport.write(CTRL_C)
            time.sleep(INTERRUPT_SETTLE_S)
            self.transport.write(CTRL_C)    # a second one interrupts a running loop
            time.sleep(INTERRUPT_SETTLE_S)
            self.transport.reset_input()
            self.transport.write(CTRL_A)
            banner = self.transport.read_until(RAW_REPL_BANNER, 2.0)
            if RAW_REPL_BANNER in banner:
                # The prompt nearly always arrives with the banner; wait only if not.
                if b">" not in banner.split(RAW_REPL_BANNER, 1)[1]:
                    self.transport.read_until(b">", 1.0)
                return
        raise ProtocolError(f"board did not enter raw REPL; last reply: {banner!r}")

    # --- raw execution ----------------------------------------------------

    def run(self, code: str) -> str:
        """Execute code in the raw REPL and return its stdout."""
        started = _trace(">", code)
        self.transport.reset_input()
        self.transport.write(code.encode() + CTRL_D)
        # The board answers OK, stdout, 0x04, stderr, 0x04, then the raw prompt.
        raw = self.transport.read_until(b"\x04>", COMMAND_TIMEOUT_S)
        if b"OK" not in raw:
            _trace("!", f"no acknowledgement: {raw!r}", started)
            raise ProtocolError(f"no acknowledgement from board: {raw!r}")
        if b"\x04>" not in raw:
            _trace("!", f"timed out mid-reply: {raw!r}", started)
            raise ProtocolError(f"timed out mid-reply after {COMMAND_TIMEOUT_S}s: {raw!r}")
        fields = raw.split(b"OK", 1)[1].split(CTRL_D)
        stdout = fields[0].decode("utf-8", "replace").strip()
        stderr = fields[1].decode("utf-8", "replace").strip() if len(fields) > 1 else ""
        if stderr:
            _trace("!", stderr.replace("\n", " | "), started)
            raise ProtocolError(f"board raised:\n{stderr}")
        _trace("<", stdout or "(no output)", started)
        return stdout

    def evaluate(self, expression: str):
        """A board expression's value, as a Python literal.

        A MemoryError on the board is usually heap fragmentation, so one
        collection and a retry is worth it; `gc` is already a board global.
        """
        code = f"print('{RESULT_MARKER}', repr({expression}))"
        try:
            text = self.run(code)
        except ProtocolError as error:
            if "MemoryError" not in str(error):
                raise
            self.run("gc.collect()")
            text = self.run(code)
        payload = None
        for line in text.splitlines():
            if line.startswith(RESULT_MARKER):
                payload = line[len(RESULT_MARKER):].strip()
        if payload is None:
            raise ProtocolError(f"no result line in reply {text!r}")
        try:
            return ast.literal_eval(payload)
        except (SyntaxError, ValueError) as error:
            raise ProtocolError(f"unparseable result {payload!r} in reply {text!r}") from error

    # --- motion -----------------------------------------------------------

    def set_position(self, position: Position, duration_ms: int,
                     nozzle: Optional[Tuple[float, int]] = None) -> Optional[bool]:
        """Send the cup to a true position: True accepted, False refused, None in the blind cylinder.

        `nozzle`, as (angle, duration_ms), turns the cup in the same round trip,
        so the two start together. True is not a promise -- servo 2 can clamp
        silently -- which is why targets are checked with `geometry` first.
        """
        trace_note(f"to {raw.describe(position)}")
        x, y, z = raw.raw_position(position)
        command = f"arm.set_position(({x:.2f}, {y:.2f}, {z:.2f}), {int(duration_ms)})"
        if nozzle is not None:
            command = f"({_set_angle(*nozzle)}, {command})[1]"
        return self.evaluate(command)

    def set_nozzle_angle(self, angle_deg: float, duration_ms: int) -> None:
        """Turn the cup about z. The board ramps it over `duration_ms` and reports nothing."""
        self.run(_set_angle(angle_deg, duration_ms))

    def set_suction(self, is_on: bool) -> None:
        """Pump on; or pump off with the vent valve held open for a second, on a board thread."""
        self.run("nozzle.on()" if is_on else "nozzle.off()")

    def read_pulses(self) -> Optional[Tuple[int, int, int]]:
        """The three servos' own positions, in pulses. None if the bus dropped one."""
        reply = self.evaluate(READ_PULSES)
        if not isinstance(reply, list) or any(value is False or value is None for value in reply):
            return None
        return tuple(reply)

    def read_position(self) -> Optional[Position]:
        """Where the servos say the tip is, true z. None if the bus dropped a read.

        The servos' pulses through the firmware's own kinematics, on the host. The
        board's `arm.read_position()` does the same and rounds with `int(v + 0.5)`,
        which truncates negative values: y in front of the arm, and x to its right,
        read up to 1 mm out.
        """
        return _true_position(self.read_pulses())

    def read_settled_position(self) -> Optional[Position]:
        """Let the arm stop creeping, then the tip, true z, from the median of several reads, servo by servo."""
        return _true_position(self.read_settled_pulses())

    def read_settled_pulses(self) -> Optional[Tuple[float, float, float]]:
        """Let the arm stop creeping, then each servo's median pulse over several reads."""
        time.sleep(SETTLE_MS / 1000.0)
        samples = [sample for sample in (self.read_pulses() for _ in range(SETTLE_READS))
                   if sample is not None]
        if not samples:
            return None
        return tuple(median(sample[servo] for sample in samples) for servo in range(3))


def _true_position(pulses: Optional[Tuple[float, float, float]]) -> Optional[Position]:
    if not pulses:
        return None
    position = raw.true_position(geometry.forward(geometry.pulse_to_deg(pulses)))
    trace_note(f"at {raw.describe(position)}")
    return position


def _set_angle(angle_deg: float, duration_ms: int) -> str:
    angle = max(-90.0, min(90.0, float(angle_deg)))
    return f"nozzle.set_angle({angle:.2f}, {int(duration_ms)})"
