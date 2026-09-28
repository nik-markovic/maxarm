#!/usr/bin/env python3
"""The board's vocabulary: every expression we ever send, in one place.

This file is deliberately parallel to `maxarm.py`. Here is what the *board*
understands; there is what *we* decide to do about it. Swapping transports
later (the 0xAA 0x55 binary protocol, if the board is ever reflashed) means
writing another `BoardProtocol` and changing nothing above it.

`ReplProtocol` drives stock firmware over its MicroPython REPL. Two properties
are worth keeping:

  * **It writes nothing and binds nothing.** No files, no flash, not even a
    name in the board's namespace -- every command is a throwaway expression
    evaluated in the raw REPL, which lives in RAM. The board can be unplugged
    at any moment and is exactly as it shipped.
  * **It reads results by marker.** The board's compiled IK prints diagnostics
    to stdout when a target will not solve, so stdout is not ours alone and the
    answer has to be picked out rather than assumed to be the whole reply.

The methods map onto `espmax.ESPMax` and `SuctionNozzle` as the board's
`main.py` leaves them: `arm`, `bus_servo` and `nozzle` are live globals because
`main.py` arms a timer ISR and then falls through to the REPL.
"""

import ast
import os
import sys
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from .geometry import Angles, Position, Pulses, pulse_to_deg
from .transport import Transport, TransportError
from .tuning import COMMAND_TIMEOUT_S, CONNECT_TIMEOUT_S

CTRL_A = b"\x01"
CTRL_B = b"\x02"
CTRL_C = b"\x03"
CTRL_D = b"\x04"

RAW_REPL_BANNER = b"raw REPL"
PROMPT = b">>>"
RESULT_MARKER = "#R#"

# `MAXARM_TRACE=1` echoes every expression sent to the board, on stderr.
#
# This file is the only place anything reaches the board, and `run()` is the
# only way out of it, so the trace is complete by construction: if a command is
# not in the trace, it was not sent. That is worth having because the arm makes
# noises -- the vent solenoid clicks twice per release, the pump spins down,
# servos twitch on a correction -- and working out which command caused which
# noise by reasoning about the code is exactly the kind of guessing this ends.
IS_TRACED = bool(os.environ.get("MAXARM_TRACE"))
_TRACE_START = time.monotonic()
# Long enough for a `set_position` call, short enough to stay one line.
_TRACE_WIDTH = 96


def trace_note(text: str) -> None:
    """Put a line in the trace that is not a board command.

    For the things a command log cannot show on its own -- what a position
    reading meant, why a correction fired. Marked `#` so it is obviously not
    traffic. Callers guard on `IS_TRACED` when producing the text costs
    anything, because a diagnostic that changes what the arm does is not one.
    """
    _trace("#", text)


def _trace(marker: str, text: str, started: Optional[float] = None) -> float:
    """One traced line: `>` sent, `<` answered, `!` refused. Returns the clock.

    Timestamps are seconds since import and the round-trip time is printed
    beside every answer, because the questions worth asking of a trace are
    *when* something was sent and *how long the board took* -- a gap in the
    timestamps is proof that nothing was sent in it.
    """
    if not IS_TRACED:
        return 0.0
    now = time.monotonic()
    took = f" [{(now - started) * 1000:.0f} ms]" if started is not None else ""
    body = text if len(text) <= _TRACE_WIDTH else text[:_TRACE_WIDTH - 1] + "…"
    print(f"  {now - _TRACE_START:8.3f}s {marker} {body}{took}", file=sys.stderr)
    return now

# How long to let a Ctrl-C land before sending the next thing. This is a real
# board waiting to notice an interrupt -- if `main.py` is inside its own loop,
# Ctrl-A sent too soon is line-edited into garbage by the friendly REPL. Two of
# these on every connect, so it is not free; it has a name so the fake board
# can set it to zero, having no loop to interrupt.
INTERRUPT_SETTLE_S = 0.2

# How many times the connect check may re-read the servo bus before believing
# what it says. About one connect in twenty was failing on a healthy arm, and
# both readings it takes come off a bus whose reads are known to drop. The
# delay is there so the retry is not simply the same 50 ms window again: the
# board has just homed the arm, and a supply sagging under that surge wants a
# moment rather than another immediate look.
BOARD_CHECK_ATTEMPTS = 3
BOARD_CHECK_RETRY_S = 0.3

# How long the board's own homing routine is given -- its own default. The
# board cannot be told to abandon it once it has started; only the waiting for
# it is the caller's, which is what `go_home(is_waited=False)` is for.
HOME_DURATION_MS = 2000

# `off()` spawns a thread that holds the valve open for a second. So a release
# is not instantaneous, and two releases in quick succession overlap.
RELEASE_HOLD_MS = 1000

# A second vent, after the board has closed its own.
#
# **Not used by `release()`. It was, and it made the arm worse.** Kept because
# it is the only way to vent the cup twice and the research behind it is the
# expensive part; `agenttools/pulse-suction.py --vent` is how to try it again.
#
# There is no "blow out" on this hardware. `SuctionNozzle.on()` runs the pump
# one way and `_off()` opens a solenoid that vents the line to atmosphere for
# a second, then closes it -- which leaves the cup a sealed dead volume, still
# holding whatever vacuum did not bleed off in that second. Calling `off()`
# again does nothing: it is guarded by `nozzle_st`, which is already False.
# So the valve is driven directly instead, which is the same thing the rest of
# this file does: calling the board's own objects over the REPL.
#
# Why it made things worse, most likely: `_off()` runs on a board thread that
# closes the valve at its own thousand-millisecond mark, which is exactly when
# a release would fire this. Re-energising a solenoid while its armature is
# still returning can leave it part-seated, and a valve that has not reseated
# is a leak the pump then has to fight on the *next* pick. The coil also ends
# up held on for 1.4 s a cycle rather than 1 s, and these are intermittent-duty
# parts. Neither was instrumented -- the evidence is the owner's, off the arm.
#
# The pump is *not* reversed: it sits on an H-bridge and could be, but a
# diaphragm pump's check valves are passive, so running the motor backwards
# would stall it rather than blow through it.
VENT_CUP = ("(nozzle.valve_f.duty(nozzle.hz), nozzle.valve_b.duty(0),"
            " time.sleep_ms({ms}), nozzle.valve_f.duty(0))[-1]")

# One round trip: command, let the servos settle on the board, then report the
# verdict next to the measured pose. The lambda holds set_position()'s result
# while the tuple's left-to-right evaluation does the delay before the readback.
MOVE_AND_READ = (
    "(lambda ok: (time.sleep_ms({wait}), (ok, arm.read_position()))[1])"
    "(arm.set_position(({x}, {y}, {z}), {ms}))"
)

# The four things that must be true before the board is fit to drive. All
# reads. `get_position` on an id that is not fitted is the control: if that
# answers too, the bus is echoing and no readback means anything.
BOARD_CHECK = (
    "([n in globals() for n in ('arm', 'bus_servo', 'nozzle', 'time', 'gc')],"
    " arm.read_position(), bus_servo.get_position({absent}), bus_servo.get_vin(1))"
)

ABSENT_SERVO_ID = 5
# The servos run off the barrel jack; USB powers the ESP32 alone. Measured on
# this arm: 12.32 V at the servo, which is the supply, not a servo spec. The
# floor below is deliberately far under that -- it is there to catch "the DC
# supply is off or failing", a state in which the servos still answer reads and
# cannot drive the arm.
SERVO_VOLTAGE_MIN_MV = 6000


class ProtocolError(RuntimeError):
    """The board answered with a traceback, or with nothing we can parse."""


class BoardNotReadyError(ProtocolError):
    """The board is reachable but not in a fit state to drive the arm.

    Raised by the post-connect check rather than left for the first move to
    discover. A session that starts against a board whose servos are not
    listening will otherwise report a plausible-looking `CLAMPED` on every
    move and waste the operator's afternoon.
    """


class BoardProtocol(ABC):
    """What any transport to this arm has to be able to do."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def check_board(self) -> None:
        """Prove the board can drive the arm, or raise `BoardNotReadyError`."""

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def set_position(self, position: Position, duration_ms: int) -> Optional[bool]: ...

    @abstractmethod
    def move_and_read(self, position: Position, duration_ms: int,
                      settle_ms: int) -> Tuple[Optional[bool], Optional[Position]]: ...

    @abstractmethod
    def read_position(self) -> Optional[Position]: ...

    @abstractmethod
    def read_settled_position(self, reads: int, wait_s: float) -> Optional[Position]: ...

    @abstractmethod
    def read_servo_pulses(self) -> Optional[Pulses]: ...

    @abstractmethod
    def read_joint_angles(self) -> Optional[Angles]: ...

    @abstractmethod
    def get_commanded_position(self) -> Optional[Position]: ...

    @abstractmethod
    def go_home(self, duration_ms: int = HOME_DURATION_MS,
                is_waited: bool = True) -> None: ...

    @abstractmethod
    def set_suction(self, is_on: bool) -> None: ...

    @abstractmethod
    def vent_cup(self, duration_ms: int) -> None: ...

    @abstractmethod
    def set_nozzle_angle(self, angle_deg: float, duration_ms: int) -> None: ...


class ReplProtocol(BoardProtocol):
    """Stock firmware over the raw MicroPython REPL."""

    def __init__(self, transport: Transport,
                 connect_timeout_s: float = CONNECT_TIMEOUT_S,
                 command_timeout_s: float = COMMAND_TIMEOUT_S) -> None:
        self.transport = transport
        self.connect_timeout_s = connect_timeout_s
        self.command_timeout_s = command_timeout_s
        self.is_board_sleep_available = False

    # --- session ----------------------------------------------------------

    def connect(self) -> None:
        self.transport.open()
        self._wait_for_prompt()
        self._enter_raw_repl()
        # Run one statement that does nothing, and require the board to say so.
        # This clears any half-typed line a previous session left in the REPL's
        # buffer and proves the Ctrl-D framing round-trips before anything
        # depends on it.
        #
        # It has to be `pass`, not an empty program: in the v1.12 raw REPL,
        # Ctrl-D on an *empty* buffer is a soft reboot, which re-runs main.py
        # and re-homes the arm. Measured on the board, 2026-09-22.
        self.run("pass")
        # main.py imports time at module scope, so it is normally already a
        # live global. Checking is read-only; importing would bind a name.
        self.is_board_sleep_available = bool(self.evaluate("'time' in globals()"))

    def check_board(self) -> None:
        """Prove the board can actually drive the arm. Reads only; no motion.

        Raises `BoardNotReadyError` naming the remedy. What it cannot prove is
        that the servos will *obey* -- torque and mode are not readable on this
        firmware -- so `motion.py` carries the complementary check: accepted
        commands in open space that produce no movement at all.

        The two readings that come off the servo bus are **re-read before they
        are believed**, because a dropped bus read says nothing -- the same
        rule `read_position()` already followed and this check did not. The
        symptom was a connect in twenty failing with "servo supply reads
        0.00 V" on a perfectly healthy arm. `get_vin()` gives up after 50 ms
        and the board has just finished homing, so a momentary miss or a
        genuine sag under that surge are both expected and both recover.
        """
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
        """What is wrong with the board, and whether asking again could help.

        Structural faults -- a half-built `main.py`, a bus that echoes -- will
        read the same every time and are reported at once. Servo-bus readings
        are the transient ones.
        """
        globals_present, position, absent, voltage = self.evaluate(
            BOARD_CHECK.format(absent=ABSENT_SERVO_ID))

        if not all(globals_present):
            return ("the board's main.py did not finish: arm/bus_servo/nozzle are not all "
                    "defined. Power-cycle the arm and reconnect.", False)
        if absent is not False:
            return (f"servo id {ABSENT_SERVO_ID} is not fitted but answered {absent!r}: the "
                    "bus is echoing, so no readback can be trusted. Power-cycle the arm.",
                    False)
        if not position:
            return ("the servo bus did not answer a position read. Check the arm's DC supply "
                    "and the servo cabling.", True)
        if not isinstance(voltage, int) or voltage < SERVO_VOLTAGE_MIN_MV:
            reading = f"{voltage / 1000.0:.2f} V" if isinstance(voltage, int) else repr(voltage)
            return (f"servo supply reads {reading}, below {SERVO_VOLTAGE_MIN_MV / 1000:.1f} V. "
                    "USB powers the ESP32 but not the servos -- check the barrel jack and its "
                    "switch.", True)
        return (None, False)

    def get_servo_voltage_mv(self, servo_id: int = 1):
        """Servo supply in millivolts, or False if that servo did not answer."""
        return self.evaluate(f"bus_servo.get_vin({int(servo_id)})")

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
        """Wait out the port-open reset. Do not send anything until it lands.

        Writing into a booting board is what once left `main.py` half-built,
        with the timer ISR and `ble` undefined. So: no newline first, and no
        fixed sleep -- one connect in five takes 13 s rather than 3.3 s.
        """
        banner = self.transport.read_until(PROMPT, self.connect_timeout_s)
        if PROMPT in banner:
            self.transport.reset_input()
            return
        # Nudge it, in case we attached after main.py had already settled.
        self.transport.write(b"\r\n")
        banner = self.transport.read_until(PROMPT, 5.0)
        if PROMPT not in banner:
            raise ProtocolError(
                f"no prompt within {self.connect_timeout_s}s of opening the port; "
                f"last reply: {banner[-200:]!r}")
        self.transport.reset_input()

    def _enter_raw_repl(self, attempts: int = 3) -> None:
        """Switch to the classic raw REPL and prove it took.

        v1.12 predates raw-paste mode, so this is the plain Ctrl-A protocol.
        Confirming the banner matters: if Ctrl-A does not take, the friendly
        REPL line-edits everything sent after it into garbage.
        """
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
                # The board sends "raw REPL; CTRL-B to exit\r\n>" in one go, so
                # the prompt is nearly always already in hand. Waiting for it
                # unconditionally meant waiting out this whole timeout on every
                # single connect -- a second of nothing, measured.
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
        raw = self.transport.read_until(b"\x04>", self.command_timeout_s)
        if b"OK" not in raw:
            _trace("!", f"no acknowledgement: {raw!r}", started)
            raise ProtocolError(f"no acknowledgement from board: {raw!r}")
        if b"\x04>" not in raw:
            _trace("!", f"timed out mid-reply: {raw!r}", started)
            raise ProtocolError(f"timed out mid-reply after {self.command_timeout_s}s: {raw!r}")
        fields = raw.split(b"OK", 1)[1].split(CTRL_D)
        stdout = fields[0].decode("utf-8", "replace").strip()
        stderr = fields[1].decode("utf-8", "replace").strip() if len(fields) > 1 else ""
        if stderr:
            _trace("!", stderr.replace("\n", " | "), started)
            raise ProtocolError(f"board raised:\n{stderr}")
        _trace("<", stdout or "(no output)", started)
        return stdout

    def evaluate(self, expression: str):
        """Evaluate a board expression and return it as a Python literal.

        Tagged with a marker because the board's IK writes its own diagnostics
        to stdout on unsolvable targets; those would otherwise parse as the
        answer. A MemoryError here is usually heap fragmentation rather than a
        full heap, so one collection and retry is worth having. `gc` is already
        imported by the board's main.py, so this still binds nothing.
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

    def set_position(self, position: Position, duration_ms: int) -> Optional[bool]:
        """True accepted, False refused, None inside the 50 mm blind cylinder.

        True is not a promise: servo 2 can be silently clamped and still report
        success. Anything that cares must read the pose back.
        """
        x, y, z = position
        return self.evaluate(f"arm.set_position(({x}, {y}, {z}), {int(duration_ms)})")

    def move_and_read(self, position: Position, duration_ms: int,
                      settle_ms: int) -> Tuple[Optional[bool], Optional[Position]]:
        x, y, z = position
        wait = int(duration_ms) + int(settle_ms)
        if self.is_board_sleep_available:
            verdict, measured = self.evaluate(MOVE_AND_READ.format(
                x=x, y=y, z=z, ms=int(duration_ms), wait=wait))
        else:
            # Two round trips, settling on this side. Slower, touches nothing.
            verdict = self.set_position(position, duration_ms)
            time.sleep(wait / 1000.0)
            measured = self.evaluate("arm.read_position()")
        return verdict, _as_position(measured)

    def read_position(self) -> Optional[Position]:
        """Servo-feedback pose in mm, or None if the bus servos did not answer.

        Quantised to 1 mm and jittery by about that much, so a single reading
        is not evidence of a 1-2 mm move.
        """
        return _as_position(self.evaluate("arm.read_position()"))

    def read_settled_position(self, reads: int = 5, wait_s: float = 0.4) -> Optional[Position]:
        """Wait for the creep to stop, then take a median of several reads.

        After a command lands the error keeps shrinking for a few hundred ms,
        worth about 0.8 mm. Re-commanding the same coordinate instead of
        waiting was measured to gain nothing, so this costs no servo traffic.
        """
        time.sleep(wait_s)
        samples: List[Position] = []
        for _ in range(max(reads, 1)):
            reading = self.read_position()
            if reading is not None:
                samples.append(reading)
        if not samples:
            return None
        return tuple(sorted(sample[axis] for sample in samples)[len(samples) // 2]
                     for axis in range(3))

    def get_commanded_position(self) -> Optional[Position]:
        """The last pose the board believes it commanded. No servo read."""
        return _as_position(self.evaluate("arm.position"))

    def read_servo_pulses(self, attempts: int = 3) -> Optional[Pulses]:
        """Measured pulses for servos 1..3, or None if a servo stayed quiet.

        The board returns False for a servo that does not answer, hence the
        per-element check rather than a truth test on the list.

        Retried because a dropped read is transient and says nothing about the
        arm -- the board's own `read_position()` retries each servo three times
        for the same reason. Seen on the hardware: one connect in a handful
        comes back short, and without this a UI blanks its joint display.
        """
        for _ in range(max(attempts, 1)):
            reply = self.evaluate("[bus_servo.get_position(i) for i in (1, 2, 3)]")
            if reply and all(value is not False and value is not None for value in reply):
                return tuple(float(value) for value in reply)
        return None

    def read_joint_angles(self) -> Optional[Angles]:
        pulses = self.read_servo_pulses()
        return pulse_to_deg(pulses) if pulses is not None else None

    def is_position_solvable(self, position: Position) -> bool:
        """Cheap no-motion IK check on the board itself. Advisory only.

        Ignores the servo clamps, and `verify_position()` feeds x straight into
        the solver while `set_position()` negates it first -- so the sign is
        flipped here to match a real move. Normally there is no reason to ask:
        `geometry.limit_reason()` answers the same question locally, and more
        completely. Kept for cross-checking the host model against the board.
        """
        x, y, z = position
        return bool(self.evaluate(f"arm.verify_position({-x}, {y}, {z})"))

    def go_home(self, duration_ms: int = HOME_DURATION_MS,
                is_waited: bool = True) -> None:
        """Run the board's own homing routine.

        `is_waited=False` sends it and returns at once, leaving the waiting to
        the caller -- which is how `MaxArm.home()` makes a Ctrl-C land during
        the homing rather than after it. The board cannot be called off either
        way; only the waiting is the caller's to abandon.
        """
        self.run(f"arm.go_home({int(duration_ms)})")
        if is_waited:
            time.sleep(duration_ms / 1000.0 + 0.3)

    # --- nozzle -----------------------------------------------------------

    def set_suction(self, is_on: bool) -> None:
        self.run("nozzle.on()" if is_on else "nozzle.off()")

    def vent_cup(self, duration_ms: int) -> None:
        """Hold the vent valve open again, after the board has closed it.

        Blocks the board for `duration_ms`, so keep it well inside the command
        timeout. See `VENT_CUP` for why this exists and why the pump is left
        alone.
        """
        self.run(VENT_CUP.format(ms=int(duration_ms)))

    def set_nozzle_angle(self, angle_deg: float, duration_ms: int = 500) -> None:
        """Rotate the cup about Z. The servo takes -90..+90 and clamps beyond."""
        angle = max(-90.0, min(90.0, float(angle_deg)))
        self.run(f"nozzle.set_angle({angle}, {int(duration_ms)})")

    # --- diagnostics ------------------------------------------------------

    def set_servos_loaded(self, is_loaded: bool) -> None:
        """Torque on or off. Off is teaching mode -- the arm goes limp and drops."""
        call = "load" if is_loaded else "unload"
        self.run(f"[bus_servo.{call}(i) for i in (1, 2, 3)]")

    def get_free_memory(self) -> int:
        """Board heap, as a liveness check. `gc` is already a board global."""
        return int(self.evaluate("gc.mem_free()"))


def _as_position(reply) -> Optional[Position]:
    """read_position() returns False when the servo bus stays quiet."""
    if not reply:
        return None
    return tuple(float(value) for value in reply)
