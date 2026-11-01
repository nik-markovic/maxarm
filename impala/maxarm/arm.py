#!/usr/bin/env python3
"""`MaxArm`: move, grab and drop, in whichever coordinates the caller works in.

    arm = MaxArm()
    arm.connect()                         # the board resets, and homes the arm
    arm.use_coordinates(Space.CAMERA)     # once: a point is now a pixel and a height
    arm.set_clearance(14.0)               # once: grab() and drop() need it
    arm.grab(812.0, 640.0, 4.0, angle=12.0)
    arm.drop(1020.0, 700.0, 5.0, angle=0.0)
    arm.carry((812.0, 640.0, 4.0), (1020.0, 700.0, 5.0), angle=12.0, facing=180.0)   # both: any turn

Every motion -- `move()`, `grab()`, `drop()`, `carry()`, `home()` -- goes through
`_move()`, which plans and checks the whole path (`path.py`) before anything
is sent, so a refusal means nothing moved. Each leg is one command, given the
time its speed implies, and waited out. Nothing is read back on the way.

Positions in and out are true, where the cup really is; the firmware's own
are corrected at the board (`raw.py`).
"""

import math
import signal
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union

from . import geometry, nozzle, raw
from .geometry import Position
from .path import CLEARANCE_MARGIN_MM, CROSSING_MM, Planner, Refused
from .protocol import IS_TRACED, ProtocolError, ReplProtocol, trace_note
from .spaces import CONFIG_DIR, Space, Spaces, angle_from_desk, angle_to_desk
from .transport import DEVICE, SerialTransport
from .tunable import tunable

# The owner's yardstick: about a second to cross the desk.
MOVE_FAST_MM_S = tunable("MOVE_FAST_MM_S", 200.0)
# Slow enough not to yank a heavy piece; every move onto or off the desk in grab() and drop().
MOVE_SLOW_MM_S = tunable("MOVE_SLOW_MM_S", 10.0)
# A few millimetres at travel speed would ask the servos for a jerk at full speed.
MOVE_MIN_MS = 150
# The command carries two decimals: a move shorter than this is the same command.
# Anything longer is sent, so the arm's own response to tiny steps can be seen.
MOVE_RESOLUTION_MM = 0.01
# The servos take a move's duration as a 16-bit field, specified to 30 s. A slow
# move longer than that goes faster rather than being garbled.
MOVE_MAX_MS = 30000

# The pump is a physical thing and the board reports nothing about it, so a grip
# and a release are waits. A release is longer: the board vents the cup for 1 s
# and then shuts the valve, and lifting before the residual vacuum bleeds away
# brings the piece up with the cup.
SUCTION_GRAB_MS = tunable("SUCTION_GRAB_MS", 1500.0)
SUCTION_RELEASE_MS = tunable("SUCTION_RELEASE_MS", 3000.0)

# Where the servos are, from their reads, which flicker by a pulse or more: once
# they have stopped drifting -- from PRECISE_STEADY_FROM_S after the travel, the
# mean of the last PRECISE_STEADY_HALF_S against that of the one before it, every
# servo within PRECISE_STEADY_PULSES -- the mean of both (STATUS-impala §19). At
# most PRECISE_SETTLE_MAX_S after the travel. From AT_REST_S after it, the mean of
# PRECISE_STEADY_HALF_S of reads.
PRECISE_STEADY_FROM_S = 1.0
PRECISE_STEADY_HALF_S = 0.5
PRECISE_STEADY_PULSES = 0.35
PRECISE_SETTLE_MAX_S = 3.0
AT_REST_S = 2.0


class MoveType(Enum):
    FAST = "fast"            # straight there at MOVE_FAST_MM_S, no correction
    SLOW = "slow"            # the same at MOVE_SLOW_MM_S: nothing heavy gets yanked
    # Slow, then read until the servos are steady: where they are is the mean of
    # their reads (PRECISE_* above). Not corrected; the nudge that was is kept in
    # arm-nudge.py.
    PRECISE = "precise"


# The three servos' positions, pulses: as each reads, or a mean of reads.
Pulses = Tuple[float, float, float]


class MoveStatus(Enum):
    DONE = "done"
    REFUSED = "refused"      # nothing moved; `detail` says why
    STOPPED = "stopped"      # stop() or Ctrl-C


@dataclass(frozen=True)
class MoveResult:
    """How a call ended. Falsy unless it did everything it was asked."""

    status: MoveStatus
    position: Position       # where the arm was last sent, in the caller's space
    detail: str = ""

    def __bool__(self) -> bool:
        return self.status is MoveStatus.DONE

    def __str__(self) -> str:
        where = "({:.1f}, {:.1f}, {:.1f})".format(*self.position)
        text = f"{self.status.value} at {where}"
        return f"{text}: {self.detail}" if self.detail else text


class NotConnectedError(RuntimeError):
    """A motion was asked for before `connect()`."""


@dataclass(frozen=True)
class _Held:
    height: float            # the desk height it was picked up at
    offset: float            # which way it faces, less which way the cup does; arm degrees
    cup: float               # which way the cup faced as it gripped; arm degrees
    miss: Tuple[float, float]  # where the cup came down, less the piece's centre; arm mm


class MaxArm:
    """A MaxArm on the end of a serial port.

    `config_dir` holds the calibration files (`spaces.py`), None for none;
    `protocol` is for the offline tests' fake board. Normal callers pass neither.
    """

    def __init__(self, device: str = DEVICE, config_dir: Optional[Union[str, Path]] = CONFIG_DIR,
                 protocol: Optional[ReplProtocol] = None) -> None:
        self._protocol = protocol or ReplProtocol(SerialTransport(device))
        self._spaces = Spaces(config_dir)
        self._coordinates = Space.ARM
        self._space = self._spaces.arm
        self._planner = Planner(self._spaces.arm, self._spaces.is_desk_known)
        self._clearance_mm = 0.0
        self._is_connected = False
        self._position: Position = home_position()     # arm space: where it was last sent
        self._servo_deg = 0.0
        self._is_suction_on = False
        self._held: Optional[_Held] = None
        self._abort = threading.Event()
        self._is_frozen = False
        self._travel_ended = 0.0                       # time.monotonic() when the last command's travel was over
        self._settled: Optional[Pulses] = None         # where a precise move ended, its servos' mean, until a command

    # --- session ----------------------------------------------------------

    def connect(self) -> None:
        """Open the port and check the board can drive the arm.

        Opening the port resets the board, which homes the arm and centres the
        cup. Raises `BoardNotReadyError` naming the remedy.
        """
        self._protocol.connect()
        self._protocol.check_board()
        self._position = home_position()     # where the board's main.py sends the arm on every reset
        self._servo_deg, self._is_suction_on, self._held = 0.0, False, None
        self._is_connected, self._is_frozen = True, False
        self._travel_ended, self._settled = time.monotonic(), None
        self._abort.clear()

    def disconnect(self) -> None:
        self._protocol.disconnect()
        self._is_connected = False

    def __enter__(self) -> "MaxArm":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.disconnect()

    # --- configuration, once per run --------------------------------------

    def use_coordinates(self, space: Space) -> None:
        """Take and give every point in `space` from now on. ARM until this is called."""
        self._space = self._spaces.get(space)
        self._coordinates = space

    def use_calibration(self, config_dir: Optional[Union[str, Path]]) -> None:
        """Read the calibration files afresh from `config_dir`; None for no calibration at all.

        For a calibration app. It measures the desk with no desk calibration in
        the way -- no desk floor, so the cup can find a desk lower than the last
        calibration thought -- and then tries out the file it has just written.
        Raises, and changes nothing, if the coordinates in use need a file that
        is not there.
        """
        spaces = Spaces(config_dir)
        space = spaces.get(self._coordinates)
        self._spaces, self._space = spaces, space
        self._planner = Planner(spaces.arm, spaces.is_desk_known)

    def set_clearance(self, height_mm: float) -> None:
        """The height above the desk below which nothing travels sideways.

        Objects on the desk shorter than this are never swept. What the cup
        carries is allowed for on top of it. The desk is `desk.json`'s, or the
        default one; with no calibration in use at all (`config_dir=None`) it
        only sets the travel height of `grab()` and `drop()`.
        """
        if height_mm < 0.0:
            raise ValueError(f"a clearance is a height above the desk, not {height_mm}")
        self._clearance_mm = float(height_mm)

    # --- motion -----------------------------------------------------------

    def move(self, x: float, y: float, z: float, move_type: MoveType = MoveType.FAST, *,
             angle: Optional[float] = None) -> MoveResult:
        """Send the cup to (x, y, z) in the caller's space.

        One leg, or a few when it has to climb over the clearance or go round
        the arm-base limit. `angle` turns the cup -- or the piece it holds -- to
        face that way, degrees from the space's +x towards its +y; the turn
        rides on the leg that crosses the desk, and the move waits for it.
        """
        self._begin()
        point = (float(x), float(y), float(z))
        servo = self._servo_for(point, angle) if angle is not None else None
        return self._move(self._space.to_desk(point), move_type, servo)

    def grab(self, x: float, y: float, height: float, *,
             angle: Optional[float] = None) -> MoveResult:
        """Pick up the piece whose top face is at (x, y, height).

        Lets go of anything held, then: slowly up to the travel height if below
        it, fast across with the cup centring itself, slowly down onto the
        piece and read there as PRECISE is, suction on and waited for,
        slowly back up. The travel height is `clearance + height + 5 mm`, so
        the piece hangs 5 mm above the clearance.

        `angle` says which way the piece faces; `drop()` turns it from there.
        """
        self._begin()
        self._require_clearance("grab")
        return self._grab((float(x), float(y), float(height)), angle, 0.0)

    def drop(self, x: float, y: float, height: float, *,
             angle: Optional[float] = None) -> MoveResult:
        """Set the held piece down with its top face at (x, y, height), facing `angle`.

        `grab()` in reverse: the piece is turned on the way across, and the
        release is waited out -- longer than a grip -- before the slow lift.
        The cup turns 90 degrees either way from where `grab()` centred it; a
        bigger turn is refused before anything moves. `angle=None` leaves the
        cup alone, so the piece turns with the arm.
        """
        self._begin()
        self._require_clearance("drop")
        return self._drop((float(x), float(y), float(height)), angle)

    def carry(self, start: Position, end: Position, *, angle: Optional[float] = None,
              facing: Optional[float] = None) -> MoveResult:
        """Pick up the piece whose top face is at `start`, and set it down with its top face at `end`.

        `grab()`, then `drop()`: `start` and `end` are (x, y, height), and the
        drop is checked before the grip, so a refusal means nothing moved.
        `angle` says which way the piece faces at the start, `facing` which way
        it is to face at the end. A turn bigger than the cup's 90 degrees from
        centre is shared: the cup turns half of it the other way on its way over
        the piece, and the other half after the lift. The piece hangs off the
        cup by however far the grip missed its centre, read before the suction,
        and turns with the cup: the cup comes down that miss, turned as the
        piece has turned, off `end`, so that the piece's centre lands on it.
        """
        self._begin()
        self._require_clearance("carry")
        start, end = _point(start), _point(end)
        servo, end_servo = self._carry_servos(start, angle, end, facing)
        start_desk, end_desk = self._space.to_desk(start), self._space.to_desk(end)
        problem = self._visit_problem(end_desk, self._travel_height(end_desk, start_desk[2]), end_servo)
        if problem:
            return self._result(MoveStatus.REFUSED, f"it could not be set down there: {problem}")
        result = self._grab(start, angle, servo)
        if not result:
            return result
        return self._drop(end, facing, is_miss_carried=True)

    def home(self) -> MoveResult:
        """Back to the reset pose, by way of a move: from low down, it climbs first."""
        self._begin()
        return self._move(self._spaces.arm.to_desk(home_position()), MoveType.FAST)

    def set_suction(self, is_on: bool) -> MoveResult:
        """Suction on or off by hand, and waited for -- for jobs `grab()` and `drop()` do not fit.

        The library then does not know what the cup holds, so the clearance
        protects the cup alone.
        """
        self._begin()
        self._require_connection()
        self._held = None
        stopped = self._switch_suction(is_on)
        return stopped if stopped is not None else self._result(MoveStatus.DONE)

    def stop(self) -> None:
        """Stop what is running: a leg is held where it has got to, a wait is cut short.

        Safe from a signal handler. The next call goes ahead as normal.
        """
        self._abort.set()

    def stop_on_ctrl_c(self) -> None:
        """Make Ctrl-C hold the arm where it is for the rest of the run. A second one quits."""
        def on_interrupt(signum, frame) -> None:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            print("\n  holding the arm where it is; Ctrl-C again to quit", file=sys.stderr)
            self._is_frozen = True
            self.stop()
        signal.signal(signal.SIGINT, on_interrupt)

    # --- asking -----------------------------------------------------------

    def can_reach(self, x: float, y: float, z: float) -> bool:
        """Whether the cup can be sent to (x, y, z). Arithmetic only: works before `connect()`."""
        point = self._space.to_desk((float(x), float(y), float(z)))
        return self._planner.problem_with(point) is None

    def get_position(self) -> Position:
        """Where the arm is, as its servos measure it, in the caller's space: the mean of their reads once steady.
        After a precise move, and nothing sent since, the mean it ended on."""
        self._require_connection()
        pulses = self._settled or self._read_steady()
        if pulses is None:
            raise ProtocolError("the servo bus did not answer a position read")
        return self._space.from_desk(self._spaces.arm.to_desk(_position_of(pulses)))

    # --- the one way anything moves ---------------------------------------

    def _move(self, target: Position, move_type: MoveType,
              servo: Optional[float] = None) -> MoveResult:
        """Plan the whole path to a desk-space target, check it, and drive it leg by leg."""
        self._require_connection()
        problem = nozzle.problem_with(servo)
        if problem:
            return self._result(MoveStatus.REFUSED, problem)
        start = self._desk_position()
        try:
            path = self._planner.plan(start, target, self._crossing_height())
        except Refused as reason:
            return self._result(MoveStatus.REFUSED, str(reason))
        turning_leg = _first_sideways(start, path) if servo is not None else None
        speed = MOVE_FAST_MM_S if move_type is MoveType.FAST else MOVE_SLOW_MM_S
        for index, point in enumerate(path):
            failed = self._drive(self._spaces.arm.from_desk(point), speed,
                                 servo if index == turning_leg else None)
            if failed is not None:
                return failed
        if move_type is MoveType.PRECISE:
            return self._land(self._spaces.arm.from_desk(target))
        return self._result(MoveStatus.DONE)

    def _land(self, target: Position) -> MoveResult:
        """Read until the servos are steady (`_steady()`): the arm is then taken to be at `target`, and their mean
        is where it is (`get_position()`)."""
        pulses = self._steady()
        if self._abort.is_set():
            return self._result(MoveStatus.STOPPED)
        if pulses is None:
            return self._result(MoveStatus.DONE, "the servos did not answer a position read")
        self._position, self._settled = target, pulses
        return self._result(MoveStatus.DONE)

    def _steady(self) -> Optional[Pulses]:
        """The servos read until they have stopped drifting (PRECISE_STEADY_* above): the mean of their last
        2 x PRECISE_STEADY_HALF_S of reads; at PRECISE_SETTLE_MAX_S after the travel, however they are. None if none
        answered, or stop() was called."""
        reads: list = []
        half = PRECISE_STEADY_HALF_S
        while not self._abort.is_set():
            pulses = self._protocol.read_pulses()
            now = time.monotonic()
            if pulses is not None:
                reads.append((now, pulses))
            recent = [read for when, read in reads if when > now - half]
            before = [read for when, read in reads if now - 2 * half < when <= now - half]
            since = now - self._travel_ended
            if since >= PRECISE_STEADY_FROM_S and len(recent) >= 3 and len(before) >= 3 and all(
                    abs(late - early) <= PRECISE_STEADY_PULSES for late, early in zip(_mean(recent), _mean(before))):
                return _mean(before + recent)
            if since >= PRECISE_SETTLE_MAX_S:
                return _mean(before + recent) if before + recent else (reads[-1][1] if reads else None)
        return None

    def _read_steady(self) -> Optional[Pulses]:
        """Where the servos are, read now: steady (`_steady()`) soon after a travel; from AT_REST_S after it, the
        mean of PRECISE_STEADY_HALF_S of reads. None if none answered."""
        if time.monotonic() - self._travel_ended < AT_REST_S:
            return self._steady()
        reads, started = [], time.monotonic()
        while not reads or time.monotonic() - started < PRECISE_STEADY_HALF_S:
            pulses = self._protocol.read_pulses()
            if pulses is not None:
                reads.append(pulses)
            elif time.monotonic() - started >= PRECISE_SETTLE_MAX_S:
                return None
        return _mean(reads)

    def _drive(self, point: Position, speed_mm_s: float,
               servo: Optional[float]) -> Optional[MoveResult]:
        """One leg, in arm space: one command, waited out. A result only if it went wrong."""
        if self._abort.is_set():
            return self._result(MoveStatus.STOPPED)
        distance = math.dist(self._position, point)
        move_ms = _move_ms(distance, speed_mm_s) if distance >= MOVE_RESOLUTION_MM else 0
        turn_ms = nozzle.turn_ms(self._servo_deg, servo) if servo is not None else 0
        if turn_ms:
            self._servo_deg = servo
        if move_ms:
            turn = (servo, turn_ms) if turn_ms else None
            if not self._protocol.set_position(point, move_ms, turn):
                return self._result(MoveStatus.REFUSED, "the board refused ({:.0f}, {:.0f}, "
                                    "{:.0f})".format(*point))
            self._position = point
        elif turn_ms:
            self._protocol.set_nozzle_angle(servo, turn_ms)
        if not self._wait(max(move_ms, turn_ms + nozzle.NOZZLE_SETTLE_MS if turn_ms else 0)):
            return self._halt()
        if move_ms or turn_ms:
            self._travel_ended, self._settled = time.monotonic(), None
        return None

    def _grab(self, point: Position, angle: Optional[float], servo: float) -> MoveResult:
        """grab(), the cup turned to `servo` for the grip."""
        if self._is_suction_on:
            stopped = self._switch_suction(False)
            if stopped is not None:
                return stopped
        self._held = None
        target = self._space.to_desk(point)
        travel = self._travel_height(target, target[2])
        result = self._visit(target, travel, servo)
        if not result:
            return result
        landed = self._landed()                     # before the pump: the seal can pull the cup further off
        stopped = self._switch_suction(True)
        if stopped is not None:
            return stopped
        self._held = self._hold(point, angle, landed)
        return self._move((target[0], target[1], travel), MoveType.SLOW)

    def _drop(self, point: Position, angle: Optional[float], is_miss_carried: bool = False) -> MoveResult:
        """drop(); carry()'s also brings the cup down off `point` by the grip's miss (`_release_at()`)."""
        target = self._space.to_desk(point)
        if is_miss_carried:
            cup, servo = self._release_at(point, angle)
        else:
            cup, servo = target, self._servo_for(point, angle) if angle is not None else None
        travel = self._travel_height(target, self._held.height if self._held else target[2])
        result = self._visit(cup, travel, servo)
        if not result:
            return result
        stopped = self._switch_suction(False)
        if stopped is not None:
            return stopped
        self._held = None
        return self._move((target[0], target[1], travel), MoveType.SLOW)

    def _visit(self, target: Position, travel: float, servo: Optional[float]) -> MoveResult:
        """Slowly up to `travel` if below it, fast across, slowly down onto `target` and read there."""
        problem = self._visit_problem(target, travel, servo)
        if problem:
            return self._result(MoveStatus.REFUSED, problem)
        here = self._desk_position()
        legs = [((target[0], target[1], travel), MoveType.FAST, servo), (target, MoveType.PRECISE, None)]
        if here[2] < travel:
            legs.insert(0, ((here[0], here[1], travel), MoveType.SLOW, None))
        for point, move_type, leg_servo in legs:
            result = self._move(point, move_type, leg_servo)
            if not result:
                return result
        return result

    def _visit_problem(self, target: Position, travel: float, servo: Optional[float]) -> Optional[str]:
        return (self._planner.problem_with((target[0], target[1], travel)) or self._planner.problem_with(target)
                or nozzle.problem_with(servo))

    def _halt(self) -> MoveResult:
        """stop() landed during a leg: send the arm to where it has got to, which holds it."""
        here = self._protocol.read_position()
        if here is None:
            return self._result(MoveStatus.STOPPED, "no position read, so the leg ran on")
        self._protocol.set_position(here, MOVE_MIN_MS)
        self._position = here
        self._travel_ended, self._settled = time.monotonic() + MOVE_MIN_MS / 1000.0, None
        return self._result(MoveStatus.STOPPED)

    def _switch_suction(self, is_on: bool) -> Optional[MoveResult]:
        """Pump on or off, waited out. A result only if stop() cut it short."""
        if self._abort.is_set():
            return self._result(MoveStatus.STOPPED)
        before = self._protocol.read_position() if IS_TRACED else None
        self._protocol.set_suction(is_on)
        self._is_suction_on = is_on
        is_waited = self._wait(SUCTION_GRAB_MS if is_on else SUCTION_RELEASE_MS)
        if before is not None:
            _trace_cup_pull(is_on, before, self._protocol.read_position())
        return None if is_waited else self._result(MoveStatus.STOPPED)

    # --- heights and angles -----------------------------------------------

    def _travel_height(self, target: Position, carried: float) -> float:
        """Where grab() and drop() cross the desk: clearance, plus the target's
        height or the carried piece's, whichever is higher, plus 5 mm."""
        return self._clearance_mm + max(target[2], carried) + CLEARANCE_MARGIN_MM

    def _crossing_height(self) -> Optional[float]:
        """How high anything travelling sideways must be, or None with no clearance to keep."""
        if not self._clearance_mm or not self._spaces.is_desk_known:
            return None
        carried = self._held.height if self._held else 0.0
        return self._clearance_mm + carried + CLEARANCE_MARGIN_MM

    def _servo_for(self, point: Position, angle: float) -> float:
        """The cup angle that turns the held piece -- or the empty cup -- to face `angle`."""
        offset = self._held.offset if self._held else 0.0
        return nozzle.servo_for(*self._to_arm(point, angle), offset)

    def _landed(self) -> Position:
        """Where the last precise move came down, arm space: the servos' steady mean; where it was sent if they
        did not answer."""
        return _position_of(self._settled) if self._settled else self._position

    def _hold(self, point: Position, angle: Optional[float], landed: Position) -> _Held:
        """The piece whose top is at `point`, just gripped by the cup at `landed`: how it sits under the cup.
        Its facing is taken as 0 off the cup's when not given."""
        centre = self._position
        cup = geometry.bearing_degrees(landed[0], landed[1]) + self._servo_deg
        offset = 0.0 if angle is None else nozzle.offset_of(landed, self._to_arm(point, angle)[1], self._servo_deg)
        miss = (landed[0] - centre[0], landed[1] - centre[1])
        if IS_TRACED:
            trace_note(f"grip missed the centre by x {miss[0]:+.1f}, y {miss[1]:+.1f} mm (arm)")
        return _Held(self._space.to_desk(point)[2], offset, cup, miss)

    def _release_at(self, point: Position, angle: Optional[float]) -> Tuple[Position, Optional[float]]:
        """Where the cup comes down to set the held piece's centre at `point`, desk space, and the cup's angle
        there: None leaves it alone. The grip's miss, turned as far as the piece has turned since."""
        target = self._spaces.arm.from_desk(self._space.to_desk(point))
        facing = self._to_arm(point, angle)[1] if angle is not None else None
        if not self._held:
            return self._spaces.arm.to_desk(target), None if facing is None else nozzle.servo_for(target, facing)
        cup = target
        for _ in range(2):                          # left alone, the cup turns with the arm's bearing, which moves
            turned = (facing - self._held.offset if facing is not None
                      else geometry.bearing_degrees(cup[0], cup[1]) + self._servo_deg)
            cup = _turned_off(target, self._held.miss, turned - self._held.cup)
        servo = None if facing is None else nozzle.servo_for(cup, facing, self._held.offset)
        return self._spaces.arm.to_desk(cup), servo

    def _carry_servos(self, start: Position, angle: Optional[float], end: Position,
                      facing: Optional[float]) -> Tuple[float, Optional[float]]:
        """The cup's angle for the grip, and at the end -- None there to leave it alone."""
        if facing is None:
            return 0.0, None
        if angle is None:
            return 0.0, nozzle.servo_for(*self._to_arm(end, facing))
        start_arm, start_facing = self._to_arm(start, angle)
        end_arm, end_facing = self._to_arm(end, facing)
        servo = nozzle.grip_servo(start_arm, start_facing, end_arm, end_facing)
        return servo, nozzle.servo_for(end_arm, end_facing, nozzle.offset_of(start_arm, start_facing, servo))

    def _to_arm(self, point: Position, angle: float) -> Tuple[Position, float]:
        """A point and a direction in the caller's space, as the arm has them."""
        desk_point = self._space.to_desk(point)
        desk_angle = angle_to_desk(self._space, point, angle)
        return (self._spaces.arm.from_desk(desk_point),
                angle_from_desk(self._spaces.arm, desk_point, desk_angle))

    # --- bookkeeping ------------------------------------------------------

    def _begin(self) -> None:
        """A new call re-arms stop() -- unless Ctrl-C has frozen the arm for the run."""
        if not self._is_frozen:
            self._abort.clear()

    def _wait(self, milliseconds: float) -> bool:
        """Sleep, unless stop() is called. True if it slept it out."""
        return not self._abort.wait(max(0.0, milliseconds) / 1000.0)

    def _desk_position(self) -> Position:
        return self._spaces.arm.to_desk(self._position)

    def _result(self, status: MoveStatus, detail: str = "") -> MoveResult:
        return MoveResult(status, self._space.from_desk(self._desk_position()), detail)

    def _require_connection(self) -> None:
        if not self._is_connected:
            raise NotConnectedError("connect() first")

    def _require_clearance(self, call: str) -> None:
        self._require_connection()
        if not self._clearance_mm:
            raise RuntimeError(f"{call}() needs a clearance: call set_clearance() once, with the "
                               f"height above the desk below which nothing may travel sideways")


def home_position() -> Position:
    """The reset pose, where every connect leaves the arm, in arm space."""
    return raw.true_position(geometry.HOME)


def _point(point: Position) -> Position:
    return (float(point[0]), float(point[1]), float(point[2]))


def _position_of(pulses: Pulses) -> Position:
    """The cup's true position, arm space, for the servos at `pulses`."""
    return raw.true_position(geometry.forward(geometry.pulse_to_deg(pulses)))


def _turned_off(point: Position, miss: Tuple[float, float], turn_deg: float) -> Position:
    """`point` moved by `miss`, turned `turn_deg` from +x towards +y."""
    turn = math.radians(turn_deg)
    return (point[0] + miss[0] * math.cos(turn) - miss[1] * math.sin(turn),
            point[1] + miss[0] * math.sin(turn) + miss[1] * math.cos(turn), point[2])


def _mean(reads) -> Pulses:
    return tuple(sum(read[servo] for read in reads) / len(reads) for servo in range(3))


def _move_ms(distance_mm: float, speed_mm_s: float) -> int:
    return int(min(max(distance_mm / speed_mm_s * 1000.0, MOVE_MIN_MS), MOVE_MAX_MS))


def _first_sideways(start: Position, path) -> int:
    """The first leg that crosses the desk, where a turn belongs; the first leg if none does."""
    cursor = start
    for index, point in enumerate(path):
        if math.hypot(point[0] - cursor[0], point[1] - cursor[1]) >= CROSSING_MM:
            return index
        cursor = point
    return 0


def _trace_cup_pull(is_on: bool, before: Position, after: Optional[Position]) -> None:
    """Under MAXARM_TRACE: how far the tip moved across a suction switch.

    The only sign this firmware gives that a grip sealed: the cup is a bellows,
    and contracts only against a seal.
    """
    if after is None:
        return
    trace_note(f"suction {'on ' if is_on else 'off'}: tip dz {after[2] - before[2]:+.0f} mm, "
               f"moved {math.dist(before, after):.0f} mm")
