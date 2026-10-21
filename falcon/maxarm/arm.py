#!/usr/bin/env python3
"""`MaxArm`: move, grab and drop, in whichever coordinates the caller works in.

    arm = MaxArm()
    arm.connect()                         # the board resets, and homes the arm
    arm.use_coordinates(Space.CAMERA)     # once: a point is now a pixel and a height
    arm.set_clearance(14.0)               # once: grab() and drop() need it
    arm.grab(812.0, 640.0, 4.0, angle=12.0)
    arm.drop(1020.0, 700.0, 5.0, angle=0.0)

Every motion -- `move()`, `grab()`, `drop()`, `home()` -- goes through
`_move()`, which plans and checks the whole path (`path.py`) before anything
is sent, so a refusal means nothing moved. Each leg is one command, given the
time its speed implies, and waited out. Nothing is read back on the way.

Heights in and out are true z, where the cup really is; the firmware's own
are corrected at the board (`raw.py`).
"""

import math
import signal
import sys
import threading
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

# Each servo has a deadband: it does not move for a target a few pulses away, and
# stops short of one further off, the same way every time from the same side. So
# sending the target plus what it fell short by lands it.
# Landed is every servo within this many pulses of the pulse the firmware sends
# for the target: the servos only take whole pulses, and the firmware truncates.
PRECISE_TOLERANCE_PULSES = 1.0
PRECISE_ROUNDS = 3
# A correction is never bigger than this on any axis: a cup held up by something
# solid must not be pushed into it much further.
PRECISE_CAP_MM = 3.0
# A round that does not take the worst servo's miss below this share of the last
# one ends it: something is in the way, or what is left is within the deadband.
PRECISE_PROGRESS = 0.8


class MoveType(Enum):
    FAST = "fast"            # straight there at MOVE_FAST_MM_S, no correction
    SLOW = "slow"            # the same at MOVE_SLOW_MM_S: nothing heavy gets yanked
    # Slow, then corrected from the servos' own reading until each is within a
    # pulse of its target: about a second more, for a fraction of the error.
    PRECISE = "precise"


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
        piece, suction on and waited for, slowly back up. The travel height is
        `clearance + height + 5 mm`, so the piece hangs 5 mm above the clearance.

        `angle` says which way the piece faces; `drop()` turns it from there.
        """
        self._begin()
        self._require_clearance("grab")
        if self._is_suction_on:
            stopped = self._switch_suction(False)
            if stopped is not None:
                return stopped
        self._held = None
        point = (float(x), float(y), float(height))
        target = self._space.to_desk(point)
        travel = self._travel_height(target)
        result = self._visit(target, travel, servo=0.0)
        if not result:
            return result
        stopped = self._switch_suction(True)
        if stopped is not None:
            return stopped
        self._held = _Held(target[2], self._piece_offset(point, angle))
        return self._move((target[0], target[1], travel), MoveType.SLOW)

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
        point = (float(x), float(y), float(height))
        target = self._space.to_desk(point)
        servo = self._servo_for(point, angle) if angle is not None else None
        travel = self._travel_height(target)
        result = self._visit(target, travel, servo)
        if not result:
            return result
        stopped = self._switch_suction(False)
        if stopped is not None:
            return stopped
        self._held = None
        return self._move((target[0], target[1], travel), MoveType.SLOW)

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
        """Where the arm is, as its servos measure it, in the caller's space."""
        self._require_connection()
        measured = self._protocol.read_settled_position()
        if measured is None:
            raise ProtocolError("the servo bus did not answer a position read")
        return self._space.from_desk(self._spaces.arm.to_desk(measured))

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
            return self._correct(self._spaces.arm.from_desk(target))
        return self._result(MoveStatus.DONE)

    def _correct(self, target: Position) -> MoveResult:
        """Send `target` (arm space) plus what the servos fell short by, until they land or stop gaining.

        The arm is then taken to be at `target`: what it was sent to last is
        target plus a correction, which says less about where it is.
        """
        wanted = [int(pulse) for pulse in geometry.deg_to_pulse(geometry.inverse(raw.raw_position(target)))]
        last_miss = math.inf
        for _ in range(PRECISE_ROUNDS):
            if self._abort.is_set():
                return self._result(MoveStatus.STOPPED)
            pulses = self._protocol.read_settled_pulses()
            if pulses is None:
                return self._result(MoveStatus.DONE, "the servos did not answer, so it was not corrected")
            miss = max(abs(goal - now) for goal, now in zip(wanted, pulses))
            if miss <= PRECISE_TOLERANCE_PULSES or miss > PRECISE_PROGRESS * last_miss:
                break
            here = raw.true_position(geometry.forward(geometry.pulse_to_deg(pulses)))
            aim = tuple(goal + max(-PRECISE_CAP_MM, min(PRECISE_CAP_MM, goal - now))
                        for goal, now in zip(target, here))
            if self._planner.problem_on_the_way(self._spaces.arm.to_desk(aim)):
                break
            failed = self._drive(aim, MOVE_SLOW_MM_S, None)
            if failed is not None:
                return failed
            last_miss = miss
        self._position = target
        return self._result(MoveStatus.DONE)

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
        return None

    def _visit(self, target: Position, travel: float, servo: Optional[float]) -> MoveResult:
        """Slowly up to `travel` if below it, fast across, slowly down onto `target`."""
        above = (target[0], target[1], travel)
        problem = (self._planner.problem_with(above) or self._planner.problem_with(target)
                   or nozzle.problem_with(servo))
        if problem:
            return self._result(MoveStatus.REFUSED, problem)
        here = self._desk_position()
        legs = [(above, MoveType.FAST, servo), (target, MoveType.SLOW, None)]
        if here[2] < travel:
            legs.insert(0, ((here[0], here[1], travel), MoveType.SLOW, None))
        for point, move_type, leg_servo in legs:
            result = self._move(point, move_type, leg_servo)
            if not result:
                return result
        return result

    def _halt(self) -> MoveResult:
        """stop() landed during a leg: send the arm to where it has got to, which holds it."""
        here = self._protocol.read_position()
        if here is None:
            return self._result(MoveStatus.STOPPED, "no position read, so the leg ran on")
        self._protocol.set_position(here, MOVE_MIN_MS)
        self._position = here
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

    def _travel_height(self, target: Position) -> float:
        """Where grab() and drop() cross the desk: clearance, plus the target's
        height or the held piece's, whichever is higher, plus 5 mm."""
        carried = self._held.height if self._held else target[2]
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

    def _piece_offset(self, point: Position, angle: Optional[float]) -> float:
        """How the piece just picked up sits under the cup; 0 when its facing is not given."""
        return 0.0 if angle is None else nozzle.offset_of(*self._to_arm(point, angle), self._servo_deg)

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
