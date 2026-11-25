#!/usr/bin/env python3
"""`MaxArm`: move, grab and drop, in whichever coordinates the caller works in.

    arm = MaxArm()
    arm.connect()                         # the board resets, and homes the arm
    arm.use_coordinates(Space.CAMERA)     # once: a point is now a pixel and a height
    arm.set_clearance(14.0)               # once: grab() and drop() need it
    arm.grab(812.0, 640.0, 4.0, angle=12.0)
    arm.drop(1020.0, 700.0, 5.0, angle=0.0)
    arm.carry((812.0, 640.0, 4.0), (1020.0, 700.0, 5.0), angle=12.0, facing=180.0)   # both: any turn
    arm.down_to(900.0, 660.0, 4.0)        # the cup down to exactly there, precisely, and stay there

Every motion -- `move()`, `down_to()`, `grab()`, `drop()`, `carry()`, `home()` -- goes through
`_move()`, which plans and checks the whole path (`path.py`) before anything
is sent, so a refusal means nothing moved. A landing's last turn and descent
(`landing.py`) are checked the same way before the grab or drop starts. Each
leg is one command, given the time its speed implies, and waited out. Nothing
is read back on the way.

Positions in and out are true, where the cup really is; the firmware's own
are corrected at the board (`raw.py`).
"""

import math
import signal
import sys
import threading
import time
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union

from . import geometry, landing, nozzle, raw, seal
from .geometry import Position
from .path import CLEARANCE_MARGIN_MM, CROSSING_MM, Planner, Refused
from .protocol import IS_TRACED, ProtocolError, ReplProtocol, trace_note
from .seal import SealState
from .spaces import CONFIG_DIR, Space, Spaces, angle_from_desk, angle_to_desk
from .transport import DEVICE, SerialTransport
from .tunable import tunable

# About a second to cross the desk.
MOVE_FAST_MM_S = tunable("MOVE_FAST_MM_S", 200.0)
# Slow enough not to yank a heavy piece; every move onto the desk in grab() and drop().
MOVE_SLOW_MM_S = tunable("MOVE_SLOW_MM_S", 10.0)
# Straight up: off the desk after a grip or a release, and to the travel height before a landing. Faster than
# MOVE_SLOW_MM_S: going up, nothing is landed on.
MOVE_LIFT_MM_S = 30.0
# A few millimeters at travel speed would ask the servos for a jerk at full speed.
MOVE_MIN_MS = 150
# The command carries two decimals: a move shorter than this is the same command.
# Anything longer is sent, so the arm's own response to tiny steps can be seen.
MOVE_RESOLUTION_MM = 0.01
# The servos take a move's duration as a 16-bit field, specified to 30 s. A slow
# move longer than that goes faster rather than being garbled.
MOVE_MAX_MS = 30000

# A grip and a release are waits, the servos and the supply read all through them (`seal.py`). The board's own
# release vents the cup for 1 s and then shuts the valve, the bellows still springing back: a syringe pulled, a soft
# re-grip of the piece, which SUCTION_RELEASE_MS waits out before the lift. Where the board's vent
# can be held (`ReplProtocol.has_vent`) a drop holds it open itself, through the wait and the lift (`_let_go()`):
# no closed cup near the piece, and the wait, SUCTION_VENTED_MS, is for the bellows to let the piece down.
SUCTION_GRAB_MS = tunable("SUCTION_GRAB_MS", 1500.0)
SUCTION_RELEASE_MS = tunable("SUCTION_RELEASE_MS", 3000.0)
SUCTION_VENTED_MS = 500.0

# grab()'s and carry()'s `retry_deeper_mm`: a grip that does not seal is let go and landed again `retry_step_mm`
# deeper, RETRY_STEP_MM unless the caller says, up to the caller's limit, which may be at most RETRY_DEEPER_MAX_MM.
# A step under RETRY_STEP_MIN_MM is refused: the servos would land the same, and the tries would be many.
RETRY_STEP_MM = 0.5
# Before a grip the cup's height is read until it has stopped creeping down: two GRIP_SETTLE_HALF_S of reads within
# GRIP_SETTLED_MM of each other, at most GRIP_SETTLE_MAX_S. A late creep, up to ~0.7 mm after a landing reads steady,
# would otherwise count as the seal pulling the cup down (`seal.PULLED_MM`).
GRIP_SETTLE_HALF_S = 0.5
GRIP_SETTLED_MM = 0.1
GRIP_SETTLE_MAX_S = 1.5
RETRY_STEP_MIN_MM = 0.1
RETRY_DEEPER_MAX_MM = 2.0

# Where the servos are, from their reads, which flicker by a pulse or more: once
# they have stopped drifting -- from PRECISE_STEADY_FROM_S after the travel, the
# mean of the last PRECISE_STEADY_HALF_S against that of the one before it, every
# servo within PRECISE_STEADY_PULSES -- the mean of both. At
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
    NOT_GRIPPED = "not_gripped"     # grab() or carry() with retry_deeper_mm: no seal down to the limit; let go, lifted


@dataclass(frozen=True)
class MoveResult:
    """How a call ended. Falsy unless it did everything it was asked."""

    status: MoveStatus
    position: Position       # where the arm was last sent, in the caller's space
    detail: str = ""
    # grab() and carry(): how much deeper than asked the cup went for its grip, mm -- 0.0 on the first try, the
    # sealed try's otherwise, the last try's if none sealed. None for the other calls.
    deeper_mm: Optional[float] = None

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
    miss: Tuple[float, float]  # where the cup came down, less the piece's center; arm mm


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
        self._supply_off: Optional[float] = None       # the supply where the arm stands, the pump off (`seal.py`)
        self._rest_z: Optional[float] = None           # and the cup's height read with it
        self._seal: Optional[Tuple[SealState, Optional[float], Optional[float]]] = None   # since the pump went on
        self._vent_over_at = 0.0                       # time.monotonic() from when the vent valve draws no more

    # --- session ----------------------------------------------------------

    def connect(self) -> None:
        """Open the port and check the board can drive the arm.

        Opening the port resets the board, which homes the arm and centers the
        cup. Raises `BoardNotReadyError` naming the remedy. A port still open
        from before -- a connect that failed, a board that stopped answering --
        is let go first, and the port is let go again if this connect fails:
        a board plugged in again then takes the first ttyUSB, and the next
        connect finds it (`transport.py`).
        """
        self.disconnect()
        try:
            self._protocol.connect()
            self._protocol.check_board()
        except Exception:
            self.disconnect()
            raise
        self._position = home_position()     # where the board's main.py sends the arm on every reset
        self._servo_deg, self._is_suction_on, self._held = 0.0, False, None
        self._is_connected, self._is_frozen = True, False
        self._travel_ended, self._settled = time.monotonic(), None
        self._supply_off, self._rest_z, self._seal = None, None, None
        self._vent_over_at = 0.0                # the reset shut the vent
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
        face that way, degrees from the space's +x toward its +y; the turn
        rides on the leg that crosses the desk, and the move waits for it.
        """
        self._begin()
        point = (float(x), float(y), float(z))
        servo = self._servo_for(point, angle) if angle is not None else None
        return self._move(self._space.to_desk(point), move_type, servo)

    def grab(self, x: float, y: float, height: float, *,
             angle: Optional[float] = None, retry_deeper_mm: float = 0.0,
             retry_step_mm: float = RETRY_STEP_MM) -> MoveResult:
        """Pick up a piece, the cup brought down to (x, y, height) for the grip.

        `height` is where the cup is sent, exactly; nothing is added to it.
        Lets go of anything held, then: up to the travel height if below
        it, fast across to beside the piece with the cup centering itself, the
        base turned onto it, slowly straight down (`landing.py`) and read there
        as PRECISE is, suction on and waited for, back up. The travel height is `clearance + height + 5 mm`,
        so the piece hangs 5 mm above the clearance.

        `angle` says which way the piece faces; `drop()` turns it from there.

        `retry_deeper_mm`, 0 to RETRY_DEEPER_MAX_MM (2 mm), else ValueError:
        a grip whose seal reads LOOSE or UNCERTAIN (`get_seal()`) is let go,
        lifted to the travel height and landed again `retry_step_mm` deeper
        (0.5 mm unless given; RETRY_STEP_MIN_MM, 0.1, to 2 mm, else
        ValueError), down to `height - retry_deeper_mm`. If none seals, the
        cup is let go and lifted, and the result is NOT_GRIPPED. Either way the
        result's `deeper_mm` says how much deeper than `height` the cup went.
        At 0, the default, the grip is not checked. The seal is told right only
        against what does not move, the desk or a piece the cup presses onto it;
        it does not say whether a light piece came along (`seal.py`).
        """
        retry = _retry(retry_deeper_mm, retry_step_mm)
        self._begin()
        self._require_clearance("grab")
        return self._grab((float(x), float(y), float(height)), angle, 0.0, *retry)

    def drop(self, x: float, y: float, height: float, *,
             angle: Optional[float] = None) -> MoveResult:
        """Set the held piece down, the cup brought down to (x, y, height), facing `angle`.

        `grab()` in reverse: the piece is turned on the way across, and the
        release is waited out before the lift -- the cup's vent held open
        until it is up, where the board lets that be (`_let_go()`).
        The cup turns 90 degrees either way from where `grab()` centered it; a
        bigger turn is refused before anything moves. `angle=None` leaves the
        cup alone, so the piece turns with the arm.
        """
        self._begin()
        self._require_clearance("drop")
        return self._drop((float(x), float(y), float(height)), angle)

    def down_to(self, x: float, y: float, height: float) -> MoveResult:
        """Move the cup down to (x, y, height), as precisely as the arm lands, and stay there.

        A move and nothing more: `height` is where the cup is sent, exactly,
        never lower. The suction is left as it is, on or off, and so is
        anything held. It comes down the way `grab()` and `drop()` do
        (`landing.py`): up to the travel height if below it, fast across
        to beside the point, the base turned onto it, slowly straight down, and
        read there as PRECISE is. `get_position()` then says where the cup came
        down. The travel height is `clearance + height + 5 mm`; with no
        clearance set, 5 mm over the point. The cup is not turned: `grab()`,
        `drop()` and `carry()` do that.
        """
        self._begin()
        self._require_connection()
        target = self._space.to_desk((float(x), float(y), float(height)))
        return self._visit(target, self._travel_height(target, self._held.height if self._held else target[2]), None)

    def carry(self, start: Position, end: Position, *, angle: Optional[float] = None,
              facing: Optional[float] = None, retry_deeper_mm: float = 0.0,
              retry_step_mm: float = RETRY_STEP_MM) -> MoveResult:
        """Pick up a piece, the cup brought down to `start`, and set it down, the cup brought down to `end`.

        `grab()`, then `drop()`: `start` and `end` are (x, y, height), and the
        drop is checked before the grip, so a refusal means nothing moved.
        `angle` says which way the piece faces at the start, `facing` which way
        it is to face at the end. A turn bigger than the cup's 90 degrees from
        center is shared: the cup turns half of it the other way on its way over
        the piece, and the other half after the lift. The piece hangs off the
        cup by however far the grip missed its center, read before the suction,
        and turns with the cup: the cup comes down that miss, turned as the
        piece has turned, off `end`, so that the piece's center lands on it.

        `retry_deeper_mm` and `retry_step_mm` are `grab()`'s, for the grip; nothing is set down
        after NOT_GRIPPED. The result's `deeper_mm` is the grip's.
        """
        retry = _retry(retry_deeper_mm, retry_step_mm)
        self._begin()
        self._require_clearance("carry")
        start, end = _point(start), _point(end)
        servo, end_servo = self._carry_servos(start, angle, end, facing)
        start_desk, end_desk = self._space.to_desk(start), self._space.to_desk(end)
        problem = self._visit_problem(end_desk, self._travel_height(end_desk, start_desk[2]), end_servo)
        if problem:
            return self._result(MoveStatus.REFUSED, f"it could not be set down there: {problem}")
        result = self._grab(start, angle, servo, *retry, is_seal_wanted=bool(retry[0]))
        if not result:
            return result
        dropped = self._drop(end, facing, is_miss_carried=True)
        detail = "; ".join(text for text in (result.detail, dropped.detail) if text)
        return replace(dropped, detail=detail, deeper_mm=result.deeper_mm)

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
        if stopped is not None:
            return stopped
        return self._result(MoveStatus.DONE, seal.describe(*self._seal) if self._seal else "")

    def get_seal(self) -> Optional[SealState]:
        """Whether the cup sealed when the pump last went on: by how far the pump pulled the supply down, and
        whether the seal pulled the arm down (`seal.py`). None while the pump is off, and once the arm has moved.
        Told right against what does not move, the desk; not whether a light piece is held."""
        return self._seal[0] if self._seal else None

    def get_seal_measures(self) -> Tuple[Optional[float], Optional[float]]:
        """What `get_seal()` was judged by: the pump's dip, in the supply pin's counts, and how far the cup pulled
        the arm down, mm. None for each not read, or as there."""
        return (self._seal[1], self._seal[2]) if self._seal else (None, None)

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
              servo: Optional[float] = None, speed_mm_s: Optional[float] = None) -> MoveResult:
        """Plan the whole path to a desk-space target, check it, and drive it leg by leg: at `speed_mm_s`, or at
        the move type's own."""
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
        speed = speed_mm_s or (MOVE_FAST_MM_S if move_type is MoveType.FAST else MOVE_SLOW_MM_S)
        for index, point in enumerate(path):
            failed = self._drive(self._spaces.arm.from_desk(point), speed,
                                 servo if index == turning_leg else None)
            if failed is not None:
                return failed
        if move_type is MoveType.PRECISE:
            return self._land(self._spaces.arm.from_desk(target))
        return self._result(MoveStatus.DONE)

    def _lift(self, target: Position) -> MoveResult:
        return self._move(target, MoveType.SLOW, speed_mm_s=MOVE_LIFT_MM_S)

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
            pulses, supply = self._protocol.read_pulses_and_supply()
            now = time.monotonic()
            if pulses is not None:
                reads.append((now, pulses, supply))
            recent = [read for read in reads if read[0] > now - half]
            before = [read for read in reads if now - 2 * half < read[0] <= now - half]
            since = now - self._travel_ended
            if since >= PRECISE_STEADY_FROM_S and len(recent) >= 3 and len(before) >= 3 and all(
                    abs(late - early) <= PRECISE_STEADY_PULSES
                    for late, early in zip(_mean(_pulses_of(recent)), _mean(_pulses_of(before)))):
                return self._kept(before + recent)
            if since >= PRECISE_SETTLE_MAX_S:
                return self._kept(before + recent) if before + recent else (reads[-1][1] if reads else None)
        return None

    def _read_steady(self) -> Optional[Pulses]:
        """Where the servos are, read now: steady (`_steady()`) soon after a travel; from AT_REST_S after it, the
        mean of PRECISE_STEADY_HALF_S of reads. None if none answered."""
        if time.monotonic() - self._travel_ended < AT_REST_S:
            return self._steady()
        reads, started = [], time.monotonic()
        while not reads or time.monotonic() - started < PRECISE_STEADY_HALF_S:
            pulses, supply = self._protocol.read_pulses_and_supply()
            if pulses is not None:
                reads.append((time.monotonic(), pulses, supply))
            elif time.monotonic() - started >= PRECISE_SETTLE_MAX_S:
                return None
        return self._kept(reads)

    def _kept(self, reads) -> Pulses:
        """The servos' mean over `reads`, each (when, pulses, supply); with the pump off, the supply and the cup's
        height over their last PRECISE_STEADY_HALF_S are kept as where the arm stands, to judge a seal against."""
        latest = [read for read in reads if read[0] >= reads[-1][0] - PRECISE_STEADY_HALF_S]
        if not self._is_suction_on and latest[0][0] >= self._vent_over_at:
            level = seal.level([supply for _, _, supply in latest])
            if level is not None:
                self._supply_off, self._rest_z = level, _position_of(_mean(_pulses_of(latest)))[2]
        return _mean(_pulses_of(reads))

    def _settle_height(self) -> None:
        """Read on until the cup's height has stopped creeping (GRIP_SETTLE_*), and keep the last reads as where
        the arm stands, pump off, for the seal to be judged against."""
        reads: list = []
        started, half = time.monotonic(), GRIP_SETTLE_HALF_S
        while not self._abort.is_set():
            pulses, supply = self._protocol.read_pulses_and_supply()
            now = time.monotonic()
            if pulses is not None:
                reads.append((now, pulses, supply))
            recent = [read for read in reads if read[0] > now - half]
            before = [read for read in reads if now - 2 * half < read[0] <= now - half]
            if len(recent) >= 3 and len(before) >= 3 and abs(
                    _position_of(_mean(_pulses_of(recent)))[2]
                    - _position_of(_mean(_pulses_of(before)))[2]) <= GRIP_SETTLED_MM:
                break
            if now - started >= GRIP_SETTLE_MAX_S:
                break
        if reads:
            self._kept(reads)

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
            self._supply_off, self._rest_z, self._seal = None, None, None
        return None

    def _grab(self, point: Position, angle: Optional[float], servo: float,
              retry_deeper_mm: float = 0.0, retry_step_mm: float = RETRY_STEP_MM,
              is_seal_wanted: bool = True) -> MoveResult:
        """grab(), the cup turned to `servo` for the grip; landed again `retry_step_mm` deeper while it does not
        seal, down to `retry_deeper_mm` under `point`. Not `is_seal_wanted`: nobody asks how the grip sealed, so
        the pump does not wait for the cup's height to stop creeping (`_settle_height()`), a second and a half."""
        if self._is_suction_on:
            stopped = self._switch_suction(False)
            if stopped is not None:
                return stopped
        self._held = None
        target = self._space.to_desk(point)
        travel = self._travel_height(target, target[2])
        if retry_deeper_mm:
            problem = self._visit_problem((target[0], target[1], target[2] - retry_deeper_mm), travel, servo)
            if problem:
                return self._result(MoveStatus.REFUSED, f"{retry_deeper_mm:g} mm deeper: {problem}")
        deeper, tries, judged = 0.0, 1, []
        while True:
            result = self._visit((target[0], target[1], target[2] - deeper), travel, servo)
            if not result:
                return replace(result, deeper_mm=deeper)
            landed = self._landed()                 # before the pump: the seal can pull the cup further off
            if is_seal_wanted:
                self._settle_height()
            stopped = self._switch_suction(True)
            if stopped is not None:
                return replace(stopped, deeper_mm=deeper)
            state = self._seal[0] if self._seal else SealState.UNKNOWN
            judged.append(_judged(deeper, self._seal))
            if not retry_deeper_mm or state in (SealState.SEALED, SealState.UNKNOWN):
                break
            stopped = self._switch_suction(False)
            if stopped is not None:
                return replace(stopped, deeper_mm=deeper)
            if deeper >= retry_deeper_mm:
                lifted = self._lift((target[0], target[1], travel))
                if not lifted:
                    return replace(lifted, deeper_mm=deeper)
                return replace(self._result(MoveStatus.NOT_GRIPPED,
                                            f"no seal down to {deeper:g} mm deeper than asked, {tries} tries: "
                                            f"{'; '.join(judged)}; let go"), deeper_mm=deeper)
            if IS_TRACED:
                trace_note(f"grip {state.value} at {deeper:g} mm deeper than asked; landing again deeper")
            deeper, tries = min(deeper + retry_step_mm, retry_deeper_mm), tries + 1
        self._held = self._hold(point, angle, landed)
        result = self._lift((target[0], target[1], travel))
        if result and tries > 1:
            result = replace(result, detail=f"sealed {deeper:g} mm deeper than asked, on try {tries}: "
                                            f"{'; '.join(judged)}")
        return replace(result, deeper_mm=deeper)

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
        return self._let_go((target[0], target[1], travel))

    def _let_go(self, lift: Position) -> MoveResult:
        """Pump off, the release waited out, and up to `lift`. Where the board's vent can be held
        (`has_vent`) it is open all the while, until the cup is up there, and not for the board's own second: with
        the vent shut, a cup still sealed to a smooth piece draws it up after itself as its bellows stretch."""
        is_vent_held = self._protocol.has_vent
        try:
            stopped = self._switch_suction(False, is_vent_held)
            if stopped is not None:
                return stopped
            self._held = None
            return self._lift(lift)
        finally:
            if is_vent_held:
                self._protocol.shut_vent()
                self._vent_over_at = time.monotonic() + seal.VENT_SHUT_S

    def _visit(self, target: Position, travel: float, servo: Optional[float]) -> MoveResult:
        """Up to `travel` if below it, then slowly down onto `target` as `landing.py` comes down, and read there
        as PRECISE is. The base is read until steady beside the target before its last turn -- the swing there
        spent -- and after it: one whose turn settled too far off to the side, with a nearer whole pulse to be had
        (`landing.is_worth_again()`), goes back beside the target and turns onto it again, once, from the same side
        (the base's play taken up the same way), aimed a pulse against its miss (`landing.again()`), before anything
        comes down."""
        problem = self._visit_problem(target, travel, servo)
        if problem:
            return self._result(MoveStatus.REFUSED, problem)
        here = self._desk_position()
        if here[2] < travel:
            result = self._lift((here[0], here[1], travel))
            if not result:
                return result
        way = self._landings(target, travel)[0]
        result = self._move(self._spaces.arm.to_desk(way.approach), MoveType.FAST, servo)
        if not result:
            return result
        for attempt in (1, 2):
            if self._steady() is None and self._abort.is_set():
                return self._result(MoveStatus.STOPPED)
            failed = self._drive(way.over, MOVE_FAST_MM_S, None)
            if failed is not None:
                return failed
            turned = self._steady()
            if turned is None and self._abort.is_set():
                return self._result(MoveStatus.STOPPED)
            if attempt == 2 or turned is None or not landing.is_worth_again(way, turned[0]):
                break
            if IS_TRACED:
                trace_note(f"landing: the base settled {turned[0] - way.base:+.1f} pulses off the target; "
                           f"turning onto it again from beside it")
            way = landing.again(way, turned[0])
            failed = self._drive(way.approach, MOVE_FAST_MM_S, None)
            if failed is not None:
                return failed
        failed = self._drive(way.down, MOVE_SLOW_MM_S, None)
        if failed is not None:
            return failed
        return self._land(self._spaces.arm.from_desk(target))

    def _visit_problem(self, target: Position, travel: float, servo: Optional[float]) -> Optional[str]:
        problem = (self._planner.problem_with((target[0], target[1], travel)) or self._planner.problem_with(target)
                   or nozzle.problem_with(servo))
        if problem is None and not self._landings(target, travel):
            problem = "the cup could not be brought down to it from either side"
        return problem

    def _landings(self, target: Position, travel: float) -> list:
        """The ways down to desk point `target` from `travel` (`landing.plans()`) whose every point
        the planner takes, the better first."""
        arm_space = self._spaces.arm
        arm_target = arm_space.from_desk(target)
        travel_z = arm_space.from_desk((target[0], target[1], travel))[2]
        return [way for way in landing.plans(arm_target, travel_z)
                if not any(self._planner.problem_with(arm_space.to_desk(point))
                           for point in (way.approach, way.over, way.down, way.target))]

    def _halt(self) -> MoveResult:
        """stop() landed during a leg: send the arm to where it has got to, which holds it."""
        here = self._protocol.read_position()
        if here is None:
            return self._result(MoveStatus.STOPPED, "no position read, so the leg ran on")
        self._protocol.set_position(here, MOVE_MIN_MS)
        self._position = here
        self._travel_ended, self._settled = time.monotonic() + MOVE_MIN_MS / 1000.0, None
        return self._result(MoveStatus.STOPPED)

    def _switch_suction(self, is_on: bool, is_vent_held: bool = False) -> Optional[MoveResult]:
        """Pump on or off, waited out. A result only if stop() cut it short. `is_vent_held`, with the pump going
        off: the vent is opened to stay open, and is the caller's to shut (`_let_go()`)."""
        if self._abort.is_set():
            return self._result(MoveStatus.STOPPED)
        before = self._protocol.read_position() if IS_TRACED else None
        if is_on and self._supply_off is None and self._protocol.has_supply and not self._is_suction_on:
            self._read_steady()                     # nothing read since the arm last moved: the level, pump off
        was_on = self._is_suction_on
        if is_vent_held:
            self._protocol.open_vent()
        else:
            self._protocol.set_suction(is_on)
        self._is_suction_on = is_on
        if is_on:
            is_waited, supply, height = self._read_for(SUCTION_GRAB_MS, seal.PUMP_START_S)
            if not was_on:
                pulled = self._rest_z - height if self._rest_z is not None and height is not None else None
                self._seal = seal.judge(self._supply_off, supply, pulled)
                if IS_TRACED:
                    trace_note(seal.describe(*self._seal))
        elif is_vent_held:
            # The open valve draws on the supply: nothing read until it is shut says where the supply stands.
            self._seal, self._supply_off, self._rest_z, self._vent_over_at = None, None, None, math.inf
            is_waited = self._wait(SUCTION_VENTED_MS)
        else:
            self._seal, self._vent_over_at = None, time.monotonic() + seal.VENT_S
            settled_from = max(seal.VENT_S, SUCTION_RELEASE_MS / 1000.0 - PRECISE_STEADY_HALF_S)
            is_waited, self._supply_off, self._rest_z = self._read_for(SUCTION_RELEASE_MS, settled_from)
        if before is not None:
            _trace_cup_pull(is_on, before, self._protocol.read_position())
        return None if is_waited else self._result(MoveStatus.STOPPED)

    def _read_for(self, milliseconds: float, from_s: float) -> Tuple[bool, Optional[float], Optional[float]]:
        """Wait, the servos and the supply read all the while: whether it waited it out, unless stop(), and the
        supply and the cup's height over the reads from `from_s` on. Just a wait on a board that cannot read its
        supply."""
        if not self._protocol.has_supply:
            return self._wait(milliseconds), None, None
        started = time.monotonic()
        deadline, supplies, heights = started + milliseconds / 1000.0, [], []
        while time.monotonic() < deadline:
            if self._abort.is_set():
                return False, None, None
            pulses, supply = self._protocol.read_pulses_and_supply()
            if time.monotonic() - started >= from_s:
                supplies.append(supply)
                if pulses is not None:
                    heights.append(_position_of(pulses)[2])
        return True, seal.level(supplies), seal.level(heights)

    # --- heights and angles -----------------------------------------------

    def _travel_height(self, target: Position, carried: float) -> float:
        """Where grab() and drop() cross the desk: clearance, plus the target's
        height or the carried piece's, whichever is higher, plus 5 mm."""
        return self._clearance_mm + max(target[2], carried) + CLEARANCE_MARGIN_MM

    def _crossing_height(self) -> Optional[float]:
        """How high anything traveling sideways must be, or None with no clearance to keep."""
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
        center = self._position
        cup = geometry.bearing_degrees(landed[0], landed[1]) + self._servo_deg
        offset = 0.0 if angle is None else nozzle.offset_of(landed, self._to_arm(point, angle)[1], self._servo_deg)
        miss = (landed[0] - center[0], landed[1] - center[1])
        if IS_TRACED:
            trace_note(f"grip missed the center by x {miss[0]:+.1f}, y {miss[1]:+.1f} mm (arm)")
        return _Held(self._space.to_desk(point)[2], offset, cup, miss)

    def _release_at(self, point: Position, angle: Optional[float]) -> Tuple[Position, Optional[float]]:
        """Where the cup comes down to set the held piece's center at `point`, desk space, and the cup's angle
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


def _judged(deeper: float, judged: Optional[tuple]) -> str:
    """One grip try, as the result's detail tells it: how deep, the seal, and what it was judged by."""
    if not judged:
        return f"{deeper:g} mm: not judged"
    state, dip, pulled = judged
    measures = ", ".join(text for text in (f"dip {dip:.1f}" if dip is not None else "",
                                           f"pulled {pulled:.2f} mm" if pulled is not None else "") if text)
    return f"{deeper:g} mm: {state.value}" + (f" ({measures})" if measures else "")


def _retry(deeper: float, step: float) -> Tuple[float, float]:
    """grab()'s and carry()'s `retry_deeper_mm` and `retry_step_mm`, checked."""
    deeper, step = float(deeper), float(step)
    if not 0.0 <= deeper <= RETRY_DEEPER_MAX_MM:
        raise ValueError(f"retry_deeper_mm goes from 0 to {RETRY_DEEPER_MAX_MM:g} mm, not {deeper:g}")
    if not RETRY_STEP_MIN_MM <= step <= RETRY_DEEPER_MAX_MM:
        raise ValueError(f"retry_step_mm goes from {RETRY_STEP_MIN_MM:g} to {RETRY_DEEPER_MAX_MM:g} mm, not {step:g}")
    return deeper, step


def _point(point: Position) -> Position:
    return (float(point[0]), float(point[1]), float(point[2]))


def _position_of(pulses: Pulses) -> Position:
    """The cup's true position, arm space, for the servos at `pulses`."""
    return raw.true_position(geometry.forward(geometry.pulse_to_deg(pulses)))


def _turned_off(point: Position, miss: Tuple[float, float], turn_deg: float) -> Position:
    """`point` moved by `miss`, turned `turn_deg` from +x toward +y."""
    turn = math.radians(turn_deg)
    return (point[0] + miss[0] * math.cos(turn) - miss[1] * math.sin(turn),
            point[1] + miss[0] * math.sin(turn) + miss[1] * math.cos(turn), point[2])


def _pulses_of(reads) -> list:
    return [pulses for _, pulses, _ in reads]


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
