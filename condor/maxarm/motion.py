#!/usr/bin/env python3
"""Driving a route, with the readback in the loop. Internal to the library.

`route.py` decides where the arm goes; this decides whether it got there. The
split matters because the first half is arithmetic and the second half is an
argument with a machine that lies: `set_position()` returns `True` while servo
2 sits silently clamped and the arm stands still. Nothing here is believed
without a readback.

Two ways a hop is driven, and only two:

  * **Flown** -- one command for the whole hop, watched while it travels.
    `route.py` has already checked where the arm's own interpolation will take
    it, so there is nothing to correct mid-flight and no reason to interrupt
    it. Reading the pose while it flies costs no commands and buys two things:
    a stall is noticed *during* the move, and `stop()` can halt the arm at once
    by retargeting it to where it currently stands.
  * **Stepped** -- 2 mm at a time with a readback after each. Only for a
    descent into the desk band, where what matters happens *between* waypoints.

Progress, not error, is what identifies a limit. Command-versus-readback error
needs calibration and varies across the envelope; "it stopped moving" does not.
"""

import math
import threading
import time
from typing import Callable, List, Optional

from . import geometry, tuning
from .protocol import BoardProtocol, trace_note
from .route import Hop, NoRoute, Position, Router
from .status import MoveResult, MoveStatus, Step

StepCallback = Callable[[Step], None]


class Mover:
    """Drives moves for one session, and tracks where the arm actually is."""

    def __init__(self, protocol: BoardProtocol, router: Router) -> None:
        self.protocol = protocol
        self.router = router
        self.position: Position = geometry.HOME_COMMAND
        # Measured travel since the current move began. What separates "the arm
        # is against something" from "the arm is not listening".
        self._moved_mm = 0.0

    def sync_position(self) -> Optional[Position]:
        """Adopt the arm's own idea of where it is. Call after connecting."""
        measured = self.protocol.read_position()
        if measured is None:
            measured = self.protocol.get_commanded_position()
        if measured is not None:
            self.position = measured
        return measured

    def move(self, request: Position, on_step: Optional[StepCallback] = None,
             abort: Optional[threading.Event] = None) -> MoveResult:
        """Plan and run one move. Never raises for a bad target."""
        target, status, detail = self.router.resolve(request)
        if status not in (MoveStatus.REACHED, MoveStatus.APPROXIMATED):
            return MoveResult(status, request, target, self.position, detail)
        try:
            hops = self.router.route(self.position, target)
        except NoRoute as error:
            return MoveResult(MoveStatus.NO_FLY, request, target, self.position, str(error))

        self._moved_mm = 0.0
        steps: List[Step] = []
        for index, hop in enumerate(hops):
            is_last = index == len(hops) - 1
            outcome = (self._walk(hop, steps, on_step, abort) if hop.is_stepped
                       else self._fly(hop, steps, on_step, abort, is_last))
            if outcome is not MoveStatus.REACHED:
                return MoveResult(outcome, request, target, self.position,
                                  f"stopped on the {hop.purpose} leg", steps)

        # The owner's "if you see that we missed a threshold, make minor moves
        # to adjust". Skipped after a stepped descent: the arm is a couple of
        # millimetres off the desk and nudging it is how the cup gets scraped.
        if not hops[-1].is_stepped:
            status = self._correct(target, status, steps, on_step, abort)
        return MoveResult(status, request, target, self.position, detail, steps)

    # --- one hop ----------------------------------------------------------

    def _fly(self, hop: Hop, steps: List[Step], on_step: Optional[StepCallback],
             abort: Optional[threading.Event], is_last: bool) -> MoveStatus:
        """One command for the whole hop, watched while it travels."""
        target = _safe_command(hop.position)
        start = self.position
        duration_ms = _duration_ms(_distance(start, target), tuning.TRAVEL_SPEED_MM_S)
        verdict = self.protocol.set_position(target, duration_ms)
        if not verdict:
            self._record(steps, on_step, Step(target, None, MoveStatus.BLOCKED))
            return MoveStatus.BLOCKED

        heading = _unit_vector(start, target)
        deadline = time.monotonic() + duration_ms / 1000.0
        previous, idle_polls = start, 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            time.sleep(min(tuning.POLL_MS / 1000.0, remaining))
            measured = self.protocol.read_position()
            if measured is None:
                continue
            self.position = measured
            self._record(steps, on_step, Step(target, measured, MoveStatus.REACHED))

            advance = sum((measured[i] - previous[i]) * heading[i] for i in range(3))
            self._moved_mm += max(advance, 0.0)
            previous = measured

            if abort is not None and abort.is_set():
                self._halt_here(measured)
                return MoveStatus.STOPPED
            # Still short of the target and no longer moving: something is in
            # the way, or nothing is listening. Either way, stop pushing.
            if _distance(measured, target) > tuning.TRACKING_TOLERANCE_MM:
                idle_polls = idle_polls + 1 if advance < tuning.NOISE_FLOOR_MM else 0
                if idle_polls >= tuning.IDLE_POLLS_BEFORE_BLOCKED:
                    self._halt_here(measured)
                    return self._motionless_status()

        if is_last:
            self._settle()
        if _distance(self.position, target) > tuning.TRACKING_TOLERANCE_MM:
            return self._motionless_status()
        return MoveStatus.REACHED

    def _walk(self, hop: Hop, steps: List[Step], on_step: Optional[StepCallback],
              abort: Optional[threading.Event]) -> MoveStatus:
        """A descent into the desk band: 2 mm at a time, measured after each.

        The first step sets this descent's normal droop -- the arm is still
        clear of the surface there, by construction, because the guard band
        starts above the floor. Every later step is judged against it, which is
        why no global drift constant has to be guessed anywhere in this
        library.
        """
        baseline: Optional[float] = None
        last_good = self.position
        idle_steps = 0

        for waypoint in _waypoints(self.position, hop.position, tuning.DESCENT_STEP_MM):
            if abort is not None and abort.is_set():
                return MoveStatus.STOPPED

            previous = self.position
            step = self._command(waypoint, tuning.DESCENT_SPEED_MM_S, tuning.SETTLE_MS // 4)
            self._record(steps, on_step, step)
            if step.measured is None or step.status is MoveStatus.BLOCKED:
                return MoveStatus.BLOCKED

            droop = waypoint[2] - step.measured[2]
            if baseline is None:
                baseline = droop
            elif droop - baseline > tuning.DESK_SAG_MM and self._is_contact_confirmed(
                    waypoint, baseline):
                # Readback is quantised to 1 mm with ~2.4 mm of noise while
                # real contact is about 4 mm, so one sample cannot tell them
                # apart -- hence the second look before backing off.
                if not self._retreat_to(last_good):
                    return MoveStatus.BLOCKED
                return MoveStatus.DESK_CONTACT

            advance = _distance(previous, step.measured)
            self._moved_mm += advance
            # The arm has a deadband -- 8.1 mm measured for a 10 mm command --
            # so a single short step means nothing and only a run of them does.
            idle_steps = idle_steps + 1 if advance < tuning.NOISE_FLOOR_MM else 0
            if idle_steps >= tuning.IDLE_POLLS_BEFORE_BLOCKED:
                return self._motionless_status()
            last_good = waypoint
        return MoveStatus.REACHED

    def _correct(self, target: Position, status: MoveStatus, steps: List[Step],
                 on_step: Optional[StepCallback],
                 abort: Optional[threading.Event]) -> MoveStatus:
        """Close the last couple of millimetres by aiming past the target.

        Re-commanding the same coordinate was measured to gain nothing -- as
        far as the arm is concerned it is already there -- so a correction has
        to ask for a move the servos will actually act on. Aiming past the
        target by the size of the miss does that: the commanded delta is then
        twice the error, comfortably clear of the 8 mm deadband.

        Which is also why it overshoots, and why one attempt is not enough. A
        small move fails in two opposite ways and each is read off the
        readback rather than guessed at:

          * **the arm moved and is now past the target** -- aim less far past,
            so halve the gain;
          * **the arm barely moved at all** -- the command was inside the
            deadband, so aim further past and double it.

        It stops when it is inside tolerance, when the gain collapses (the
        remaining error is under the readback's own noise and pushing at it
        only makes the arm twitch), or when the miss is too big to be an aim
        error in the first place.
        """
        gain: Optional[float] = tuning.CORRECTION_GAIN

        for attempt in range(1, tuning.CORRECTION_ATTEMPTS + 1):
            error = _distance(self.position, target)
            if error <= tuning.ARRIVAL_TOLERANCE_MM:
                break
            # Too far out to be an aim error: that is a blockage or a limit,
            # and the result says so rather than shoving at it.
            if error > tuning.MAX_CORRECTION_MM:
                break
            if abort is not None and abort.is_set():
                return MoveStatus.STOPPED

            nudge = tuple(target[i] + gain * (target[i] - self.position[i]) for i in range(3))
            if not self.router.is_in_envelope(nudge) or not self.router.zones.is_clear(nudge):
                break

            before = self.position
            step = self._command(nudge, tuning.DESCENT_SPEED_MM_S, tuning.SETTLE_MS)
            self._record(steps, on_step, step)
            if step.measured is None:
                break

            moved = _distance(before, self.position)
            landed = _distance(self.position, target)
            # One line per attempt, and only when one actually fires -- usually
            # none, at most four. Enough to tell "the nudge worked" from "the
            # nudge moved nothing, so something is in the way".
            trace_note(f"nudge {attempt}: {error:.1f} mm out, aimed {gain:.2f}x past, "
                       f"arm moved {moved:.1f} mm, now {landed:.1f} mm out")
            gain = _next_gain(gain, moved=moved, error=landed, previous_error=error)
            if gain is None:
                break
        return status

    # --- talking to the board ---------------------------------------------

    def _command(self, position: Position, speed_mm_s: float, settle_ms: int) -> Step:
        """One commanded increment, measured. Updates the tracked pose."""
        commanded = _safe_command(position)
        duration_ms = _duration_ms(_distance(self.position, commanded), speed_mm_s)
        verdict, measured = self.protocol.move_and_read(commanded, duration_ms, settle_ms)
        if measured is None and verdict:
            measured = self.protocol.read_position()    # a dropped read is transient
        if measured is not None:
            self.position = measured
        status = (MoveStatus.BLOCKED if not verdict or measured is None
                  else MoveStatus.REACHED)
        return Step(commanded, measured, status)

    def _is_contact_confirmed(self, waypoint: Position, baseline: float) -> bool:
        _, measured = self.protocol.move_and_read(
            _safe_command(waypoint), tuning.MIN_MOVE_MS, tuning.SETTLE_MS)
        if measured is None:
            return False
        self.position = measured
        return waypoint[2] - measured[2] - baseline > tuning.DESK_SAG_MM

    def _retreat_to(self, position: Position, attempts: int = 3) -> bool:
        """Back out to a pose that was good a moment ago; confirm z recovers.

        One step back is often not enough: if the cup is loaded against the
        desk, the previous pose is still pressing. So each failed attempt lifts
        a little further before retrying.
        """
        target = position
        for _ in range(attempts + 1):
            duration_ms = _duration_ms(_distance(self.position, target),
                                       tuning.DESCENT_SPEED_MM_S)
            verdict, measured = self.protocol.move_and_read(
                _safe_command(target), duration_ms, tuning.SETTLE_MS)
            if measured is not None:
                self.position = measured
                if verdict and target[2] - measured[2] <= tuning.FREE_SAG_MM:
                    return True
            target = (target[0], target[1],
                      min(target[2] + tuning.DESK_CLEARANCE_MM / 2.0, geometry.Z_COMMAND_MAX))
        return False

    def _halt_here(self, measured: Position) -> None:
        """Stop the arm where it stands by making that its target.

        The board has no stop command, but retargeting it to its own current
        pose amounts to one, and it takes effect in a servo time constant
        rather than at the end of whatever was queued.
        """
        self.protocol.set_position(_safe_command(measured), tuning.MIN_MOVE_MS)

    def _settle(self) -> None:
        settled = self.protocol.read_settled_position(tuning.SETTLE_READS,
                                                      tuning.SETTLE_MS / 1000.0)
        if settled is not None:
            self.position = settled

    def _motionless_status(self) -> MoveStatus:
        """Nothing moved. Whether that is a limit depends on the move so far.

        An arm that has not travelled a millimetre since the move began is not
        against anything -- it is not listening. One that moved and then
        stopped dead has met something, or a servo has hit its silent clamp.
        """
        return (MoveStatus.NOT_RESPONDING if self._moved_mm < tuning.NOISE_FLOOR_MM
                else MoveStatus.BLOCKED)

    def _record(self, steps: List[Step], on_step: Optional[StepCallback], step: Step) -> None:
        steps.append(step)
        if on_step is not None:
            on_step(step)


def _next_gain(gain: float, moved: float, error: float,
               previous_error: float) -> Optional[float]:
    """How far past the target the next nudge should aim. `None` means stop.

    **A nudge that produced no movement is the end of it.** The arm was told
    to move several millimetres and did not, which means something is in the
    way -- and a correction is the one place in this library where "in the
    way" is most likely to be the thing the cup is about to pick up. Aiming
    further past it would press harder into whatever that is.

    (An earlier version doubled the gain here, reasoning that the command had
    fallen inside the arm's deadband. That reasoning was wrong: 8.1 mm
    measured for a 10 mm command is 81% tracking, not a dead zone, so a small
    command does move the arm proportionally. The only thing that stops it
    dead is an obstruction, and this escalated into it three times.)

    Overshoot is the opposite case and is safe to work on: the arm moved, it
    went too far, so aim less far.
    """
    if moved < tuning.NOISE_FLOOR_MM:
        return None
    if error >= previous_error:
        if gain <= tuning.CORRECTION_GAIN_MIN:
            return None                         # oscillating inside the noise
        return max(gain / 2.0, tuning.CORRECTION_GAIN_MIN)
    return gain                                 # closing on it; leave it alone


def _duration_ms(span_mm: float, speed_mm_s: float) -> int:
    """How long to give the servos for a move of this length."""
    return max(tuning.MIN_MOVE_MS,
               min(int(span_mm / max(speed_mm_s, 1.0) * 1000.0), tuning.MAX_MOVE_MS))


def _safe_command(position: Position) -> Position:
    """Last-resort clamp on the two commands the firmware mishandles.

    Above z=225 `set_position()` pins the value and reports it back as if it
    had been honoured, and inside the 50 mm cylinder it refuses outright. The
    route avoids both; this makes it impossible to reach them by accident.
    """
    x, y, z = position
    z = min(z, geometry.Z_COMMAND_MAX)
    radius = math.hypot(x, y)
    if 0.0 < radius < geometry.BLIND_RADIUS:
        scale = geometry.BLIND_RADIUS / radius
        x, y = x * scale, y * scale
    return (round(x, 2), round(y, 2), round(z, 2))


def _waypoints(start: Position, target: Position, step_mm: float) -> List[Position]:
    span = _distance(start, target)
    if span <= step_mm:
        return [target]
    count = int(math.ceil(span / max(step_mm, 0.5)))
    points = [tuple(round(start[i] + (target[i] - start[i]) * n / count, 2) for i in range(3))
              for n in range(1, count + 1)]
    points[-1] = target
    return points


def _distance(a: Position, b: Position) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _unit_vector(origin: Position, target: Position) -> Position:
    span = _distance(origin, target)
    if span == 0.0:
        return (0.0, 0.0, 0.0)
    return tuple((target[i] - origin[i]) / span for i in range(3))
