#!/usr/bin/env python3
"""Endpoint reachability testing for the MaxArm, by small guarded steps.

Never jump at a target. Walk toward it in ~2 mm increments and read the pose
back after every increment, because the arm fails silently in three different
ways: set_position() returns None inside a 50 mm blind cylinder, returns False
when the IK throws, and — worst — returns True while set_servo_in_range() has
quietly clamped a joint. Near the envelope edge the clamp shows up as the
end effector sagging in Z, which is the case that scrubs the nozzle on the desk.

So Z is watched on every step: if it drifts past `z_drift_mm` the walk backs out
to the last good pose, which should also restore Z. Failure to restore means
something is physically stuck and the caller should stop.
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from maxarm_link import MaxArmLink, Position

# From work/PILOT-anaconda.md, whose "Observed Limits" are what the owner saw,
# NOT walls. Only three things below are real constraints; everything else runs
# until the arm itself refuses. Coding the observations as walls once clipped
# 40 mm of genuine reach and hid the true envelope.
#
# TESTING-ONLY limits. Both exist to keep the reach survey safe, not because the
# arm cannot go there. A pick-and-place library may legitimately lower Z_MIN to
# snug the nozzle onto a card, and X_MIN exists only because there is no desk to
# the left of the arm -- the left half is mirrored at analysis time.
X_MIN = 0.0
Z_MIN = 48.0

# FIRMWARE limit, and a real one: set_position() silently clamps z > 225 while
# reporting the commanded value back. Never command into that lie.
Z_MAX = 224.0

# Sanity bounds only, set beyond any reach the links allow (~291 mm). These are
# not limits; they stop runaway arithmetic, and the arm's refusal defines the edge.
X_MAX = 320.0
Y_MIN = -320.0
Y_MAX = 320.0
MIN_RADIUS = 50.0  # ESPMax refuses anything closer to the base axis than this

# ENFORCED no-fly square in X/Y. Inside it the suction cup or its hose can foul
# the robot's own square base. Applies at every Z, and moves may not cross it --
# see is_segment_clear() and ReachProbe.route_around().
NO_FLY_HALF = 70.0
# Planning keeps this much further out than the physical square. The arm lands
# within about 2.6 mm of what it is told, so aiming at the boundary itself puts
# it inside roughly half the time -- which is how a sweep once parked the arm in
# the zone and deadlocked, unable to move at all.
NO_FLY_MARGIN = 8.0
NO_FLY_PLANNING_HALF = NO_FLY_HALF + NO_FLY_MARGIN
# Outside the square's circumscribed circle (70 * sqrt(2) = 99 mm), so an arc at
# this radius clears the corners. Used when routing around the zone.
SAFE_ORBIT = 110.0

HOME = (0.0, -163.0, 212.0)
# The shoulder joint, from espmax's L0. Backing off after a stall heads here:
# it unloads the arm radially and vertically at once.
SHOULDER = (0.0, 0.0, 84.0)


class Outcome(Enum):
    REACHED = "reached"
    REFUSED = "refused"           # set_position() returned False
    BLIND_ZONE = "blind_zone"     # inside the 50 mm cylinder, returned None
    STALLED = "stalled"           # accepted, but the arm stopped advancing: the edge
    CLAMPED = "clamped"           # so far from the command that tracking is lost
    Z_SAG = "z_sag"               # drooped below baseline: extended too far, a real limit
    Z_DESK = "z_desk"             # drooped while commanded near the desk: possible contact
    OUT_OF_BOUNDS = "out_of_bounds"
    NO_FLY = "no_fly"             # target or path would cross the base exclusion square
    SKIPPED = "skipped"           # model says only our own fence is there; no motion spent
    READ_FAILED = "read_failed"   # bus servos did not report a position
    STUCK = "stuck"               # backed out but Z did not recover


@dataclass
class StepResult:
    commanded: Position
    measured: Optional[Position]
    verdict: Optional[bool]
    outcome: Outcome

    @property
    def is_good(self) -> bool:
        return self.outcome is Outcome.REACHED


@dataclass
class WalkResult:
    """What happened walking from a start pose toward a target.

    `last_measured` is the readback where the arm actually ended up, and is what
    a reach envelope should be built from. `last_good` is what we asked for --
    keep it only to see how far the two diverge.
    """
    target: Position
    outcome: Outcome
    last_good: Position
    last_measured: Optional[Position] = None
    steps: List[StepResult] = field(default_factory=list)

    @property
    def is_reached(self) -> bool:
        return self.outcome is Outcome.REACHED

    @property
    def edge(self) -> Position:
        """Where the arm reached, preferring the readback over the command."""
        return self.last_measured if self.last_measured is not None else self.last_good


def distance(a: Position, b: Position) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def unit_vector(origin: Position, target: Position) -> Position:
    span = distance(origin, target)
    if span == 0.0:
        return (0.0, 0.0, 0.0)
    return tuple((target[i] - origin[i]) / span for i in range(3))


def is_in_no_fly(x: float, y: float) -> bool:
    """Inside the square where the cup or hose can strike the robot's base."""
    return abs(x) <= NO_FLY_HALF and abs(y) <= NO_FLY_HALF


def is_segment_clear(start: Position, end: Position) -> bool:
    """True if the straight XY path between two poses misses the no-fly square.

    Endpoints outside the square are not enough: a move from one side to the
    other passes straight over the base. Slab method against the axis-aligned
    box, in XY only, since the zone applies at every Z.
    """
    if is_in_no_fly(start[0], start[1]) or is_in_no_fly(end[0], end[1]):
        return False

    low, high = 0.0, 1.0
    for axis in (0, 1):
        origin, delta = start[axis], end[axis] - start[axis]
        if abs(delta) < 1e-9:
            if abs(origin) <= NO_FLY_HALF:
                continue           # parallel and within the slab: cannot exclude
            return True            # parallel and outside it: never enters
        near = (-NO_FLY_HALF - origin) / delta
        far = (NO_FLY_HALF - origin) / delta
        if near > far:
            near, far = far, near
        low, high = max(low, near), min(high, far)
        if low > high:
            return True            # slabs do not overlap: no crossing
    return False


def clamp_to_bounds(position: Position) -> Position:
    """Pull a pose into the enforced envelope, leaving X and Y otherwise alone.

    The arm homes to Z=212, just above the enforced ceiling, so the first
    commanded move of a session has to be clamped or it is refused outright.
    """
    x, y, z = position
    return (min(max(x, X_MIN), X_MAX), min(max(y, Y_MIN), Y_MAX), min(max(z, Z_MIN), Z_MAX))


def is_within_bounds(position: Position) -> bool:
    x, y, z = position
    return (X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX and Z_MIN <= z <= Z_MAX
            and math.hypot(x, y) >= MIN_RADIUS and not is_in_no_fly(x, y))


class ReachProbe:
    """Guarded walker. One instance per session; it tracks the live pose."""

    def __init__(
        self,
        link: MaxArmLink,
        step_mm: float = 2.0,
        tolerance_mm: float = 12.0,
        z_drift_mm: float = 10.0,
        desk_z_drift_mm: float = 3.0,
        step_ms: int = 80,
        settle_ms: int = 40,
        is_bounds_enforced: bool = True,
        stall_fraction: float = 0.25,
        stall_steps: int = 3,
        stall_reference_steps: int = 3,
        retreat_lift_mm: float = 5.0,
        desk_zone_mm: float = 15.0,
    ) -> None:
        self.link = link
        self.step_mm = step_mm
        self.tolerance_mm = tolerance_mm
        self.z_drift_mm = z_drift_mm
        self.desk_z_drift_mm = desk_z_drift_mm
        self.stall_fraction = stall_fraction
        self.stall_steps = stall_steps
        self.stall_reference_steps = stall_reference_steps
        self.retreat_lift_mm = retreat_lift_mm
        self.desk_zone_mm = desk_zone_mm
        # The arm's ordinary Z error, measured locally rather than assumed. Sag
        # is judged against this, so a constant offset raises no alarm.
        self.z_baseline = 0.0
        self.last_commanded: Optional[Position] = None
        self.step_ms = step_ms
        self.settle_ms = settle_ms
        self.is_bounds_enforced = is_bounds_enforced
        self.position: Position = link.get_position() or link.get_cached_position()

    def _classify(self, commanded: Position, verdict, measured) -> Outcome:
        if verdict is None:
            return Outcome.BLIND_ZONE
        if verdict is False:
            return Outcome.REFUSED
        if measured is None:
            return Outcome.READ_FAILED
        # One-sided on purpose: only a readback BELOW the command is physical.
        # Reading high is lag or feedback offset. The baseline is subtracted
        # first, so the arm's ordinary Z error does not register as a fault.
        # Two thresholds, because the two situations are not alike. Near the
        # desk a few mm of droop means contact, so stay sensitive. In free space
        # droop is just the arm bending under its own weight -- real, smooth and
        # predictable -- so let the walk continue and record how much there was.
        sag = commanded[2] - measured[2] - self.z_baseline
        is_near_desk = commanded[2] <= Z_MIN + self.desk_zone_mm
        if sag > (self.desk_z_drift_mm if is_near_desk else self.z_drift_mm):
            return Outcome.Z_DESK if is_near_desk else Outcome.Z_SAG
        # Deliberately loose: X/Y/Z all carry a standing offset between command
        # and readback, and that is normal. This only catches tracking being
        # lost outright. A real limit is found by stall detection in walk_to(),
        # which needs no calibration because it watches progress, not error.
        if any(abs(measured[i] - commanded[i]) > self.tolerance_mm for i in range(3)):
            return Outcome.CLAMPED
        return Outcome.REACHED

    def sag_at(self, step: StepResult) -> Optional[float]:
        """Droop below the commanded Z for one step, net of the local baseline."""
        if step.measured is None:
            return None
        return step.commanded[2] - step.measured[2] - self.z_baseline

    def _confirm_sag(self, commanded: Position) -> Optional[Outcome]:
        """Re-read before believing a sag. Returns the outcome if it persists.

        Readback is quantised to 1 mm and carries ~2.4 mm of noise, while a real
        desk contact shows up as about 4 mm. Those are close enough that a
        single sample cannot separate them, so a sag has to survive a second
        look before the walk is abandoned.
        """
        verdict, measured = self.link.probe(commanded, self.step_ms, self.settle_ms * 2)
        if measured is not None:
            self.position = measured
        outcome = self._classify(commanded, verdict, measured)
        return outcome if outcome in (Outcome.Z_SAG, Outcome.Z_DESK) else None

    def step_to(self, commanded: Position, read_retries: int = 2) -> StepResult:
        """One increment. Updates the tracked pose from the measurement.

        A failed bus-servo read is transient and says nothing about the move,
        so it is retried rather than allowed to end the walk. Letting one
        dropped read abort a scan line loses that line's edge entirely.
        """
        verdict, measured = self.link.probe(commanded, self.step_ms, self.settle_ms)
        for _ in range(read_retries):
            if measured is not None or not verdict:
                break
            measured = self.link.get_position()
        self.last_commanded = commanded
        outcome = self._classify(commanded, verdict, measured)
        if measured is not None:
            self.position = measured
        return StepResult(commanded, measured, verdict, outcome)

    def _waypoints(self, target: Position, step_mm: float) -> List[Position]:
        start = self.position
        span = distance(start, target)
        if span <= step_mm:
            return [target]
        count = int(math.ceil(span / step_mm))
        points = []
        for n in range(1, count + 1):
            fraction = n / count
            points.append(tuple(
                round(start[i] + (target[i] - start[i]) * fraction, 2) for i in range(3)
            ))
        points[-1] = target
        return points

    def walk_to(self, target: Position, on_step=None,
                step_mm: Optional[float] = None) -> WalkResult:
        """Step toward target until it is reached or the arm refuses/sags.

        `step_mm` overrides the default increment -- pass a coarse value to
        cross interior the model already says is safe, and leave it at the
        default fine step when hunting an actual edge.

        On a Z sag the arm is backed out to the last good pose before returning,
        so the caller is always left somewhere safe.
        """
        if self.is_bounds_enforced:
            if not is_within_bounds(target):
                outcome = (Outcome.NO_FLY if is_in_no_fly(target[0], target[1])
                           else Outcome.OUT_OF_BOUNDS)
                return WalkResult(target, outcome, self.position, self.position)
            if not is_segment_clear(self.position, target):
                # Refuse rather than plough through; route_around() gives a path.
                return WalkResult(target, Outcome.NO_FLY, self.position, self.position)

        step_mm = step_mm or self.step_mm
        heading = unit_vector(self.position, target)
        last_good = self.position
        last_measured = self.position
        stall_anchor = self.position
        stalled_run = 0
        advances: List[float] = []
        steps: List[StepResult] = []

        for waypoint in self._waypoints(target, step_mm):
            previous = self.position
            step = self.step_to(waypoint)
            steps.append(step)
            if on_step is not None:
                on_step(step)

            if self.is_bounds_enforced and is_in_no_fly(waypoint[0], waypoint[1]):
                return WalkResult(target, Outcome.NO_FLY, last_good, last_measured, steps)

            if not step.is_good:
                outcome = step.outcome
                if outcome in (Outcome.Z_SAG, Outcome.Z_DESK):
                    persisted = self._confirm_sag(waypoint)
                    if persisted is None:
                        last_good = waypoint          # one noisy sample, not a sag
                        last_measured = self.position
                        continue
                    outcome = persisted
                if outcome in (Outcome.Z_SAG, Outcome.Z_DESK, Outcome.CLAMPED):
                    if not self.retreat_to(last_good):
                        outcome = Outcome.STUCK
                return WalkResult(target, outcome, last_good, last_measured, steps)

            # Progress along the heading, from readback to readback. A standing
            # offset cancels out here, so drift on any axis is harmless; only
            # the arm actually ceasing to advance counts as an edge.
            if step.measured is not None:
                advance = sum((step.measured[i] - previous[i]) * heading[i] for i in range(3))
                advances.append(advance)
                # What counts as "still moving" is measured, not assumed. The
                # arm has a deadband: a 2 mm command may only move it 0.3 mm,
                # and comparing against the commanded step would then read as a
                # stall on every step. The first few steps happen in known-good
                # space, so their advance is this walk's normal.
                if len(advances) > self.stall_reference_steps:
                    reference = sorted(advances[:self.stall_reference_steps])[
                        self.stall_reference_steps // 2]
                else:
                    reference = step_mm
                if advance < self.stall_fraction * max(reference, 0.2):
                    if stalled_run == 0:
                        stall_anchor = previous
                    stalled_run += 1
                    if stalled_run >= self.stall_steps:
                        return WalkResult(target, Outcome.STALLED, last_good,
                                          stall_anchor, steps)
                else:
                    stalled_run = 0
                last_measured = step.measured
            last_good = waypoint

        return WalkResult(target, Outcome.REACHED, target, last_measured, steps)

    def back_off_from_limit(self, distance_mm: float = 20.0, on_step=None) -> bool:
        """Unload the arm after a stall by retreating toward the shoulder joint.

        A stall means the arm is pressed against something while the firmware
        reports success, so leaving it there strains the servos and starts the
        next move from a loaded pose. Heading for the shoulder backs off radially
        and drops height together, which is the direction that relieves it.
        """
        span = distance(self.position, SHOULDER)
        if span < 1.0:
            return True
        fraction = min(distance_mm / span, 1.0)
        target = tuple(round(self.position[i]
                             + (SHOULDER[i] - self.position[i]) * fraction, 2)
                       for i in range(3))
        target = clamp_to_bounds(target)
        if is_in_no_fly(target[0], target[1]):
            return False        # backing off would put us over the base
        return self.walk_to(target, on_step=on_step, step_mm=max(distance_mm / 3, 5.0)
                            ).is_reached

    def escape_no_fly(self, on_step=None) -> bool:
        """Get out of the exclusion square if the arm is somehow inside it.

        Deliberately bypasses the usual guard: that guard refuses any move
        starting inside the zone, which would otherwise leave the arm trapped
        there. The escape is radially outward along the current bearing, which
        is the shortest way out and cannot drive deeper in.
        """
        x, y, z = self.position
        if not is_in_no_fly(x, y):
            return True

        reach = max(abs(x), abs(y))
        if reach < 1.0:
            return False        # sitting on the base axis; no bearing to follow
        scale = (NO_FLY_PLANNING_HALF + 4.0) / reach
        target = (round(x * scale, 2), round(y * scale, 2), z)
        print(f"    escaping no-fly zone: {(x, y, z)} -> {target}")

        start = self.position
        count = max(int(math.ceil(distance(start, target) / 5.0)), 1)
        for n in range(1, count + 1):
            waypoint = tuple(round(start[i] + (target[i] - start[i]) * n / count, 2)
                             for i in range(3))
            step = self.step_to(waypoint)
            if on_step is not None:
                on_step(step)
        return not is_in_no_fly(self.position[0], self.position[1])

    def route_around(self, target: Position) -> List[Position]:
        """Waypoints to `target` that keep clear of the no-fly square.

        Radially out to SAFE_ORBIT, an arc in small steps to the target's
        bearing, then radially in. Both radial legs run along a ray, and the
        target is outside the square, so neither can enter it; the arc stays at
        a radius beyond the square's corners.
        """
        start = self.position
        if is_segment_clear(start, target):
            return [target]

        start_angle = math.atan2(start[1], start[0])
        end_angle = math.atan2(target[1], target[0])
        waypoints: List[Position] = []
        if math.hypot(start[0], start[1]) < SAFE_ORBIT:
            waypoints.append((SAFE_ORBIT * math.cos(start_angle),
                              SAFE_ORBIT * math.sin(start_angle), start[2]))

        sweep = (end_angle - start_angle + math.pi) % (2 * math.pi) - math.pi
        arc_steps = max(1, int(math.ceil(abs(sweep) / math.radians(15))))
        for n in range(1, arc_steps + 1):
            angle = start_angle + sweep * n / arc_steps
            waypoints.append((SAFE_ORBIT * math.cos(angle),
                              SAFE_ORBIT * math.sin(angle), start[2]))
        waypoints.append(target)
        return [tuple(round(v, 2) for v in point) for point in waypoints]

    def walk_around_to(self, target: Position, on_step=None,
                       step_mm: Optional[float] = None) -> WalkResult:
        """walk_to(), detouring around the no-fly square when the direct path crosses it."""
        if is_in_no_fly(self.position[0], self.position[1]):
            if not self.escape_no_fly(on_step):
                return WalkResult(target, Outcome.NO_FLY, self.position, self.position)
        result = self.walk_to(target, on_step=on_step, step_mm=step_mm)
        if result.outcome is not Outcome.NO_FLY or is_in_no_fly(target[0], target[1]):
            return result
        for waypoint in self.route_around(target):
            result = self.walk_to(waypoint, on_step=on_step, step_mm=step_mm)
            if not result.is_reached:
                return result
        return result

    def retreat_to(self, position: Position, lift_attempts: int = 3) -> bool:
        """Back out to a pose known good a moment ago; confirm Z recovers.

        Retreating to the last waypoint is often not enough: if the nozzle is
        loaded against the desk, the pose one step back is still pressing. So
        each failed attempt lifts further before retrying.
        """
        target = position
        for _ in range(lift_attempts + 1):
            verdict, measured = self.link.probe(target, self.step_ms * 3, self.settle_ms * 2)
            if measured is not None:
                self.position = measured
            if verdict and measured is not None:
                if target[2] - measured[2] - self.z_baseline <= self.z_drift_mm:
                    return True
            target = (target[0], target[1], min(target[2] + self.retreat_lift_mm, Z_MAX))
        return False

    def calibrate_z_baseline(self, samples: int = 2) -> float:
        """Learn the arm's ordinary Z error at the pose it is standing in.

        Called at a known-safe interior point, so whatever error shows up here
        is the arm's normal behaviour, not a limit. Sag is judged relative to
        it, which is why no global drift constant has to be guessed.
        """
        if self.last_commanded is None:
            return self.z_baseline
        commanded_z = self.last_commanded[2]
        readings = []
        for _ in range(max(samples, 1)):
            _, measured = self.link.probe(self.last_commanded, self.step_ms, self.settle_ms)
            if measured is not None:
                self.position = measured
                readings.append(commanded_z - measured[2])
        if readings:
            self.z_baseline = sum(readings) / len(readings)
        return self.z_baseline

    def settle(self, wait_s: float = 0.4, reads: int = 5) -> Optional[Position]:
        """Wait for the arm to finish creeping, then read it properly.

        Measured on the arm over 24 paired trials: after a command lands, the
        error keeps shrinking for a few hundred ms, worth about 0.8 mm. Whether
        those milliseconds are spent re-commanding the same coordinate or simply
        waiting makes no difference -- final error 2.57 vs 2.58 mm, a paired
        difference of 0.01 mm against a 0.08 mm standard error. So this waits,
        and costs no servo traffic at all.

        The reads are then reduced by median rather than taken singly, because
        the readback is quantised to 1 mm and jitters by about that much.
        """
        time.sleep(wait_s)
        samples = []
        for _ in range(max(reads, 1)):
            reading = self.link.get_position()
            if reading is not None:
                samples.append(reading)
        if not samples:
            return self.position
        self.position = tuple(
            sorted(s[axis] for s in samples)[len(samples) // 2] for axis in range(3))
        return self.position

    def walk_to(self, target: Position, on_step=None,
                step_mm: Optional[float] = None) -> WalkResult:
        """Step toward target until it is reached or the arm refuses/sags.

        `step_mm` overrides the default increment -- pass a coarse value to
        cross interior the model already says is safe, and leave it at the
        default fine step when hunting an actual edge.

        On a Z sag the arm is backed out to the last good pose before returning,
        so the caller is always left somewhere safe.
        """
        if self.is_bounds_enforced:
            if not is_within_bounds(target):
                outcome = (Outcome.NO_FLY if is_in_no_fly(target[0], target[1])
                           else Outcome.OUT_OF_BOUNDS)
                return WalkResult(target, outcome, self.position, self.position)
            if not is_segment_clear(self.position, target):
                # Refuse rather than plough through; route_around() gives a path.
                return WalkResult(target, Outcome.NO_FLY, self.position, self.position)

        step_mm = step_mm or self.step_mm
        heading = unit_vector(self.position, target)
        last_good = self.position
        last_measured = self.position
        stall_anchor = self.position
        stalled_run = 0
        advances: List[float] = []
        steps: List[StepResult] = []

        for waypoint in self._waypoints(target, step_mm):
            previous = self.position
            step = self.step_to(waypoint)
            steps.append(step)
            if on_step is not None:
                on_step(step)

            if self.is_bounds_enforced and is_in_no_fly(waypoint[0], waypoint[1]):
                return WalkResult(target, Outcome.NO_FLY, last_good, last_measured, steps)

            if not step.is_good:
                outcome = step.outcome
                if outcome in (Outcome.Z_SAG, Outcome.Z_DESK):
                    persisted = self._confirm_sag(waypoint)
                    if persisted is None:
                        last_good = waypoint          # one noisy sample, not a sag
                        last_measured = self.position
                        continue
                    outcome = persisted
                if outcome in (Outcome.Z_SAG, Outcome.Z_DESK, Outcome.CLAMPED):
                    if not self.retreat_to(last_good):
                        outcome = Outcome.STUCK
                return WalkResult(target, outcome, last_good, last_measured, steps)

            # Progress along the heading, from readback to readback. A standing
            # offset cancels out here, so drift on any axis is harmless; only
            # the arm actually ceasing to advance counts as an edge.
            if step.measured is not None:
                advance = sum((step.measured[i] - previous[i]) * heading[i] for i in range(3))
                advances.append(advance)
                # What counts as "still moving" is measured, not assumed. The
                # arm has a deadband: a 2 mm command may only move it 0.3 mm,
                # and comparing against the commanded step would then read as a
                # stall on every step. The first few steps happen in known-good
                # space, so their advance is this walk's normal.
                if len(advances) > self.stall_reference_steps:
                    reference = sorted(advances[:self.stall_reference_steps])[
                        self.stall_reference_steps // 2]
                else:
                    reference = step_mm
                if advance < self.stall_fraction * max(reference, 0.2):
                    if stalled_run == 0:
                        stall_anchor = previous
                    stalled_run += 1
                    if stalled_run >= self.stall_steps:
                        return WalkResult(target, Outcome.STALLED, last_good,
                                          stall_anchor, steps)
                else:
                    stalled_run = 0
                last_measured = step.measured
            last_good = waypoint

        return WalkResult(target, Outcome.REACHED, target, last_measured, steps)

    def back_off_from_limit(self, distance_mm: float = 20.0, on_step=None) -> bool:
        """Unload the arm after a stall by retreating toward the shoulder joint.

        A stall means the arm is pressed against something while the firmware
        reports success, so leaving it there strains the servos and starts the
        next move from a loaded pose. Heading for the shoulder backs off radially
        and drops height together, which is the direction that relieves it.
        """
        span = distance(self.position, SHOULDER)
        if span < 1.0:
            return True
        fraction = min(distance_mm / span, 1.0)
        target = tuple(round(self.position[i]
                             + (SHOULDER[i] - self.position[i]) * fraction, 2)
                       for i in range(3))
        target = clamp_to_bounds(target)
        if is_in_no_fly(target[0], target[1]):
            return False        # backing off would put us over the base
        return self.walk_to(target, on_step=on_step, step_mm=max(distance_mm / 3, 5.0)
                            ).is_reached

    def escape_no_fly(self, on_step=None) -> bool:
        """Get out of the exclusion square if the arm is somehow inside it.

        Deliberately bypasses the usual guard: that guard refuses any move
        starting inside the zone, which would otherwise leave the arm trapped
        there. The escape is radially outward along the current bearing, which
        is the shortest way out and cannot drive deeper in.
        """
        x, y, z = self.position
        if not is_in_no_fly(x, y):
            return True

        reach = max(abs(x), abs(y))
        if reach < 1.0:
            return False        # sitting on the base axis; no bearing to follow
        scale = (NO_FLY_PLANNING_HALF + 4.0) / reach
        target = (round(x * scale, 2), round(y * scale, 2), z)
        print(f"    escaping no-fly zone: {(x, y, z)} -> {target}")

        start = self.position
        count = max(int(math.ceil(distance(start, target) / 5.0)), 1)
        for n in range(1, count + 1):
            waypoint = tuple(round(start[i] + (target[i] - start[i]) * n / count, 2)
                             for i in range(3))
            step = self.step_to(waypoint)
            if on_step is not None:
                on_step(step)
        return not is_in_no_fly(self.position[0], self.position[1])

    def route_around(self, target: Position) -> List[Position]:
        """Waypoints to `target` that keep clear of the no-fly square.

        Radially out to SAFE_ORBIT, an arc in small steps to the target's
        bearing, then radially in. Both radial legs run along a ray, and the
        target is outside the square, so neither can enter it; the arc stays at
        a radius beyond the square's corners.
        """
        start = self.position
        if is_segment_clear(start, target):
            return [target]

        start_angle = math.atan2(start[1], start[0])
        end_angle = math.atan2(target[1], target[0])
        waypoints: List[Position] = []
        if math.hypot(start[0], start[1]) < SAFE_ORBIT:
            waypoints.append((SAFE_ORBIT * math.cos(start_angle),
                              SAFE_ORBIT * math.sin(start_angle), start[2]))

        sweep = (end_angle - start_angle + math.pi) % (2 * math.pi) - math.pi
        arc_steps = max(1, int(math.ceil(abs(sweep) / math.radians(15))))
        for n in range(1, arc_steps + 1):
            angle = start_angle + sweep * n / arc_steps
            waypoints.append((SAFE_ORBIT * math.cos(angle),
                              SAFE_ORBIT * math.sin(angle), start[2]))
        waypoints.append(target)
        return [tuple(round(v, 2) for v in point) for point in waypoints]

    def walk_around_to(self, target: Position, on_step=None,
                       step_mm: Optional[float] = None) -> WalkResult:
        """walk_to(), detouring around the no-fly square when the direct path crosses it."""
        if is_in_no_fly(self.position[0], self.position[1]):
            if not self.escape_no_fly(on_step):
                return WalkResult(target, Outcome.NO_FLY, self.position, self.position)
        result = self.walk_to(target, on_step=on_step, step_mm=step_mm)
        if result.outcome is not Outcome.NO_FLY or is_in_no_fly(target[0], target[1]):
            return result
        for waypoint in self.route_around(target):
            result = self.walk_to(waypoint, on_step=on_step, step_mm=step_mm)
            if not result.is_reached:
                return result
        return result

    def retreat_to(self, position: Position, lift_attempts: int = 3) -> bool:
        """Back out to a pose known good a moment ago; confirm Z recovers.

        Retreating to the last waypoint is often not enough: if the nozzle is
        loaded against the desk, the pose one step back is still pressing. So
        each failed attempt lifts further before retrying.
        """
        target = position
        for _ in range(lift_attempts + 1):
            verdict, measured = self.link.probe(target, self.step_ms * 3, self.settle_ms * 2)
            if measured is not None:
                self.position = measured
            if verdict and measured is not None:
                if target[2] - measured[2] - self.z_baseline <= self.z_drift_mm:
                    return True
            target = (target[0], target[1], min(target[2] + self.retreat_lift_mm, Z_MAX))
        return False

    def calibrate_z_baseline(self, samples: int = 2) -> float:
        """Learn the arm's ordinary Z error at the pose it is standing in.

        Called at a known-safe interior point, so whatever error shows up here
        is the arm's normal behaviour, not a limit. Sag is judged relative to
        it, which is why no global drift constant has to be guessed.
        """
        if self.last_commanded is None:
            return self.z_baseline
        commanded_z = self.last_commanded[2]
        readings = []
        for _ in range(max(samples, 1)):
            _, measured = self.link.probe(self.last_commanded, self.step_ms, self.settle_ms)
            if measured is not None:
                self.position = measured
                readings.append(commanded_z - measured[2])
        if readings:
            self.z_baseline = sum(readings) / len(readings)
        return self.z_baseline

    def settle(self, attempts: int = 6, quiet_mm: float = 0.5,
               quiet_run: int = 2) -> Optional[Position]:
        """Re-issue the last command until the readback really stops changing.

        Measured on the arm: when extended, a single command leaves ~3 mm on the
        table and re-commanding walks it in to ~1.4 mm over about four tries.
        That is stiction -- each command breaks it loose again -- not sag.

        Two consecutive identical readings do NOT mean settled: with 1 mm
        readback quantisation the arm often reads the same twice and then moves
        again on the third command. So a run of quiet readings is required.
        """
        if self.last_commanded is None:
            return self.position
        previous = self.position
        quiet = 0
        for _ in range(attempts):
            verdict, measured = self.link.probe(self.last_commanded, self.step_ms,
                                                self.settle_ms)
            if measured is None:
                break
            self.position = measured
            quiet = quiet + 1 if distance(measured, previous) <= quiet_mm else 0
            previous = measured
            if quiet >= quiet_run:
                break
        return self.position

    def walk_along_axis(
        self,
        axis: int,
        limit: float,
        on_step=None,
        step_mm: Optional[float] = None,
    ) -> WalkResult:
        """Walk one axis to `limit`, holding the others. Returns where it stopped.

        Used by the sweep to find the X reach-out and reach-in edges on a line.
        """
        target = list(self.position)
        target[axis] = limit
        return self.walk_to(tuple(round(v, 2) for v in target), on_step=on_step,
                            step_mm=step_mm)


def describe(step: StepResult) -> str:
    commanded = "({:7.1f},{:7.1f},{:7.1f})".format(*step.commanded)
    measured = ("({:7.1f},{:7.1f},{:7.1f})".format(*step.measured)
                if step.measured else "        <no read>       ")
    return f"cmd {commanded} -> got {measured}  {step.outcome.value}"
