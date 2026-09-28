#!/usr/bin/env python3
"""Predict where the arm can go, without moving it.

Every way `ESPMax.set_position()` can fail is knowable up front:

  1. blind cylinder   -- it returns None when sqrt(x^2+y^2) < 50
  2. no IK solution   -- it returns False when __espmax.inverse() throws
  3. silent joint clamp -- set_servo_in_range() pins servo 2 above pulse 700 and
     servo 3 below 470, then reports success anyway

The board answers all three for free: (1) is arithmetic, (2) is what
`verify_position()` already tests, and (3) falls out of `position_to_pulses()`.
So a whole scan line can be predicted in one round trip with the arm standing
still, and the physical probe only has to visit the predicted boundary.

The prediction is a *model*, not ground truth. It knows nothing about gravity
sag, the desk, or mechanical interference, so the sweep still walks the last
few millimetres for real. Treat it as a way to skip the boring interior.
"""

import math
from typing import List, Optional, Sequence, Tuple

from maxarm_link import MaxArmLink, Position
from reach_probe import is_in_no_fly

MIN_RADIUS_SQUARED = 2500  # ESPMax refuses sqrt(x^2+y^2) < 50
SERVO_2_MAX_PULSE = 700
SERVO_3_MIN_PULSE = 470

# One point's verdict. `and` short-circuits, so position_to_pulses() is only
# reached when the IK actually solves and therefore cannot throw. espmax negates
# x before solving, so every call here pre-negates to match a real move.
# A whole line in one comprehension: the body is written once and the X values
# ride along as bare numbers. Batching the predicate per point instead cost
# ~180 bytes of source each, which overflowed the board's heap on a full line.
# The reply is 0/1 per point, so nothing large is built on the board either.
LINE_EXPRESSION = (
    "[1 if (x * x + {yy} >= {r2}"
    " and arm.verify_position(-x, {y}, {z})"
    " and (lambda p: p[1] <= {s2} and p[2] >= {s3})"
    "(arm.position_to_pulses((-x, {y}, {z})))) else 0"
    " for x in ({xs})]"
)

# The same idea with X fixed and Y varying, for finding how far forward or back
# the arm reaches at a given X. An X-scan cannot answer that: where the boundary
# is Y-limited it runs between scan lines and is never probed.
COLUMN_EXPRESSION = (
    "[1 if ({xx} + y * y >= {r2}"
    " and arm.verify_position({nx}, y, {z})"
    " and (lambda p: p[1] <= {s2} and p[2] >= {s3})"
    "(arm.position_to_pulses(({nx}, y, {z})))) else 0"
    " for y in ({ys})]"
)

# Fully general: an explicit list of (x, y) pairs at one Z. Needed for probing
# along a bearing, where both coordinates vary, which neither the row nor the
# column form can express.
POINTS_EXPRESSION = (
    "[1 if (p[0] * p[0] + p[1] * p[1] >= {r2}"
    " and arm.verify_position(-p[0], p[1], {z})"
    " and (lambda q: q[1] <= {s2} and q[2] >= {s3})"
    "(arm.position_to_pulses((-p[0], p[1], {z})))) else 0"
    " for p in ({pairs})]"
)

# Points per round trip. Small enough that neither the source nor the reply is
# a large allocation on a board with a few tens of KB of heap.
LINE_CHUNK = 24

POINT_PREDICATE = (
    "({x} * {x} + {y} * {y} >= {r2}"
    " and arm.verify_position({nx}, {y}, {z})"
    " and (lambda p: p[1] <= {s2} and p[2] >= {s3})"
    "(arm.position_to_pulses(({nx}, {y}, {z}))))"
)


def _predicate(x: float, y: float, z: float) -> str:
    return POINT_PREDICATE.format(x=x, y=y, z=z, nx=-x, r2=MIN_RADIUS_SQUARED,
                                  s2=SERVO_2_MAX_PULSE, s3=SERVO_3_MIN_PULSE)


def _longest_run(values: Sequence[float], verdicts: Sequence[bool]):
    """Longest contiguous reachable run, as (first, last).

    A run rather than plain min/max: the reachable set along a line is normally
    contiguous, and taking the longest run keeps one stray outlier from
    stretching the span across a genuine gap.
    """
    best = None
    start = None
    for index, is_reachable in enumerate(list(verdicts) + [False]):
        if is_reachable and start is None:
            start = index
        elif not is_reachable and start is not None:
            if best is None or index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    if best is None:
        return None
    return values[best[0]], values[best[1] - 1]


class EnvelopeModel:
    """Motion-free reachability queries against the board's own kinematics."""

    def __init__(self, link: MaxArmLink) -> None:
        self.link = link
        self.query_count = 0

    def is_predicted_reachable(self, position: Position) -> bool:
        if is_in_no_fly(position[0], position[1]):
            return False        # base exclusion square; no need to ask the board
        self.query_count += 1
        return bool(self.link.evaluate(_predicate(*position)))

    def get_pulses(self, position: Position) -> Optional[Tuple[float, float, float]]:
        """Servo pulses the board would command, or None if the IK fails.

        No motion. `and` short-circuits so position_to_pulses is only reached
        when the IK solves and therefore cannot throw.
        """
        self.query_count += 1
        x, y, z = position
        reply = self.link.evaluate(
            f"arm.verify_position({-x}, {y}, {z})"
            f" and arm.position_to_pulses(({-x}, {y}, {z}))")
        return tuple(float(v) for v in reply) if reply else None

    def joint_margin(self, position: Position) -> Optional[Tuple[float, float]]:
        """How much room is left before servo 2 and servo 3 get clamped.

        set_servo_in_range() pins servo 2 above pulse 700 and servo 3 below 470
        and still reports success, so a pose whose margin is near zero will
        accept commands and not move. Negative means already clamped.
        """
        pulses = self.get_pulses(position)
        if pulses is None:
            return None
        return (SERVO_2_MAX_PULSE - pulses[1], pulses[2] - SERVO_3_MIN_PULSE)

    def predict_line(self, y: float, z: float, x_values: Sequence[float]) -> List[bool]:
        """Feasibility for every X on one scan line, in a few round trips.

        Chunked deliberately: MicroPython v1.12 on this board raises MemoryError
        compiling a long enough expression, and a full line is long enough.
        """
        if not x_values:
            return []
        self.query_count += len(x_values)

        verdicts: List[bool] = []
        for start in range(0, len(x_values), LINE_CHUNK):
            chunk = x_values[start:start + LINE_CHUNK]
            xs = ", ".join(f"{x:g}" for x in chunk) + ","
            reply = self.link.evaluate(LINE_EXPRESSION.format(
                yy=y * y, y=y, z=z, r2=MIN_RADIUS_SQUARED,
                s2=SERVO_2_MAX_PULSE, s3=SERVO_3_MIN_PULSE, xs=xs))
            verdicts.extend(bool(v) for v in reply)
        return verdicts

    def predict_column(self, x: float, z: float, y_values: Sequence[float]) -> List[bool]:
        """Feasibility for every Y at one X, chunked like predict_line()."""
        if not y_values:
            return []
        self.query_count += len(y_values)
        verdicts: List[bool] = []
        for start in range(0, len(y_values), LINE_CHUNK):
            chunk = y_values[start:start + LINE_CHUNK]
            ys = ", ".join(f"{v:g}" for v in chunk) + ","
            reply = self.link.evaluate(COLUMN_EXPRESSION.format(
                xx=x * x, nx=-x, z=z, r2=MIN_RADIUS_SQUARED,
                s2=SERVO_2_MAX_PULSE, s3=SERVO_3_MIN_PULSE, ys=ys))
            verdicts.extend(bool(v) for v in reply)
        return verdicts

    def predict_points(self, points, z: float) -> List[bool]:
        """Feasibility for an arbitrary list of (x, y) at one Z. No motion."""
        if not points:
            return []
        self.query_count += len(points)
        verdicts: List[bool] = []
        for start in range(0, len(points), LINE_CHUNK):
            chunk = points[start:start + LINE_CHUNK]
            pairs = ", ".join(f"({x:g},{y:g})" for x, y in chunk) + ","
            reply = self.link.evaluate(POINTS_EXPRESSION.format(
                z=z, r2=MIN_RADIUS_SQUARED, s2=SERVO_2_MAX_PULSE,
                s3=SERVO_3_MIN_PULSE, pairs=pairs))
            verdicts.extend(bool(v) for v in reply)
        return verdicts

    def find_ray_span(self, angle: float, z: float, radii):
        """Reachable radius range along one bearing, as (r_min, r_max).

        The natural description for this arm: both ends are usually real limits,
        because near the top of the envelope the reachable set is a narrow
        annulus rather than a disc.
        """
        points = [(r * math.cos(angle), r * math.sin(angle)) for r in radii]
        return _longest_run(radii, self.predict_points(points, z))

    def find_column_span(self, x: float, z: float,
                         y_values: Sequence[float]) -> Optional[Tuple[float, float]]:
        """Longest run of reachable Y at one X, as (y_min, y_max)."""
        return _longest_run(y_values, self.predict_column(x, z, y_values))

    def find_line_span(
        self,
        y: float,
        z: float,
        x_values: Sequence[float],
    ) -> Optional[Tuple[float, float]]:
        """Longest run of reachable X on a line, as (x_min, x_max).

        A run rather than plain min/max: the reachable set on a line is normally
        contiguous, and taking the longest run keeps one stray outlier from
        stretching the span across a genuine gap.
        """
        return _longest_run(x_values, self.predict_line(y, z, x_values))
