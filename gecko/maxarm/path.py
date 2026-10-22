#!/usr/bin/env python3
"""Where a move may go, and the way round what stands in its path. No board.

A move is planned and checked whole, as arithmetic, before the arm is sent
anywhere -- and the same checks answer `can_reach()` without a connection.

Points are in desk space (`spaces.py`): z is the height above the desk. What
belongs to the arm itself -- its reach and its base -- is checked in arm space.

- **Reach.** A move ends between `REACH_MIN_MM` and `REACH_MAX_MM` from the
  base axis, and nothing goes into the back 120 degrees at any radius. The two
  radii limit where a move goes, not the way there: unlike the arm-base limit,
  nothing goes round them, and a move that only climbs or stays put is let out
  of anywhere (an arm stopped half-way, say). The rest of the arm's envelope --
  the top, the inside -- is the firmware's, and `geometry.limit_reason()`
  predicts its refusals, servo 2's silent clamp included, from the raw z it
  will be sent, on the way as well as at the end.
- **The arm-base limit.** A box round the base that the cup and its hose stay
  out of. A move whose straight line crosses it goes by a front corner instead,
  at half-way height.
- **Clearance.** Given a crossing height, a move that travels sideways below it
  climbs first, crosses at that height, and comes down at the end.
"""

import math
from typing import Callable, List, Optional

from . import geometry, raw
from .geometry import Position
from .spaces import Affine
from .tunable import tunable

# How far from the base axis a move may end: the owner's limits.
REACH_MAX_MM = 280.0
REACH_MIN_MM = 95.0

# Inside this square of the arm's own frame, and below this height, the cup or
# its hose can strike the base.
BASE_LIMIT_HALF_MM = tunable("BASE_LIMIT_HALF_MM", 70.0)
BASE_LIMIT_HEIGHT_MM = tunable("BASE_LIMIT_HEIGHT_MM", 200.0)
# How far outside the corner, on both axes, a move round the base passes.
BASE_CORNER_MARGIN_MM = 10.0

# Added above the clearance wherever something travels sideways: the owner's spec.
CLEARANCE_MARGIN_MM = 5.0
# Corrections allowed to find the desk point over a base corner.
CORNER_SOLVE_STEPS = 20
# A sideways shift smaller than this is not travelling across the desk: it is
# below the arm's own accuracy.
CROSSING_MM = 2.0
# The lowest desk height a move may ask for. A little under the surface, as the
# cup is a bellows that seals by being pressed. Far under it is a point given in
# the wrong space -- desk heights passed as arm z, typically.
DESK_FLOOR_MM = -5.0


class Refused(Exception):
    """The move cannot be made, and nothing was sent. The message says why."""


class Planner:
    """The limits, applied to desk-space points. `arm` is the arm space."""

    def __init__(self, arm: Affine, is_desk_known: bool) -> None:
        self.arm = arm
        self.is_desk_known = is_desk_known

    def problem_with(self, point: Position) -> Optional[str]:
        """Why the cup cannot be sent to this desk point, or None if it can."""
        return self._problem(point, is_end=True)

    def problem_on_the_way(self, point: Position) -> Optional[str]:
        """Why the cup cannot pass through this desk point, or None if it can: all but the reach limits."""
        return self._problem(point, is_end=False)

    def _problem(self, point: Position, is_end: bool) -> Optional[str]:
        arm_point = self.arm.from_desk(point)
        x, y, _ = arm_point
        reach = math.hypot(x, y)
        if self.is_desk_known and point[2] < DESK_FLOOR_MM:
            return f"{-point[2]:.0f} mm below the desk"
        if is_end and reach > REACH_MAX_MM:
            return f"{reach:.0f} mm from the base axis, past the {REACH_MAX_MM:.0f} mm reach limit"
        if abs(geometry.bearing_degrees(x, y)) > geometry.BASE_FAN_DEG:
            return "behind the base, where the arm cannot turn"
        if point[2] < BASE_LIMIT_HEIGHT_MM and _is_in_base_square(arm_point):
            return "inside the arm-base limit"
        if is_end and reach < REACH_MIN_MM:
            return f"{reach:.0f} mm from the base axis, inside the {REACH_MIN_MM:.0f} mm reach limit"
        reason = geometry.limit_reason(raw.raw_position(arm_point))
        if reason:
            return f"a pose the arm cannot hold ({reason.replace('_', ' ')})"
        return None

    def plan(self, start: Position, target: Position,
             crossing_height: Optional[float]) -> List[Position]:
        """The waypoints from `start` to `target`, `target` last. Raises `Refused`.

        `crossing_height` is how high anything travelling sideways must be, or
        None when there is no clearance to keep.
        """
        problem = (self.problem_on_the_way(target) if _is_climbing_or_staying(start, target)
                   else self.problem_with(target))
        if problem:
            raise Refused(problem)
        path = _over_clearance(start, target, crossing_height)
        path = _each_leg(start, path, lambda a, b: self._round_base(a, b, depth=2))
        for point in path:
            problem = self.problem_on_the_way(point)
            if problem:
                raise Refused(f"on the way there: {problem}")
        return path

    def _round_base(self, start: Position, end: Position, depth: int) -> List[Position]:
        """`end`, or a front corner first if the straight line crosses the base limit.

        One corner is usually enough. A move from one side of the base to the
        other, behind its front edge, needs both, so the legs are checked again.
        """
        if not self._crosses_base(start, end):
            return [end]
        corners = self._front_corners(start, end)
        if depth == 0 or not corners:
            raise Refused("there is no way round the arm-base limit")
        corner = min(corners, key=lambda point: math.dist(start, point) + math.dist(point, end))
        return self._round_base(start, corner, depth - 1) + self._round_base(corner, end, depth - 1)

    def _crosses_base(self, start: Position, end: Position) -> bool:
        if min(start[2], end[2]) >= BASE_LIMIT_HEIGHT_MM:
            return False
        a, b = self.arm.from_desk(start), self.arm.from_desk(end)
        if _is_in_base_square(a):
            return False            # already inside: leaving is the one way out
        return _meets_base_square(a, b)

    def _front_corners(self, start: Position, end: Position) -> List[Position]:
        """Just outside the two front corners, at half-way height. The back two are out of reach."""
        height = (start[2] + end[2]) / 2.0
        out = BASE_LIMIT_HALF_MM + BASE_CORNER_MARGIN_MM
        corners = [self._at_desk_height(x, -out, height) for x in (-out, out)]
        return [corner for corner in corners
                if math.dist(corner, start) > 1.0 and math.dist(corner, end) > 1.0]

    def _at_desk_height(self, x: float, y: float, height: float) -> Position:
        """The desk point at a desk height whose arm-space x and y are (x, y).

        Desk x and y may differ a little from the arm's under a calibration, so the
        desk point is corrected until the arm's x and y agree.
        """
        point = (x, y, height)
        for _ in range(CORNER_SOLVE_STEPS):
            arm_x, arm_y, _ = self.arm.from_desk(point)
            if math.hypot(x - arm_x, y - arm_y) < 1e-9:
                break
            point = (point[0] + x - arm_x, point[1] + y - arm_y, height)
        return point


def _each_leg(start: Position, path: List[Position],
              expand: Callable[[Position, Position], List[Position]]) -> List[Position]:
    """Replace every leg of a path with whatever `expand` makes of it."""
    expanded: List[Position] = []
    cursor = start
    for point in path:
        expanded += expand(cursor, point)
        cursor = point
    return expanded


def _over_clearance(start: Position, target: Position,
                    height: Optional[float]) -> List[Position]:
    """Climb, cross and descend, if the move travels sideways below `height`."""
    if (height is None or min(start[2], target[2]) >= height
            or math.hypot(target[0] - start[0], target[1] - start[1]) < CROSSING_MM):
        return [target]
    path: List[Position] = []
    if start[2] < height:
        path.append((start[0], start[1], height))
    if target[2] < height:
        path.append((target[0], target[1], height))
    return path + [target]


def _is_climbing_or_staying(start: Position, target: Position) -> bool:
    """Whether a move goes nowhere new across the desk and not down: it leaves the arm no worse placed."""
    return (math.hypot(target[0] - start[0], target[1] - start[1]) < CROSSING_MM
            and target[2] >= start[2])


def _is_in_base_square(arm_point: Position) -> bool:
    return abs(arm_point[0]) <= BASE_LIMIT_HALF_MM and abs(arm_point[1]) <= BASE_LIMIT_HALF_MM


def _meets_base_square(a: Position, b: Position) -> bool:
    """Whether the segment a-b passes through the square, in x and y (Liang-Barsky)."""
    low, high = 0.0, 1.0
    for axis in (0, 1):
        delta = b[axis] - a[axis]
        if abs(delta) < 1e-9:
            if abs(a[axis]) > BASE_LIMIT_HALF_MM:
                return False
            continue
        near, far = sorted(((-BASE_LIMIT_HALF_MM - a[axis]) / delta,
                            (BASE_LIMIT_HALF_MM - a[axis]) / delta))
        low, high = max(low, near), min(high, far)
        if low > high:
            return False
    return True
