#!/usr/bin/env python3
"""Working out where to go and how to get there. Internal, and board-free.

Nothing here touches the wire, which is the point: a target can be accepted or
refused, and a whole route laid out, before a single byte goes to the board.
`Router` answers two questions for every `move_to()`:

  1. **Where are we actually going?** A target slightly outside the envelope or
     past an operator limit is pulled to the nearest legal point and reported
     as `APPROXIMATED`. One far outside, or inside an exclusion zone, is
     refused.
  2. **How do we get there in one piece?** Usually in one hop. A move that
     crosses an exclusion zone detours around it; one that ends near the desk
     travels high and comes down at the end; one whose own swing would leave
     safe space is split until each piece is safe.

That last one is the fact everything else here is built around: **the arm does
not move in straight lines.** `set_position()` solves the IK for the endpoint
and gives all three servos the same duration, so they interpolate in *pulse*
space and the tip swings -- 2 mm off the line on a short hop, 123 mm across the
front of the base. `geometry.pulse_path()` says exactly where it will go, so
checking a hop is a calculation rather than a guess.
"""

import math
from typing import List, NamedTuple, Optional, Tuple

from . import geometry, tuning
from .config import Limits
from .status import MoveStatus
from .zones import ZoneSet

Position = Tuple[float, float, float]


class Hop(NamedTuple):
    """One run to a waypoint.

    `is_stepped` is the only thing the driver needs to know: walk it 2 mm at a
    time with a readback after each, or fly it in one command. `purpose` is
    there so a failure can name the leg it stopped on.
    """

    position: Position
    purpose: str
    is_stepped: bool = False


class NoRoute(Exception):
    """No path to the target clears the exclusion zones."""


class Router:
    """Pure geometry: the operator's limits, the zones and the arm's envelope."""

    def __init__(self, limits: Limits, zones: ZoneSet) -> None:
        self.limits = limits
        self.zones = zones
        self._ceiling = geometry.max_reachable_z() - tuning.ENVELOPE_MARGIN_MM

    # --- where are we going -----------------------------------------------

    def resolve(self, request: Position) -> Tuple[Position, MoveStatus, str]:
        """Turn a request into the point that will actually be aimed at."""
        zone = self.zones.find_blocking(request)
        if zone is not None:
            # Never retargeted: sliding a target out of an exclusion zone puts
            # the nozzle somewhere the caller did not ask for, right next to
            # the one thing they were told to stay away from.
            return request, MoveStatus.NO_FLY, f"target is inside the {zone.name} zone"

        legal = self._nearest_legal(request)
        if legal is None:
            reason = geometry.limit_reason(request) or "no legal point nearby"
            return request, MoveStatus.UNREACHABLE, reason

        shift = _distance(request, legal)
        if shift < 0.05:
            return legal, MoveStatus.REACHED, ""
        if shift > tuning.RETARGET_LIMIT_MM:
            reason = geometry.limit_reason(request) or "outside the operator limits"
            return (request, MoveStatus.UNREACHABLE,
                    f"nearest legal point is {shift:.1f} mm away ({reason})")
        return legal, MoveStatus.APPROXIMATED, f"pulled {shift:.1f} mm to the nearest legal pose"

    def is_legal(self, position: Position) -> bool:
        return self.resolve(position)[1] is MoveStatus.REACHED

    def _nearest_legal(self, position: Position) -> Optional[Position]:
        """Closest pose satisfying every constraint, or None if there is none.

        Limits and envelope are applied in turn and repeated, because each can
        undo the other: clamping x to an operator bound changes the radius, and
        pulling the radius in can push x back past that bound. Two or three
        passes settle it; if they do not, there is no legal point near enough
        to be what the caller meant.
        """
        point = position
        for _ in range(4):
            previous = point
            point = self._into_envelope(self.limits.clamp(point))
            if _distance(point, previous) < 1e-6:
                break
        if not (self.limits.contains(point) and self.is_in_envelope(point)
                and self.zones.is_clear(point)):
            return None
        return tuple(round(value, 2) for value in point)

    # --- how do we get there ----------------------------------------------

    def route(self, start: Position, target: Position) -> List[Hop]:
        """The hops a move breaks into. Usually one; a handful at the most.

        A move in open space is a single hop, which is the whole point of
        checking the swing rather than chopping every move into waypoints. The
        extra hops appear only when the move needs them.
        """
        hops: List[Hop] = []
        cursor = start

        escape = self.zones.escape_waypoint(start)
        if escape is not None:
            hops.append(Hop(escape, "escape"))      # already in a zone: get out first
            cursor = escape

        if self._is_staging_needed(cursor, target):
            travel_z = self._travel_height(cursor, target)
            if travel_z > cursor[2] + 0.5:
                cursor = (cursor[0], cursor[1], round(travel_z, 2))
                hops.append(Hop(cursor, "lift"))
            cruise = (target[0], target[1], cursor[2])
            for waypoint in self._crossing(cursor, cruise):
                hops.append(Hop(waypoint, "travel"))
                cursor = waypoint

        if self.is_near_desk(target):
            # Fly down to the top of the guard band and walk only what is left.
            # Stepping the whole descent from travel height would be a crawl,
            # and there is nothing to catch until the cup is near the surface.
            guard_top = round(self.limits.z_floor + tuning.DESK_GUARD_MM, 2)
            if cursor[2] > guard_top + 0.5:
                cursor = (target[0], target[1], guard_top)
                hops.append(Hop(cursor, "approach"))

        if _distance(cursor, target) > 0.05 or not hops:
            is_stepped = self.is_near_desk(target)
            is_descent = is_stepped and target[2] < cursor[2] - 0.5
            hops.append(Hop(target, "descend" if is_descent else "move", is_stepped))

        hops = self._split_unsafe_swings(start, hops)
        if not all(geometry.is_firmware_reachable(hop.position) for hop in hops):
            # Staging put a waypoint somewhere the arm cannot hold. One direct
            # hop is the honest fallback: it may stop short, and the drive will
            # say so rather than the route pretending otherwise.
            return [Hop(target, "move", is_stepped=self.is_near_desk(target))]
        return hops

    def _is_staging_needed(self, start: Position, target: Position) -> bool:
        """Does this move need to climb, cross and come down separately?

        Only when the XY move is real *and* something about it is awkward: it
        starts or ends near the desk, or its swing would leave safe space. A
        move between two poses high over an empty desk is one hop.
        """
        if _xy_distance(start, target) < tuning.DESK_CLEARANCE_MM:
            return False
        if self.is_near_desk(start) or self.is_near_desk(target):
            return True
        return not self.is_swing_safe(start, target)

    def _crossing(self, start: Position, cruise: Position) -> List[Position]:
        """The XY crossing at travel height, detoured around zones if needed.

        Judged on the straight line between the endpoints, not on the swing,
        even though the swing is what the arm will fly. That is deliberate: a
        cross-front move bows about 120 mm *outward*, away from the base, so
        trusting it would mean almost never detouring -- and being wrong about
        the bow direction once costs a collision with the robot's own casting.
        The swing is still checked afterwards, by `_split_unsafe_swings`, so
        this is the conservative of two layers rather than the only one.
        """
        if _xy_distance(start, cruise) < 0.05:
            return []
        if self.zones.is_path_clear(start, cruise):
            return [cruise]

        orbit = self._orbit_radius(start[2])
        if orbit is None:
            raise NoRoute(f"no orbit radius clears the zones at z={start[2]:.0f}")
        waypoints = self.zones.route_around(start, cruise, orbit_radius=orbit)
        legs = [start] + waypoints
        for index in range(len(legs) - 1):
            if not self.zones.is_path_clear(legs[index], legs[index + 1]):
                raise NoRoute("no route found that clears every zone")
        return waypoints

    def _split_unsafe_swings(self, start: Position, hops: List[Hop]) -> List[Hop]:
        """Halve any hop whose swing would leave safe space, until it would not.

        Splitting is a routing decision, not a driving one: every hop that
        comes out of here is still flown with a single command. Halving a hop
        quarters its bow, so a 20 mm swing is under a millimetre after two
        rounds. Stepped hops are left alone -- they are already walked at 2 mm.
        """
        cursor = start
        safe: List[Hop] = []
        for hop in hops:
            safe.extend(self._pieces_of(cursor, hop))
            cursor = hop.position
        return safe

    def _pieces_of(self, start: Position, hop: Hop) -> List[Hop]:
        """`hop` whole if its swing is safe, else halved until it is.

        Giving up after `MAX_SPLITS` rather than refusing is deliberate: the
        zone crossings have already been detoured around by this point, so what
        is left is a swing grazing the envelope, and the drive watches for that
        anyway.
        """
        if hop.is_stepped or hop.purpose == "escape":
            # An escape starts *inside* a zone, so its swing can never pass the
            # check and halving it only strands the arm nearer the obstacle.
            # Straight out along the current bearing is the shortest way clear
            # and cannot drive deeper in, which is the whole argument for it.
            return [hop]
        points = [hop.position]
        for _ in range(tuning.MAX_SPLITS):
            if self._is_chain_safe(start, points):
                break
            points = _halve(start, points)
        return [Hop(point, hop.purpose) for point in points]

    def _is_chain_safe(self, start: Position, points: List[Position]) -> bool:
        cursor = start
        for point in points:
            if not self.is_swing_safe(cursor, point):
                return False
            cursor = point
        return True

    def is_swing_safe(self, start: Position, target: Position) -> bool:
        """Where the arm's own interpolation goes, checked point by point."""
        try:
            path = geometry.pulse_path(start, target, samples=tuning.SWING_SAMPLES)
        except geometry.Unreachable:
            return False
        # The margined envelope, not the firmware's raw limit: the margin is
        # there for sag, and the arm sags just as much halfway through a swing.
        return all(self.zones.is_clear(point) and self.limits.contains(point)
                   and self.is_in_envelope(point) for point in path)

    def _travel_height(self, start: Position, target: Position) -> float:
        """What height to cross at. Two forces pull in opposite directions.

        Never cross below either end of the move, and never within a clearance
        of the desk: the arm sags as it extends, so an XY path can scrape even
        though both endpoints clear the surface.

        The one thing that overrides that is the envelope. Above z ~ 94 the
        reachable radius falls away, so a cruise at the arm's current height
        may be unreachable at the radius it has to cross -- the home pose is
        the everyday case, at z=212 where only a narrow ring is left. Then, and
        only then, the height comes down, and never below the lower endpoint.
        """
        needed = max(math.hypot(start[0], start[1]), math.hypot(target[0], target[1]))
        ceiling = min(self._ceiling, geometry.Z_COMMAND_MAX)
        if self.limits.z[1] is not None:
            ceiling = min(ceiling, self.limits.z[1])

        low_end, high_end = sorted((start[2], target[2]))
        floor = self.limits.z_floor + tuning.DESK_CLEARANCE_MM
        lowest = max(low_end, floor)
        candidate = min(max(high_end, low_end + tuning.DESK_CLEARANCE_MM, floor), ceiling)
        while candidate > lowest:
            span = self.envelope_span(candidate)
            if span is not None and span[0] <= needed <= span[1]:
                return candidate
            candidate -= 2.0
        return max(candidate, lowest)

    def _orbit_radius(self, z: float) -> Optional[float]:
        """An arc radius that clears the zones and that the arm can hold at z."""
        span = self.envelope_span(z)
        if span is None:
            return None
        wanted = self.zones.orbit_radius()
        return max(wanted, span[0]) if wanted <= span[1] else None

    def is_near_desk(self, position: Position) -> bool:
        """Inside the band where a descent is walked rather than flown."""
        return position[2] <= self.limits.z_floor + tuning.DESK_GUARD_MM

    # --- the envelope, with margins ---------------------------------------

    def envelope_span(self, z: float) -> Optional[Tuple[float, float]]:
        """Safe radius range at a height, margins applied. None if nothing fits."""
        span = geometry.radius_span(z)
        if span is None:
            return None
        margin = tuning.ENVELOPE_MARGIN_MM
        inner = max(span[0] + margin, geometry.BLIND_RADIUS + margin)
        outer = span[1] - margin
        return (inner, outer) if inner <= outer else None

    def _into_envelope(self, position: Position) -> Position:
        x, y, z = position
        z = min(z, self._ceiling, geometry.Z_COMMAND_MAX)
        span = self.envelope_span(z)
        while span is None and z > 0.0:
            z -= 2.0            # too high for any radius: come down until it fits
            span = self.envelope_span(z)
        if span is None:
            return position
        fan = geometry.BASE_FAN_DEG - tuning.FAN_MARGIN_DEG
        bearing = max(-fan, min(fan, geometry.bearing_degrees(x, y)))
        radius = min(max(math.hypot(x, y), span[0]), span[1])
        heading = math.radians(bearing)
        return (radius * math.sin(heading), -radius * math.cos(heading), z)

    def is_in_envelope(self, position: Position) -> bool:
        span = self.envelope_span(position[2])
        if span is None:
            return False
        fan = geometry.BASE_FAN_DEG - tuning.FAN_MARGIN_DEG
        return (span[0] - 0.01 <= math.hypot(position[0], position[1]) <= span[1] + 0.01
                and abs(geometry.bearing_degrees(*position[:2])) <= fan + 0.01
                and geometry.is_firmware_reachable(position))


def _halve(start: Position, points: List[Position]) -> List[Position]:
    """Put a midpoint before every point in a chain that begins at `start`."""
    halved: List[Position] = []
    cursor = start
    for point in points:
        halved.append(tuple(round((cursor[i] + point[i]) / 2.0, 2) for i in range(3)))
        halved.append(point)
        cursor = point
    return halved


def _distance(a: Position, b: Position) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _xy_distance(a: Position, b: Position) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
