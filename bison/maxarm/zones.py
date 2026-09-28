#!/usr/bin/env python3
"""Exclusion zones: places the nozzle must not go, and paths it must not take.

The firmware knows nothing about any of this. The one zone that is not
negotiable is the square around the robot's own base: inside it the suction cup
or its air hose can strike the base casting. It is hardcoded, applies at every
height, and no configuration switches it off. Operator zones (a camera mount, a
mug, the edge of the mat) sit on top of it and are entirely configurable.

A zone is a path constraint, not just a position one. Two perfectly legal poses
can have a straight line between them that passes right over the base -- which
is exactly how an early sweep drove the cup across the robot. So every segment
is tested, not just its endpoints.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .geometry import BASE_FAN_DEG, BLIND_RADIUS, bearing_degrees
from .tuning import NO_FLY_MARGIN_MM, ORBIT_STEP_DEG

Position = Tuple[float, float, float]

BASE_HALF_MM = 70.0


@dataclass(frozen=True)
class ExclusionZone:
    """An axis-aligned XY box the nozzle may not enter.

    `z_max` limits the zone to heights at or below it -- useful for an obstacle
    the arm can simply fly over. `None` means the zone applies at every height,
    which is the only correct answer for the base square, since the cup's hose
    trails behind and above it.
    """

    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_max: Optional[float] = None

    def contains(self, position: Position, margin: float = 0.0) -> bool:
        x, y, z = position
        if self.z_max is not None and z > self.z_max:
            return False
        return (self.x_min - margin <= x <= self.x_max + margin
                and self.y_min - margin <= y <= self.y_max + margin)

    def is_segment_blocked(self, start: Position, end: Position, margin: float = 0.0) -> bool:
        """Slab test of the XY segment against the box, grown by `margin`.

        Only the lower of the two endpoint heights is considered against
        `z_max`: a move that descends into a zone's height band has to be
        caught, and a move entirely above it is genuinely free.
        """
        if self.z_max is not None and min(start[2], end[2]) > self.z_max:
            return False

        low, high = 0.0, 1.0
        for axis, (near_edge, far_edge) in enumerate(
                ((self.x_min - margin, self.x_max + margin),
                 (self.y_min - margin, self.y_max + margin))):
            origin, delta = start[axis], end[axis] - start[axis]
            if abs(delta) < 1e-9:
                if near_edge <= origin <= far_edge:
                    continue          # parallel and inside the slab: no verdict yet
                return False          # parallel and outside it: never enters
            near = (near_edge - origin) / delta
            far = (far_edge - origin) / delta
            if near > far:
                near, far = far, near
            low, high = max(low, near), min(high, far)
            if low > high:
                return False          # slabs do not overlap: no crossing
        return True

    @property
    def corner_radius(self) -> float:
        """Distance from the base axis to the furthest corner."""
        return self.expanded_corner_radius(0.0)

    def expanded_corner_radius(self, margin: float) -> float:
        """Corner distance once the box is grown by `margin`.

        An arc has to clear the *grown* box, and its corners are further out
        than the margin itself: growing a 70 mm square by 8 mm moves the corner
        from 99 mm to 110 mm, not to 107.
        """
        return max(math.hypot(x + math.copysign(margin, x or 1.0),
                              y + math.copysign(margin, y or 1.0))
                   for x in (self.x_min, self.x_max)
                   for y in (self.y_min, self.y_max))


# The robot's own base. Every height, and not configurable: `ZoneSet` puts it
# in front of the operator's zones no matter what it is handed.
BASE_ZONE = ExclusionZone("base", -BASE_HALF_MM, BASE_HALF_MM,
                          -BASE_HALF_MM, BASE_HALF_MM)


class ZoneSet:
    """The base square plus whatever the operator added. Purely geometric."""

    def __init__(self, extra_zones: Sequence[ExclusionZone] = ()) -> None:
        self.zones: Tuple[ExclusionZone, ...] = (BASE_ZONE,) + tuple(extra_zones)
        self.margin_mm = NO_FLY_MARGIN_MM

    def find_blocking(self, position: Position, margin: Optional[float] = None) -> Optional[ExclusionZone]:
        margin = self.margin_mm if margin is None else margin
        for zone in self.zones:
            if zone.contains(position, margin):
                return zone
        return None

    def is_clear(self, position: Position, margin: Optional[float] = None) -> bool:
        return self.find_blocking(position, margin) is None

    def find_blocking_path(self, start: Position, end: Position,
                           margin: Optional[float] = None) -> Optional[ExclusionZone]:
        margin = self.margin_mm if margin is None else margin
        for zone in self.zones:
            if zone.is_segment_blocked(start, end, margin):
                return zone
        return None

    def is_path_clear(self, start: Position, end: Position,
                      margin: Optional[float] = None) -> bool:
        return self.find_blocking_path(start, end, margin) is None

    def orbit_radius(self, arc_step_deg: float = ORBIT_STEP_DEG) -> float:
        """Radius of an arc whose *chords* clear every zone's corners.

        Only zones that actually surround the base axis matter for an orbit; an
        operator zone off to one side is handled by the path check instead.

        The arc is flown as straight chords, and a chord across `arc_step_deg`
        passes a factor of `cos(step/2)` closer in than the radius it was drawn
        at. Ignoring that is what made a 45-degree arc cut the corner of the
        base square: the waypoints cleared it by 4 mm and the chords between
        them passed 5 mm inside it. So the radius is the corner distance
        divided back out by that factor, plus a little for the couple of
        millimetres the arm lands away from its command.
        """
        surrounding = [zone for zone in self.zones
                       if zone.contains((0.0, 0.0, 0.0), self.margin_mm)]
        if not surrounding:
            return BLIND_RADIUS + self.margin_mm
        corner = max(zone.expanded_corner_radius(self.margin_mm) for zone in surrounding)
        return corner / math.cos(math.radians(min(arc_step_deg, 120.0) / 2.0)) + 4.0

    def route_around(self, start: Position, target: Position,
                     orbit_radius: Optional[float] = None,
                     arc_step_deg: float = ORBIT_STEP_DEG) -> List[Position]:
        """Waypoints from `start` to `target` that keep clear of the zones.

        Radially onto an orbit beyond the zone corners, an arc round to the
        target's bearing, then radially back out. Both radial legs run along a
        ray and end at or beyond the orbit radius, which by construction clears
        every corner, so neither can enter a zone; the arc stays outside them
        all by definition.

        The first radial leg is not optional even when the arm starts further
        out than the orbit. Going straight from a distant start to the first
        arc point is a chord across two different bearings, and that chord cuts
        the corner -- measured, with a start at r=161 and an orbit at r=114.

        The arc is swept in *bearing*, not in atan2 angle, so it can never take
        the short way round through the arm's blind rear quadrant: bearings are
        confined to the base fan and sweeping between two of them monotonically
        stays inside it.
        """
        if self.is_path_clear(start, target):
            return [target]

        radius = orbit_radius if orbit_radius is not None else self.orbit_radius()
        height = start[2]
        start_bearing = bearing_degrees(start[0], start[1])
        end_bearing = bearing_degrees(target[0], target[1])

        waypoints: List[Position] = []
        if abs(math.hypot(start[0], start[1]) - radius) > 1.0:
            waypoints.append(_polar(radius, start_bearing, height))

        sweep = end_bearing - start_bearing
        arc_steps = max(1, int(math.ceil(abs(sweep) / max(arc_step_deg, 1.0))))
        for step in range(1, arc_steps + 1):
            bearing = start_bearing + sweep * step / arc_steps
            waypoints.append(_polar(radius, _clamp_to_fan(bearing), height))
        waypoints.append(target)
        return [tuple(round(value, 2) for value in point) for point in waypoints]

    def escape_waypoint(self, position: Position) -> Optional[Position]:
        """Where to go if the arm is somehow already inside a zone.

        Straight out along the current bearing, which is the shortest way out
        and cannot drive deeper in. `None` when there is no bearing to follow,
        i.e. the nozzle is sitting on the rotation axis.
        """
        zone = self.find_blocking(position, margin=0.0)
        if zone is None:
            return None
        x, y, z = position
        reach = math.hypot(x, y)
        if reach < 1.0:
            return None
        clear_radius = zone.expanded_corner_radius(self.margin_mm) + 4.0
        scale = clear_radius / reach
        return (round(x * scale, 2), round(y * scale, 2), z)


def _polar(radius: float, bearing_deg: float, z: float) -> Position:
    heading = math.radians(bearing_deg)
    return (radius * math.sin(heading), -radius * math.cos(heading), z)


def _clamp_to_fan(bearing_deg: float) -> float:
    return max(-BASE_FAN_DEG, min(BASE_FAN_DEG, bearing_deg))
