#!/usr/bin/env python3
"""The cup's turn: which way it faces, and how long it takes to get there.

The cup turns about z on its own servo, -90 to +90 degrees. The arm has no yaw
of its own -- the whole arm swings to reach a point -- so the cup faces the
arm's bearing plus the servo's angle, and a piece held under it faces that
plus however it was picked up: its offset.

Directions here are arm-space degrees, from +x towards +y. With the cup at 0
straight out in front, it faces +x: square to the arm's axes.
"""

from typing import Optional

from .geometry import Position, bearing_degrees

# The cup swings its full 180 degrees in about 1.2 s, measured (condor). That is
# the servo's own rate: a shorter duration only sends the pulse ahead of the horn.
NOZZLE_TURN_DEG_PER_S = 150.0
# The board ramps the cup on a 20 ms timer and stretches anything shorter to that.
NOZZLE_TURN_MIN_MS = 20
# The cup reports nothing, so its arrival is a wait. A guess: raise it if the
# cup is still turning when the arm moves on.
NOZZLE_SETTLE_MS = 150
NOZZLE_RANGE_DEG = 90.0
# A direction reaches the servo through two coordinate conversions, which can
# leave the end of the range a few 1e-13 degrees past it: that is still the end.
NOZZLE_RANGE_SLACK_DEG = 1e-6


def servo_for(point: Position, facing: float, offset: float = 0.0) -> float:
    """The servo angle that turns a piece held at `offset` -- or the empty cup -- to `facing`, at `point`."""
    return _wrapped(facing - offset - bearing_degrees(point[0], point[1]))


def offset_of(point: Position, facing: float, servo: float) -> float:
    """How a piece facing `facing` sits under the cup at `servo`: its direction less the cup's."""
    return facing - bearing_degrees(point[0], point[1]) - servo


def turn_ms(start: float, end: float) -> int:
    """How long the servo is given to turn from one angle to another; 0 for no turn."""
    if abs(end - start) < 0.5:
        return 0
    return max(NOZZLE_TURN_MIN_MS, int(abs(end - start) / NOZZLE_TURN_DEG_PER_S * 1000.0))


def problem_with(servo: Optional[float]) -> Optional[str]:
    if servo is None or abs(servo) <= NOZZLE_RANGE_DEG + NOZZLE_RANGE_SLACK_DEG:
        return None
    return (f"the cup would have to turn to {servo:+.0f} deg; it turns "
            f"{NOZZLE_RANGE_DEG:.0f} either way from centre")


def _wrapped(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0
