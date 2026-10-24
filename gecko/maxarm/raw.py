#!/usr/bin/env python3
"""True and raw positions: the library works in true positions; the firmware is sent and reads back raw ones.

The firmware's kinematics gets heights wrong in two ways that every MaxArm
shares. A flat desk reads higher near the base than at full reach: 50 at 100
mm from the base axis, 42 at 260. And anything above the desk reads too
tall, the more the higher it is and the further out: a 120 mm stack of cubes,
3.4 mm at 207 mm out. It gets x and y wrong too: the cup is further out than
it says, most on the desk, and the base points a little off, by an amount that
changes over its turn. The library corrects every target before it is sent
and every reading after it is read:

    true z = raw z + 0.05 x (reach - 207) - (h / 120)^2 x (3.4 + 0.0126 x (reach - 207))
    true reach = reach + 6.6 - 0.118 x min(h, 44)
    true bearing = bearing - 1.36 x sin 2b + 0.33 x (cos 2b - 1), in degrees

    reach = sqrt(x^2 + y^2), the firmware's, mm;  h = true z - 44, the height above the desk, 0 below it;
    b = bearing = the firmware's direction from the base axis: 0 straight out (-y), positive toward +x

On the desk h is 0, and true and raw z agree 207 mm out, where the desk reads
44. Straight out, true and raw point the same way. A target's raw position
follows from its true one directly but for the bearing, which a few rounds
find; a reading's true z is the formula solved for it.

`debug_use_raw(True)` turns the correction off for the whole library: points
in and out of `MaxArm` are then the firmware's own. For testing the firmware;
nothing else should need it.
"""

import math

from .geometry import Position

# On the desk, true z is raw z plus this much per mm of reach past LEVEL_REACH_MM,
# where the two agree.
TRUE_Z_PER_MM_OF_REACH = 0.05
LEVEL_REACH_MM = 207.0
# Above it, raw z reads too tall with the square of the height over the desk:
# by TALL_AT_STACK_MM at STACK_MM up and LEVEL_REACH_MM out, more further out.
DESK_Z_MM = 44.0                   # heights are counted from the desk under an ideally placed arm
STACK_MM = 120.0
TALL_AT_STACK_MM = 3.4
TALL_PER_MM_OF_REACH = 0.0126      # how much more for every mm further out
# x and y: the cup is REACH_AT_DESK_MM further out than the firmware says on the desk, REACH_LESS_PER_MM less for
# every mm above it, no less above REACH_TOP_MM; and the base points WAVE_SIN_DEG x sin 2b + WAVE_COS_DEG x
# (cos 2b - 1) degrees further round than it says, b the firmware's bearing.
REACH_AT_DESK_MM = 6.6
REACH_LESS_PER_MM = 0.118
REACH_TOP_MM = 44.0
WAVE_SIN_DEG = -1.36
WAVE_COS_DEG = 0.33
BEARING_ROUNDS = 8                 # the raw bearing of a true one, found by repeating; each round 20 times closer

_is_raw = False


def debug_use_raw(is_raw: bool) -> None:
    """Raw z everywhere from now on, or back to true z. Process-wide."""
    global _is_raw
    _is_raw = bool(is_raw)


def is_debug_raw() -> bool:
    return _is_raw


def raw_position(true: Position) -> Position:
    if _is_raw:
        return true
    x, y, z = true
    height = max(z - DESK_Z_MM, 0.0)
    x, y = _raw_xy(x, y, height)
    out = math.hypot(x, y) - LEVEL_REACH_MM
    return (x, y, z - TRUE_Z_PER_MM_OF_REACH * out + _too_tall_per_mm2(out) * height ** 2)


def true_position(raw: Position) -> Position:
    """The formula solved for true z -- the height h is what reach alone makes it, less k h^2 -- then x and y."""
    if _is_raw:
        return raw
    x, y, z = raw
    out = math.hypot(x, y) - LEVEL_REACH_MM
    by_reach = z + TRUE_Z_PER_MM_OF_REACH * out - DESK_Z_MM
    if by_reach <= 0.0:
        height = by_reach
    else:
        height = 2.0 * by_reach / (1.0 + math.sqrt(1.0 + 4.0 * _too_tall_per_mm2(out) * by_reach))
    return (*_true_xy(x, y, height), DESK_Z_MM + height)


def describe(true: Position) -> str:
    """'(x, y, z)' with the raw position beside it, for logs: '(0.0, -107.0, 44.0 [raw 0.0, -100.4, 49.3])'."""
    x, y, z = true
    raw_x, raw_y, raw_z = raw_position(true)
    return f"({x:.1f}, {y:.1f}, {z:.1f} [raw {raw_x:.1f}, {raw_y:.1f}, {raw_z:.1f}])"


def _too_tall_per_mm2(out: float) -> float:
    """k in 'h above the desk reads k h^2 too tall', `out` mm further out than the level reach."""
    return (TALL_AT_STACK_MM + TALL_PER_MM_OF_REACH * out) / STACK_MM ** 2


def _true_xy(x: float, y: float, height: float) -> tuple:
    """The firmware's x and y, `height` over the desk, where the cup really is."""
    reach = math.hypot(x, y)
    if reach == 0.0:
        return x, y
    bearing = math.atan2(x, -y)
    reach, bearing = reach + _further_out(height), bearing + _further_round(bearing)
    return reach * math.sin(bearing), -reach * math.cos(bearing)


def _raw_xy(x: float, y: float, height: float) -> tuple:
    """Where the firmware is to be sent for the cup to be at x and y, `height` over the desk."""
    reach = math.hypot(x, y)
    if reach == 0.0:
        return x, y
    bearing = raw_bearing = math.atan2(x, -y)
    for _ in range(BEARING_ROUNDS):
        raw_bearing = bearing - _further_round(raw_bearing)
    reach = max(reach - _further_out(height), 0.0)
    return reach * math.sin(raw_bearing), -reach * math.cos(raw_bearing)


def _further_out(height: float) -> float:
    return REACH_AT_DESK_MM - REACH_LESS_PER_MM * min(max(height, 0.0), REACH_TOP_MM)


def _further_round(bearing: float) -> float:
    """Radians: how much further round the base points than the firmware's `bearing` says."""
    return math.radians(WAVE_SIN_DEG * math.sin(2.0 * bearing) + WAVE_COS_DEG * (math.cos(2.0 * bearing) - 1.0))
