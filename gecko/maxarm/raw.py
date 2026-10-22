#!/usr/bin/env python3
"""True and raw height: the library works in true z; the firmware is sent and reads back raw z.

The firmware's kinematics gets heights wrong in two ways that every MaxArm
shares. A flat desk reads higher near the base than at full reach: 50 at 100
mm from the base axis, 42 at 260. And anything above the desk reads too
tall, the more the higher it is and the further out: a 120 mm stack of cubes,
3.4 mm at 207 mm out. The library corrects every target before it is sent
and every reading after it is read:

    true z = raw z + 0.05 x (reach - 207) - (h / 120)^2 x (3.4 + 0.0126 x (reach - 207))

    reach = sqrt(x^2 + y^2), mm;  h = true z - 44, the height above the desk, 0 below it

Only z: x and y are the firmware's. On the desk h is 0, and true and raw agree
207 mm out, where the desk reads 44. A target's raw z follows from its true z
directly; a reading's true z is the same formula solved for it.

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
    out = math.hypot(x, y) - LEVEL_REACH_MM
    height = max(z - DESK_Z_MM, 0.0)
    return (x, y, z - TRUE_Z_PER_MM_OF_REACH * out + _too_tall_per_mm2(out) * height ** 2)


def true_position(raw: Position) -> Position:
    """The formula solved for true z: the height h is what reach alone makes it, less k h^2."""
    if _is_raw:
        return raw
    x, y, z = raw
    out = math.hypot(x, y) - LEVEL_REACH_MM
    by_reach = z + TRUE_Z_PER_MM_OF_REACH * out - DESK_Z_MM
    if by_reach <= 0.0:
        return (x, y, DESK_Z_MM + by_reach)
    height = 2.0 * by_reach / (1.0 + math.sqrt(1.0 + 4.0 * _too_tall_per_mm2(out) * by_reach))
    return (x, y, DESK_Z_MM + height)


def describe(true: Position) -> str:
    """'(x, y, z)' with the raw z beside it, for logs: '(0.0, -107.0, 44.0 [raw 49.0])'."""
    x, y, z = true
    return f"({x:.1f}, {y:.1f}, {z:.1f} [raw {raw_position(true)[2]:.1f}])"


def _too_tall_per_mm2(out: float) -> float:
    """k in 'h above the desk reads k h^2 too tall', `out` mm further out than the level reach."""
    return (TALL_AT_STACK_MM + TALL_PER_MM_OF_REACH * out) / STACK_MM ** 2
