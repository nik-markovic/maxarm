#!/usr/bin/env python3
"""True and raw height: the library works in true z; the firmware is sent and reads back raw z.

The firmware's kinematics puts the cup higher than it says near the base and
lower at full reach: a flat desk reads 50 at 100 mm from the base axis and 42
at 260. Every MaxArm is built the same, so the library corrects every target
before it is sent and every reading after it is read:

    true z = raw z + 0.05 x (reach - 207)        reach = sqrt(x^2 + y^2), mm
    raw z  = true z - 0.05 x (reach - 207)

Only z: x and y are the firmware's. At 207 mm from the base axis true and raw
agree, which is where the owner's desk reads 44.

`debug_use_raw(True)` turns the correction off for the whole library: points
in and out of `MaxArm` are then the firmware's own. For testing the firmware;
nothing else should need it.
"""

import math

from .geometry import Position

# Fitted on the owner's 66 desk touch points (STATUS-falcon), 2026-10-20: one
# straight line by reach, the same at every height. With it the desk reads
# 44.0 on average, 0.5 mm RMS, 1.4 worst; the 37 latest points alone give
# 0.0498 and 207.3.
TRUE_Z_PER_MM_OF_REACH = 0.05
LEVEL_REACH_MM = 207.0

_is_raw = False


def debug_use_raw(is_raw: bool) -> None:
    """Raw z everywhere from now on, or back to true z. Process-wide."""
    global _is_raw
    _is_raw = bool(is_raw)


def is_debug_raw() -> bool:
    return _is_raw


def true_minus_raw(x: float, y: float) -> float:
    """How much higher the cup really is than the firmware says, at (x, y). 0 under `debug_use_raw()`."""
    if _is_raw:
        return 0.0
    return TRUE_Z_PER_MM_OF_REACH * (math.hypot(x, y) - LEVEL_REACH_MM)


def true_position(raw: Position) -> Position:
    x, y, z = raw
    return (x, y, z + true_minus_raw(x, y))


def raw_position(true: Position) -> Position:
    x, y, z = true
    return (x, y, z - true_minus_raw(x, y))


def describe(true: Position) -> str:
    """'(x, y, z)' with the raw z beside it, for logs: '(0.0, -107.0, 44.0 [raw 49.0])'."""
    x, y, z = true
    return f"({x:.1f}, {y:.1f}, {z:.1f} [raw {raw_position(true)[2]:.1f}])"
