#!/usr/bin/env python3
"""A host-side port of the board's kinematics, used to derive its limits.

The board runs `__espmax.mpy`, which is compiled. Two things make it readable
anyway:

  * the board's copy is byte-identical to `MaxArm/Appendix/9. Handle Control
    Programs/__espmax.mpy`, and
  * the kit ships an Arduino twin of the same library in readable C at
    `MaxArm/8. Inverse Kinematics Basic and Application/Arduino Developement/
    Lesson 4 Move on XYZ Axis/Program File/kinematics_move/_espmax.cpp`,
    with the same L0..L4 and the same algebra.

Two differences from that C, both confirmed against the live board by
`probe-ik-behaviour.py`:

  1. The C flips the sign of x inside `forward`/`inverse`. The .mpy does not --
     `espmax.py` flips it at the call site instead. So the internal frame has
     `xi = -x`, and every branch of the base-angle test reads mirrored.
  2. Out-of-domain `acos()` gives NaN in C but *raises* in MicroPython, and
     `deg_to_pulse` raises `ValueError: Invalid angle` rather than printing.
     So both surface as `set_position()` returning False.

Angles here are the board's "angle1..3" -- servo-frame degrees, not joint
angles in any textbook sense.
"""

import math
from typing import Tuple

# From _espmax.h, and confirmed by the float constants in the .mpy itself.
# NOTE: espmax.py *also* defines L0..L4, with different values (84.0, 8.2,
# 128.0). Those are dead except for ORIGIN -- the IK uses the .mpy's own
# constants and is only ever passed L4. That discrepancy is why the documented
# reset pose (0, -163, 212) reads back as (0, -162, 210).
L0 = 84.4
L1 = 8.14
L2 = 128.4
L3 = 138.0
L4 = 16.8

# Every angle must land in [0, 240] or deg_to_pulse raises.
ANGLE_MIN = 0.0
ANGLE_MAX = 240.0

# set_servo_in_range()'s silent clamps, in pulses.
SERVO_2_MAX_PULSE = 700
SERVO_3_MIN_PULSE = 470

BLIND_RADIUS = 50.0  # set_position returns None inside this
Z_CLAMP = 225.0      # set_position silently pins z here

Position = Tuple[float, float, float]


class Unreachable(ValueError):
    """The board's IK would raise for this target."""


def base_angle(x: float, y: float) -> float:
    """angle1 in degrees, mirroring the .mpy's branch structure exactly.

    Kept branch-for-branch rather than collapsed into atan2 so it can be read
    against the C source; `base_fan_degrees()` below has the tidy form.
    """
    xi, yi = -x, y
    if xi == 0.0:
        theta1 = 90.0 if yi >= 0.0 else 270.0
    elif yi == 0.0:
        theta1 = 0.0 if xi > 0.0 else 180.0
    elif xi < 0.0:
        theta1 = math.degrees(math.atan(yi / xi)) + 180.0
    else:
        theta1 = math.degrees(math.atan(yi / xi)) + 360.0
    if theta1 <= 30.0:
        theta1 += 360.0
    return theta1 - 150.0


def inverse(position: Position) -> Tuple[float, float, float]:
    """(angle1, angle2, angle3) in degrees, or raise, as the board would.

    `position` is in the user frame -- the same coordinates `set_position()`
    takes, not the pre-negated tuple it hands to `__espmax.inverse`.
    """
    x, y, z = position
    angle1 = base_angle(x, y)

    r = math.hypot(x, y) - L1 - L4
    zz = z - L0
    d_squared = r * r + zz * zz
    d = math.sqrt(d_squared)
    if d == 0.0:
        raise Unreachable(f"degenerate reach at {position}")

    # MicroPython's acos raises outside [-1, 1]; C would hand back NaN.
    cos_beta = (L2 * L2 + L3 * L3 - d_squared) / (2.0 * L2 * L3)
    cos_gamma = (L2 * L2 + d_squared - L3 * L3) / (2.0 * L2 * d)
    if not -1.0 <= cos_beta <= 1.0 or not -1.0 <= cos_gamma <= 1.0:
        raise Unreachable(
            f"Unreachable position x:{-x:.2f} y:{y:.2f} z:{zz:.2f}")

    alpha = math.atan(zz / r) if r != 0.0 else math.copysign(math.pi / 2, zz)
    beta = math.acos(cos_beta)
    gamma = math.acos(cos_gamma)
    angle2 = 180.0 - math.degrees(alpha + gamma)
    angle3 = 180.0 - math.degrees(alpha + beta + gamma)
    return angle1, angle2, angle3


def deg_to_pulse(angles: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Servo pulses, raising on any angle outside [0, 240] as the board does."""
    for angle in angles:
        if angle < ANGLE_MIN or angle > ANGLE_MAX:
            raise Unreachable(f"Invalid angle {angle:.0f}")
    angle1, angle2, angle3 = angles
    return (
        angle1 * 1000.0 / 240.0,
        (angle2 - 210.0) * 1000.0 / -240.0,
        (angle3 + 120.0) * 1000.0 / 240.0,
    )


def position_to_pulses(position: Position) -> Tuple[float, float, float]:
    """What `arm.position_to_pulses((-x, y, z))` returns for a user position."""
    return deg_to_pulse(inverse(position))


# --- the limits, stated as predicates -------------------------------------

def base_fan_degrees(x: float, y: float) -> float:
    """Bearing in the user frame, measured so the fan is a plain interval.

    0 deg is straight out the front (-Y). Positive turns toward +X, the arm's
    right. The reachable fan is exactly +/-120 deg, which is the 240 deg the
    base servo has.
    """
    return math.degrees(math.atan2(x, -y))


def angle_margin(position: Position) -> float:
    """Degrees of slack to the nearest joint-angle limit. Negative is outside.

    The board computes in single precision, so within a few 1e-5 degrees of a
    limit its verdict and a float64 one genuinely disagree -- neither is wrong,
    the boundary is just below the arithmetic's resolution. A library should
    keep a margin rather than trusting either. For scale, 1 mm of arc at the
    far edge (r ~ 290) is about 0.2 degrees.
    """
    try:
        angles = inverse(position)
    except Unreachable:
        return float("-inf")
    return min(min(a - ANGLE_MIN, ANGLE_MAX - a) for a in angles)


def limit_reason(position: Position):
    """None if `set_position()` would move there, else why it would not.

    Covers only what the firmware itself enforces: no desk, no no-fly square,
    no gravity sag. Those belong to the caller.
    """
    x, y, z = position
    if math.hypot(x, y) < BLIND_RADIUS:
        return "blind_cylinder"
    if z > Z_CLAMP:
        return "z_clamped"
    try:
        angles = inverse((x, y, z))
    except Unreachable:
        return "out_of_reach"          # acos out of domain: links cannot span it
    try:
        pulses = deg_to_pulse(angles)
    except Unreachable:
        # Which joint ran out tells you what to change about the target.
        names = ("base_angle", "servo2_angle", "servo3_angle")
        for name, angle in zip(names, angles):
            if angle < ANGLE_MIN or angle > ANGLE_MAX:
                return f"{name}_out_of_range"
        return "out_of_range"
    if pulses[1] > SERVO_2_MAX_PULSE:
        return "servo2_clamped"
    if pulses[2] < SERVO_3_MIN_PULSE:
        return "servo3_clamped"
    return None
