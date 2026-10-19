#!/usr/bin/env python3
"""The arm's kinematics, host-side: what the firmware will and will not do.

Pure arithmetic, no board. A target is checked before a byte goes down the
wire, which matters because the firmware lies in one place: `set_position()`
returns True while servo 2 sits silently clamped and the arm goes elsewhere.

Provenance: `anaconda/agenttools/kinematics_reference.py`, checked against the
live board over 336 points. `inverse()`, `base_angle()` and `deg_to_pulse()`
stay branch-for-branch identical to it; do not tidy them without re-running
`anaconda/agenttools/validate-kinematics.py`.

Angles are the board's servo-frame "angle1..3", not textbook joint angles.
"""

import math
from typing import List, Optional, Tuple

Position = Tuple[float, float, float]
Angles = Tuple[float, float, float]

# From _espmax.h, confirmed by the float constants inside the board's .mpy. The
# board's readable espmax.py declares other values; those are dead.
L0 = 84.4    # shoulder height above the origin plane
L1 = 8.14    # base axis to shoulder, radially
L2 = 128.4   # shoulder to elbow
L3 = 138.0   # elbow to wrist
L4 = 16.8    # wrist to nozzle, radially

ANGLE_MIN = 0.0
ANGLE_MAX = 240.0
# set_servo_in_range()'s silent clamps, in pulses. Servo 2's is the lie.
SERVO_2_MAX_PULSE = 700.0
SERVO_3_MIN_PULSE = 470.0

BLIND_RADIUS = 50.0        # set_position() returns None inside this cylinder
Z_CLAMP = 225.0            # above this it pins z and still reports success
BASE_FAN_DEG = 120.0       # the base servo's 240 degrees, centred on straight out

# What the board's go_home() commands, and where every connect leaves the arm.
HOME = (0.0, -163.0, 212.0)


class Unreachable(ValueError):
    """The board's IK would raise for this target."""


def base_angle(x: float, y: float) -> float:
    """angle1 in degrees, mirroring the .mpy's branch structure exactly."""
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


def inverse(position: Position) -> Angles:
    """(angle1, angle2, angle3) in degrees, or raise, as the board would."""
    x, y, z = position
    angle1 = base_angle(x, y)

    r = math.hypot(x, y) - L1 - L4
    zz = z - L0
    d_squared = r * r + zz * zz
    d = math.sqrt(d_squared)
    if d == 0.0:
        raise Unreachable(f"degenerate reach at {position}")

    # MicroPython's acos raises outside [-1, 1]; the Arduino twin returns NaN.
    cos_beta = (L2 * L2 + L3 * L3 - d_squared) / (2.0 * L2 * L3)
    cos_gamma = (L2 * L2 + d_squared - L3 * L3) / (2.0 * L2 * d)
    if not -1.0 <= cos_beta <= 1.0 or not -1.0 <= cos_gamma <= 1.0:
        raise Unreachable(f"links cannot span {position}")

    alpha = math.atan(zz / r) if r != 0.0 else math.copysign(math.pi / 2, zz)
    beta = math.acos(cos_beta)
    gamma = math.acos(cos_gamma)
    angle2 = 180.0 - math.degrees(alpha + gamma)
    angle3 = 180.0 - math.degrees(alpha + beta + gamma)
    return angle1, angle2, angle3


def deg_to_pulse(angles: Angles) -> Tuple[float, float, float]:
    """Servo pulses, raising on any angle outside [0, 240] as the board does."""
    for angle in angles:
        if angle < ANGLE_MIN or angle > ANGLE_MAX:
            raise Unreachable(f"invalid angle {angle:.1f}")
    angle1, angle2, angle3 = angles
    return (
        angle1 * 1000.0 / 240.0,
        (angle2 - 210.0) * 1000.0 / -240.0,
        (angle3 + 120.0) * 1000.0 / 240.0,
    )


def pulse_to_deg(pulses: Tuple[float, float, float]) -> Angles:
    pulse1, pulse2, pulse3 = pulses
    return (pulse1 * 240.0 / 1000.0, 210.0 - pulse2 * 240.0 / 1000.0, pulse3 * 240.0 / 1000.0 - 120.0)


def forward(angles: Angles) -> Position:
    """The tip for a set of servo-frame angles: the inverse of `inverse()`.

    `180 - angle2` is the upper arm's elevation and `-angle3` the forearm's.
    """
    angle1, angle2, angle3 = angles
    upper, fore = math.radians(180.0 - angle2), math.radians(-angle3)
    radius = L1 + L2 * math.cos(upper) + L3 * math.cos(fore) + L4
    heading = math.radians(BASE_FAN_DEG - angle1)
    return (radius * math.sin(heading), -radius * math.cos(heading),
            L0 + L2 * math.sin(upper) + L3 * math.sin(fore))


def real_path(start: Position, end: Position, samples: int = 24) -> List[Position]:
    """Where the tip really goes on one `set_position()` from `start` to `end`.

    The board solves the end pose and gives all three servos the same duration,
    so their pulses move together in a straight line -- and the tip does not.
    """
    begin, finish = deg_to_pulse(inverse(start)), deg_to_pulse(inverse(end))
    return [forward(pulse_to_deg(tuple(b + (f - b) * step / samples
                                       for b, f in zip(begin, finish))))
            for step in range(samples + 1)]


def true_position(position: Position, offsets: Angles) -> Position:
    """Where the cup really is when the firmware is sent `position`, if its servos sit `offsets`
    degrees off the zeros the firmware assumes. Points the arm cannot reach are left as they are."""
    if not any(offsets):
        return position
    try:
        angles = inverse(position)
    except Unreachable:
        return position
    return forward(tuple(angle + offset for angle, offset in zip(angles, offsets)))


def firmware_position(position: Position, offsets: Angles) -> Position:
    """What to send the firmware for the cup to really go to `position`: `true_position()` undone."""
    if not any(offsets):
        return position
    try:
        angles = inverse(position)
    except Unreachable:
        return position
    return forward(tuple(angle - offset for angle, offset in zip(angles, offsets)))


def bearing_degrees(x: float, y: float) -> float:
    """0 straight out from the arm (-y), positive towards +x. The fan is +-120."""
    return math.degrees(math.atan2(x, -y))


def limit_reason(position: Position) -> Optional[str]:
    """None if `set_position()` would really go there, else why it would not."""
    x, y, z = position
    if math.hypot(x, y) < BLIND_RADIUS:
        return "blind_cylinder"
    if z > Z_CLAMP:
        return "z_clamped"
    try:
        angles = inverse(position)
    except Unreachable:
        return "out_of_reach"
    try:
        pulses = deg_to_pulse(angles)
    except Unreachable:
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
