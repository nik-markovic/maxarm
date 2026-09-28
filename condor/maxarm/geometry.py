#!/usr/bin/env python3
"""The arm's kinematics and the limits the *firmware* enforces, host-side.

Everything here is pure arithmetic: no board, no motion, no I/O. That is the
point -- a target can be accepted or rejected, and the rejection named, before
a single byte goes down the wire. Under any transport that does not report
`set_position()`'s verdict back (the 0xAA 0x55 one does not), this is the only
way to know what the arm will do.

Provenance: `anaconda/agenttools/kinematics_reference.py`, which was validated
against the live board over 336 points -- all agreeing bar 8 that sat within
2.4e-5 degrees of a joint limit, where the board's single-precision maths and a
host's double-precision maths honestly differ. `inverse()`, `base_angle()` and
`deg_to_pulse()` are kept branch-for-branch identical to it on purpose; do not
tidy them without re-running `anaconda/agenttools/validate-kinematics.py`.

`forward()` and `joint_frame()` are new here: the reference only needed to
reject targets, while a UI also has to *draw* the arm.

Angles are the board's "angle1..3" -- servo-frame degrees, not joint angles in
any textbook sense.
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

# From _espmax.h, confirmed by the float constants inside the board's .mpy.
# NOTE: the board's readable espmax.py *also* defines L0..L2, with different
# values (84.0, 8.2, 128.0). Those are dead except for ORIGIN.
L0 = 84.4    # shoulder height above the desk-facing origin plane
L1 = 8.14    # base axis to shoulder, radially
L2 = 128.4   # shoulder to elbow
L3 = 138.0   # elbow to wrist
L4 = 16.8    # wrist to nozzle tip, radially outward

ANGLE_MIN = 0.0
ANGLE_MAX = 240.0

# set_servo_in_range()'s silent clamps, in pulses. Servo 2 is the only place
# the board lies: it pins the pulse and still returns True.
SERVO_2_MAX_PULSE = 700.0
SERVO_3_MIN_PULSE = 470.0

BLIND_RADIUS = 50.0        # set_position() returns None inside this cylinder
Z_CLAMP = 225.0            # above this it silently pins z and reports success
Z_COMMAND_MAX = 224.0      # so never command above this
MAX_LINK_REACH = L1 + L2 + L3 + L4    # 291.34 mm, geometric ceiling

BASE_FAN_DEG = 120.0       # the base servo's 240 deg, centred on straight front
HOME_ANGLES = (120.0, 90.0, 0.0)
# What go_home() commands (espmax.ORIGIN, built from the *dead* constants).
# The pose it actually lands in reads back as about (0, -162, 210).
HOME_COMMAND = (0.0, -163.0, 212.0)

Position = Tuple[float, float, float]
Angles = Tuple[float, float, float]
Pulses = Tuple[float, float, float]


class Unreachable(ValueError):
    """The board's IK would raise for this target."""


@dataclass(frozen=True)
class JointFrame:
    """Where every joint sits in space -- enough to draw the arm.

    `base` is on the rotation axis at z=0, and the chain runs
    base -> shoulder -> elbow -> wrist -> tip. A renderer can just draw the
    polyline; `tip` is the pose `read_position()` reports.
    """
    angles: Angles
    bearing_deg: float
    base: Position
    shoulder: Position
    elbow: Position
    wrist: Position
    tip: Position

    @property
    def chain(self) -> List[Position]:
        return [self.base, self.shoulder, self.elbow, self.wrist, self.tip]


def base_angle(x: float, y: float) -> float:
    """angle1 in degrees, mirroring the .mpy's branch structure exactly.

    Kept as branches rather than collapsed into atan2 so it can be read against
    the kit's Arduino source; `bearing_degrees()` below has the tidy form.
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


def inverse(position: Position) -> Angles:
    """(angle1, angle2, angle3) in degrees, or raise, as the board would.

    `position` is in the user frame -- the coordinates `set_position()` takes,
    not the pre-negated tuple it hands to `__espmax.inverse`.
    """
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


def forward(angles: Angles) -> Position:
    """Tip position for a set of servo-frame angles. The inverse of inverse().

    Falls out of the same algebra: `180 - angle2` is the elbow link's elevation
    in the radial plane and `-angle3` is the forearm's, so the chain closes
    without any extra constants.
    """
    return joint_frame(angles).tip


def joint_frame(angles: Angles) -> JointFrame:
    """Full joint chain in 3D, for rendering or collision work."""
    angle1, angle2, angle3 = angles
    bearing = BASE_FAN_DEG - angle1
    upper = math.radians(180.0 - angle2)
    fore = math.radians(-angle3)

    shoulder_r, shoulder_z = L1, L0
    elbow_r = shoulder_r + L2 * math.cos(upper)
    elbow_z = shoulder_z + L2 * math.sin(upper)
    wrist_r = elbow_r + L3 * math.cos(fore)
    wrist_z = elbow_z + L3 * math.sin(fore)

    def place(radius: float, z: float) -> Position:
        heading = math.radians(bearing)
        return (radius * math.sin(heading), -radius * math.cos(heading), z)

    return JointFrame(
        angles=(angle1, angle2, angle3),
        bearing_deg=bearing,
        base=(0.0, 0.0, 0.0),
        shoulder=place(shoulder_r, shoulder_z),
        elbow=place(elbow_r, elbow_z),
        wrist=place(wrist_r, wrist_z),
        tip=place(wrist_r + L4, wrist_z),
    )


def deg_to_pulse(angles: Angles) -> Pulses:
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


def pulse_to_deg(pulses: Pulses) -> Angles:
    """Servo-frame angles for measured pulses -- what a joint readout needs."""
    pulse1, pulse2, pulse3 = pulses
    return (
        pulse1 * 240.0 / 1000.0,
        210.0 - pulse2 * 240.0 / 1000.0,
        pulse3 * 240.0 / 1000.0 - 120.0,
    )


def position_to_pulses(position: Position) -> Pulses:
    """What `arm.position_to_pulses((-x, y, z))` returns for a user position."""
    return deg_to_pulse(inverse(position))


def pulse_path(start: Position, end: Position, samples: int = 12) -> List[Position]:
    """Where the tip actually goes when commanded straight from one pose to another.

    `set_position()` solves the IK for the *endpoint* and hands all three
    servos the same duration, so they interpolate **linearly in pulse space**.
    The tip therefore does not travel in a straight line -- it swings. Measured
    with this function: a short hop bows about 2 mm off the straight line, a
    long one 29 mm, and a reach across the front of the base 123 mm.

    That is why a move is not simply commanded in one go: the bow has to be
    checked against the desk, the exclusion zones and the operator's limits
    first. When it is clear, one command is better in every way -- continuous
    motion, no stutter, one round trip.
    """
    from_pulses = position_to_pulses(start)
    to_pulses = position_to_pulses(end)
    path = []
    for step in range(samples + 1):
        fraction = step / samples
        pulses = tuple(from_pulses[axis] + (to_pulses[axis] - from_pulses[axis]) * fraction
                       for axis in range(3))
        path.append(forward(pulse_to_deg(pulses)))
    return path


def path_deviation(start: Position, end: Position, samples: int = 12) -> float:
    """How far the commanded swing departs from the straight line, in mm."""
    worst = 0.0
    path = pulse_path(start, end, samples)
    for step, point in enumerate(path):
        fraction = step / samples
        straight = tuple(start[axis] + (end[axis] - start[axis]) * fraction
                         for axis in range(3))
        worst = max(worst, math.dist(point, straight))
    return worst


def bearing_degrees(x: float, y: float) -> float:
    """Bearing in the user frame, measured so the base fan is a plain interval.

    0 is straight out the front (-Y); positive turns toward +X, the arm's right.
    The reachable fan is exactly +/-120 degrees.
    """
    return math.degrees(math.atan2(x, -y))


def angle_margin(position: Position) -> float:
    """Degrees of slack to the nearest joint-angle limit. Negative is outside.

    For scale, 1 mm of arc at full reach (r ~ 290) is about 0.2 degrees, so a
    1 degree planning margin is nearly free.
    """
    try:
        angles = inverse(position)
    except Unreachable:
        return float("-inf")
    return min(min(a - ANGLE_MIN, ANGLE_MAX - a) for a in angles)


def limit_reason(position: Position) -> Optional[str]:
    """None if `set_position()` would move there, else why it would not.

    Covers only what the firmware itself enforces. The desk, the base exclusion
    square and gravity sag are ours -- see `zones.py` and `route.py`.
    """
    x, y, z = position
    if math.hypot(x, y) < BLIND_RADIUS:
        return "blind_cylinder"
    if z > Z_CLAMP:
        return "z_clamped"
    try:
        angles = inverse((x, y, z))
    except Unreachable:
        return "out_of_reach"
    try:
        pulses = deg_to_pulse(angles)
    except Unreachable:
        # Which joint ran out tells the caller what to change about the target.
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


def is_firmware_reachable(position: Position) -> bool:
    return limit_reason(position) is None


@lru_cache(maxsize=4096)
def radius_span(z: float, bearing_deg: float = 0.0,
                resolution_mm: float = 0.05) -> Optional[Tuple[float, float]]:
    """Reachable radius range at one height, as (r_min, r_max), or None.

    Measured on the arm: at a fixed Z the reach radius does not depend on
    bearing (spread 0.5-1.2 mm against a 2.6 mm noise floor), so one bearing
    answers for the whole fan and the envelope is an annulus cut by the fan.

    Found by scan-then-bisect rather than in closed form because only the lower
    regime has one: below z ~ 94 the link span binds and
    `sqrt(266.4^2 - (z - 84.4)^2) + 24.94` is exact, but above it servo 3 runs
    out of angle and there is no elementary expression. Bisection covers both
    and cannot drift away from `limit_reason()`, which is the validated part.
    """
    heading = math.radians(bearing_deg)

    def is_ok(radius: float) -> bool:
        return is_firmware_reachable(
            (radius * math.sin(heading), -radius * math.cos(heading), z))

    coarse = 2.0
    best: Optional[Tuple[float, float]] = None
    run_start: Optional[float] = None
    radius = BLIND_RADIUS
    while radius <= MAX_LINK_REACH + coarse:
        if is_ok(radius):
            if run_start is None:
                run_start = radius
            last_ok = radius
        elif run_start is not None:
            if best is None or last_ok - run_start > best[1] - best[0]:
                best = (run_start, last_ok)
            run_start = None
        radius += coarse
    if run_start is not None and (best is None or last_ok - run_start > best[1] - best[0]):
        best = (run_start, last_ok)
    if best is None:
        return None

    inner = _bisect_edge(best[0] - coarse, best[0], is_ok, resolution_mm)
    outer = _bisect_edge(best[1] + coarse, best[1], is_ok, resolution_mm)
    return (max(inner, BLIND_RADIUS), outer)


def _bisect_edge(bad: float, good: float, is_ok, resolution_mm: float) -> float:
    """Narrow a bracket down to the last radius that still works."""
    for _ in range(64):
        if abs(good - bad) <= resolution_mm:
            break
        middle = (good + bad) / 2.0
        if is_ok(middle):
            good = middle
        else:
            bad = middle
    return good


@lru_cache(maxsize=8)
def max_reachable_z(resolution_mm: float = 0.05) -> float:
    """Highest z with any reachable radius. About 214 mm on this arm."""
    low, high = 0.0, Z_COMMAND_MAX
    for _ in range(64):
        if high - low <= resolution_mm:
            break
        middle = (low + high) / 2.0
        if radius_span(middle, resolution_mm=0.5) is not None:
            low = middle
        else:
            high = middle
    return low
