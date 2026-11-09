#!/usr/bin/env python3
"""How the cup is brought down to a point as precisely as the arm lands: the base's last turn and the descent.
No board.

A servo stops short of a command, in the direction it was moving, by an amount set by the move: how far, which
way, which servo (work/MAXARM_MECHANICS.md §4-7). A one-pulse correction is ignored. So a landing is made the same
way every time, and aimed past by what that way falls short:

1. across, at the travel height, to APPROACH_PULSES of the base to one side of the target: the side of higher
   pulses, so that the last turn is made with them falling, where the planner allows;
2. the base turned onto the target from there, aimed past it by AIM_PAST_* pulses;
3. straight down in one slow command, servos 2 and 3 given EXTRA_PULSES -- by reach -- beyond the target's.

The target is the point asked for, its height included. What the library keeps as sent is that target, never the
aimed-past command.

Positions in and out are arm space, true (`raw.py`). The pulses are the firmware's, before its cut to whole ones
(`geometry.nearest_step_command()` rounds them).
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

from . import geometry, raw
from .geometry import Position

# How far to one side of the target, in base pulses, the last turn starts: the same turn every time.
APPROACH_PULSES = 20
# How far past the target the base is aimed, in pulses: turning onto it with its pulses falling, and rising.
AIM_PAST_FALLING = 1
AIM_PAST_RISING = 2
# Servo 2's and servo 3's extra pulses on the way down, by the target's reach in mm: straight between rows, held
# beyond the ends. The owner's arm, measured 6 Nov (MAXARM_MECHANICS.md §7).
EXTRA_PULSES = ((120.0, 0.0, 3.1), (140.0, -0.25, 3.05), (160.0, -0.5, 2.9), (175.0, -0.5, 2.6),
                (190.0, -0.5, 2.0), (205.0, -0.5, 2.0), (220.0, -0.5, 2.25), (235.0, -0.5, 2.1),
                (250.0, -0.5, 2.0), (265.0, -0.5, 2.0))
# A landing whose base settles further than this from the target's is lifted and made again, the same way.
BASE_OFF_PULSES = 1.0


@dataclass(frozen=True)
class Landing:
    approach: Position       # at the travel height, the base APPROACH_PULSES to one side
    over: Position           # at the travel height, the base aimed past the target
    down: Position           # what the descent is sent to: the target, servos 2 and 3 beyond it
    target: Position         # where the cup is to end: the point asked for
    base: float              # the target's exact base pulse


def plans(target: Position, travel_z: float) -> List[Landing]:
    """The two ways down to `target` from arm z `travel_z`, the last turn with the base's pulses falling first: the
    aim for a rising turn held on the centre line only. Empty if the firmware's kinematics cannot take the target."""
    try:
        exact = _pulses(target)
        high = _pulses((target[0], target[1], travel_z))
    except geometry.Unreachable:
        return []
    sides = [1, -1]
    extra2, extra3 = extra_pulses(math.hypot(target[0], target[1]))
    ways = []
    for side in sides:                      # +1: from higher pulses, turning with them falling
        aimed = exact[0] - side * (AIM_PAST_FALLING if side > 0 else AIM_PAST_RISING)
        ways.append(Landing(approach=_point((exact[0] + side * APPROACH_PULSES, high[1], high[2])),
                            over=_point((aimed, high[1], high[2])),
                            down=_point((aimed, exact[1] + extra2, exact[2] + extra3)),
                            target=target, base=exact[0]))
    return ways


def extra_pulses(reach: float) -> Tuple[float, float]:
    """EXTRA_PULSES at `reach`, mm."""
    rows = EXTRA_PULSES
    if reach <= rows[0][0]:
        return rows[0][1], rows[0][2]
    for (near, near2, near3), (far, far2, far3) in zip(rows, rows[1:]):
        if reach <= far:
            share = (reach - near) / (far - near)
            return near2 + (far2 - near2) * share, near3 + (far3 - near3) * share
    return rows[-1][1], rows[-1][2]


def _pulses(point: Position) -> Tuple[float, float, float]:
    """The firmware's exact pulses for a true arm-space point."""
    return geometry.deg_to_pulse(geometry.inverse(raw.raw_position(point)))


def _point(pulses: Tuple[float, float, float]) -> Position:
    """The true arm-space point the firmware takes to `pulses`."""
    return raw.true_position(geometry.forward(geometry.pulse_to_deg(pulses)))
