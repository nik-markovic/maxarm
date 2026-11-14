#!/usr/bin/env python3
"""How far the base stops from its command at the end of a landing, by bearing, reach, approach side and how far
aside the last turn starts. MOVES THE ARM: the owner runs it, or releases the arm for it.

Each trial is the library's landing with the aim past taken out, so the shortfall shows bare: at the travel height
the base `swing` pulses to one side, turned onto the commanded pulse C, straight down with the shoulder and elbow
as `landing.py` does, read until steady (the library's PRECISE read), back up. `settled - C` is the shortfall:
+ means more pulses than commanded. Then `nudge`: from a steady landing the base alone stepped one pulse at a time
across a stretch, the cup held at the landing height, read steady after each step: what a manual nudge does.

One line per trial into agenttools/probe/out/base-shortfall.tsv as it goes.

    PYTHONPATH=src ../.venv/bin/python3 agenttools/probe/base-shortfall.py sweep
    PYTHONPATH=src ../.venv/bin/python3 agenttools/probe/base-shortfall.py nudge
"""
import sys
import time
from pathlib import Path

from maxarm import MaxArm, Space, geometry, landing, raw
from maxarm.arm import MOVE_FAST_MM_S, MOVE_SLOW_MM_S

Z = 86.0                                   # true; 2 mm over a 40 mm cube's top
TRAVEL_UP_MM = 5.0
BASES = (116, 250, 500, 750)               # ~+92 (just north of real east), +60, straight out, -60 deg
REACHES = (180.0, 250.0)
SIDES = (1, -1)                            # +1 from higher pulses (turning with them falling), -1 rising
SWINGS = (6, 20, 60)
REPEATS = 2
OUT = Path(__file__).parent / "out" / "base-shortfall.tsv"


def pulses_at(base: float, reach: float, z: float):
    """Base `base`, the shoulder and elbow that put the cup `reach` out by the firmware at true height z."""
    import math
    bearing = math.radians((500.0 - base) * landing.BASE_PULSE_DEG)
    x, y = reach * math.sin(bearing), -reach * math.cos(bearing)
    raw_z = raw.raw_position((x, y, z))[2]
    return geometry.deg_to_pulse(geometry.inverse((x, y, raw_z)))


def go(arm: MaxArm, pulses, speed: float) -> None:
    failed = arm._drive(landing._point(pulses), speed, None)
    if failed is not None:
        raise RuntimeError(f"{failed.status.value}: {failed.detail}")


def steady(arm: MaxArm):
    reads = arm._steady()
    if reads is None:
        raise RuntimeError("no steady read")
    return reads


def trial(arm: MaxArm, base: int, reach: float, side: int, swing: int):
    low = pulses_at(base, reach, Z)
    high = pulses_at(base, reach, Z + TRAVEL_UP_MM)
    extra2, extra3 = landing.extra_pulses(reach)
    go(arm, (base + side * swing, high[1], high[2]), MOVE_FAST_MM_S)
    go(arm, (base, high[1], high[2]), MOVE_FAST_MM_S)
    go(arm, (base, low[1] + extra2, low[2] + extra3), MOVE_SLOW_MM_S)
    settled = steady(arm)
    go(arm, (base, high[1], high[2]), MOVE_SLOW_MM_S)
    return settled


def log(line: str) -> None:
    print(line, flush=True)
    with OUT.open("a") as out:
        out.write(line + "\n")


arm = MaxArm()
arm.use_calibration(None)
arm.use_coordinates(Space.ARM)
arm.connect()
OUT.parent.mkdir(exist_ok=True)
which = sys.argv[1] if len(sys.argv) > 1 else "sweep"
try:
    first = pulses_at(BASES[0], REACHES[0], Z + TRAVEL_UP_MM)
    go(arm, first, MOVE_SLOW_MM_S)
    if which == "sweep":
        log("time\tbase\treach\tside\tswing\tsettled_base\tshortfall\tsettled_2\tsettled_3")
        for base in BASES:
            for reach in REACHES:
                for swing in SWINGS:
                    for side in SIDES:
                        for _ in range(REPEATS):
                            s = trial(arm, base, reach, side, swing)
                            log(f"{time.strftime('%H:%M:%S')}\t{base}\t{reach:.0f}\t{side:+d}\t{swing}\t{s[0]:.2f}\t"
                                f"{s[0] - base:+.2f}\t{s[1]:.2f}\t{s[2]:.2f}")
    elif which == "nudge":
        log("time\treach\tcommanded_base\tsettled_base\toff")
        for reach in REACHES:
            low = pulses_at(110, reach, Z)
            go(arm, (110, low[1], low[2]), MOVE_SLOW_MM_S)
            steady(arm)
            for base in list(range(111, 123)) + list(range(121, 109, -1)):
                go(arm, (base, low[1], low[2]), MOVE_SLOW_MM_S)
                s = steady(arm)
                log(f"{time.strftime('%H:%M:%S')}\t{reach:.0f}\t{base}\t{s[0]:.2f}\t{s[0] - base:+.2f}")
            go(arm, pulses_at(110, reach, Z + TRAVEL_UP_MM), MOVE_SLOW_MM_S)
finally:
    arm.disconnect()
