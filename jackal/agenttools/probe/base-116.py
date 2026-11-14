#!/usr/bin/env python3
"""Can the base be landed on pulse 116? The library's landing (`MaxArm.down_to()`) aimed at base pulse 116, again
and again, at several reaches, and with its pattern varied: how far aside the last turn starts (APPROACH_PULSES)
and how far past it aims (AIM_PAST_FALLING). Arm space, true, no calibration, as the desk page. MOVES THE ARM: the
owner runs it, or releases the arm for it.

After each landing: the base's steady mean (what the library takes as where it is) and 20 single reads of it.

    PYTHONPATH=src ../.venv/bin/python3 agenttools/probe/base-116.py repeat      # 12 landings, 206 mm out
    PYTHONPATH=src ../.venv/bin/python3 agenttools/probe/base-116.py reaches     # 160 180 230 250 mm, 3 each
    PYTHONPATH=src ../.venv/bin/python3 agenttools/probe/base-116.py pattern     # aside 20/60 pulses, aim past 0/1
"""
import collections
import math
import sys

from maxarm import MaxArm, Space, geometry, landing, raw

BASE = 116.0
Z = 86.0


def target_at(reach_raw: float) -> tuple:
    """The true point whose firmware base pulse is BASE, `reach_raw` out by the firmware, at true height Z."""
    bearing = math.radians((500.0 - BASE) * landing.BASE_PULSE_DEG)
    x, y = reach_raw * math.sin(bearing), -reach_raw * math.cos(bearing)
    true = raw.true_position((x, y, raw.raw_position((x, y, Z))[2]))
    return (true[0], true[1], Z)


def land(arm: MaxArm, point: tuple, label: str) -> float:
    aimed = geometry.deg_to_pulse(geometry.inverse(raw.raw_position(point)))[0]
    tried = []                                  # each way down the library makes: the base it aims at
    original = arm._drive
    def drive(target, speed, servo):
        tried.append(round(geometry.deg_to_pulse(geometry.inverse(raw.raw_position(target)))[0], 2))
        return original(target, speed, servo)
    arm._drive = drive
    try:
        result = arm.down_to(*point)
    finally:
        arm._drive = original
    settled = arm._settled[0] if arm._settled else float("nan")
    reads = collections.Counter(pulses[0] for pulses in (arm._protocol.read_pulses() for _ in range(20)) if pulses)
    print(f"{label:28} aimed {aimed:7.2f}  settled {settled:7.2f}  reads {dict(sorted(reads.items()))}  "
          f"base commanded along the way {tried}", flush=True)
    return settled


arm = MaxArm()
arm.use_calibration(None)
arm.use_coordinates(Space.ARM)
arm.connect()
which = sys.argv[1] if len(sys.argv) > 1 else "repeat"
try:
    if which == "repeat":
        point = target_at(204.0)
        for n in range(12):
            land(arm, point, f"206 mm, landing {n + 1}")
    elif which == "reaches":
        for reach in (160.0, 180.0, 230.0, 250.0):
            for n in range(3):
                land(arm, target_at(reach - 2.0), f"{reach:.0f} mm, landing {n + 1}")
    elif which == "pattern":
        point = target_at(204.0)
        for approach, aim in ((20, 0), (60, 0), (60, 1)):
            landing.APPROACH_PULSES, landing.AIM_PAST_FALLING, landing.AIM_PAST_RISING = approach, aim, aim
            for n in range(4):
                land(arm, point, f"aside {approach}, past {aim}, #{n + 1}")
finally:
    arm.disconnect()
