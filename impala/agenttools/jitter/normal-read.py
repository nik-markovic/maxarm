#!/usr/bin/env python3
"""get_position() after plain (slow and fast) moves, as the app reads after every command: straight after the move,
and again 2.5 s later (at rest); each against the mean of the reads over the next 3 s. Forward quadrant, above the
cubes' height. No precise move.

    .venv/bin/python impala/agenttools/jitter/normal-read.py      # MOVES THE ARM: the owner's say-so (30 Oct)
"""

import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMPALA = HERE.parents[1]
sys.path[:0] = [str(IMPALA), str(IMPALA / "calibration")]

from maxarm import MaxArm, MoveType, Space, arm as arm_module             # noqa: E402
from common.motion import Motion                                           # noqa: E402

POINTS = [(0, 170, 55), (20, 180, 60), (-20, 190, 55), (35, 160, 60), (-35, 170, 55), (0, 230, 60), (10, 150, 50),
          (-10, 210, 60)]


def main():
    arm = MaxArm(config_dir=None)
    motion = Motion(arm, IMPALA / "config")
    rows = []
    try:
        print("connecting: the board resets and homes the arm", flush=True)
        motion.connect()
        motion.use(Space.DESK)
        for index, (bearing, reach, z) in enumerate(POINTS):
            b = math.radians(bearing)
            kind = MoveType.SLOW if index % 2 else MoveType.FAST
            result = arm.move(reach * math.sin(b), -reach * math.cos(b), z, kind)
            moved = time.monotonic()
            first = arm._read_steady()
            took = time.monotonic() - moved
            time.sleep(max(0.0, 2.5 - (time.monotonic() - moved)))
            started = time.monotonic()
            rest = arm._read_steady()
            rest_took = time.monotonic() - started
            later = []
            while time.monotonic() - started < rest_took + 3.0:
                pulses = arm._protocol.read_pulses()
                if pulses:
                    later.append(pulses)
            level = arm_module._mean(later)
            off = [math.dist(arm_module._position_of(tuple(read)), arm_module._position_of(level))
                   for read in (first, rest)]
            rows.append({"point": [bearing, reach, z], "kind": kind.value, "status": result.status.value,
                         "first": list(first), "first_s": took, "rest": list(rest), "rest_s": rest_took,
                         "level": list(level), "first_off_mm": off[0], "rest_off_mm": off[1]})
            print(f"{bearing:+3d} deg {reach} z {z} {kind.value}: read after the move in {took:.2f} s, {off[0]:.2f} mm "
                  f"from the level; at rest in {rest_took:.2f} s, {off[1]:.2f} mm", flush=True)
            (HERE / "normal-read.json").write_text(json.dumps(rows))
    finally:
        try:
            motion.reset()
        finally:
            arm.disconnect()


if __name__ == "__main__":
    main()
