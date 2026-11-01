#!/usr/bin/env python3
"""The library's new precise move on the arm: how long each round waited for the servos to keep still, the read the
move ended on, and whether the servos then stay within a pulse of it for 4 s more. Near the base, and two far.

    .venv/bin/python impala/agenttools/arm-to-desk/settle-new.py      # MOVES THE ARM: the owner's say-so only
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IMPALA = HERE.parents[1]
sys.path[:0] = [str(IMPALA), str(IMPALA / "calibration")]

from maxarm import MaxArm, Space                                           # noqa: E402
from common.motion import OVER_CUBES_MM, Motion                            # noqa: E402

POINTS = [(0.0, -105.0), (-75.0, -75.0), (75.0, -75.0), (-100.0, -35.0), (100.0, -35.0), (0.0, -230.0),
          (-140.0, -220.0)]


def main():
    arm = MaxArm(config_dir=None)
    motion = Motion(arm, IMPALA / "config")
    settles, original = [], arm._settle

    def settle(reads, still_s):
        result = original(reads, still_s)
        settles.append({"still_s": still_s, "after_travel": time.monotonic() - arm._travel_ended,
                        "reads": len(reads), "pulses": list(result) if result else None})
        return result

    arm._settle = settle
    rows = []
    try:
        motion.connect()
        for x, y in POINTS:
            target = (x, y, 42.0)
            motion.use(Space.DESK)
            if not arm.can_reach(*target):
                print(f"{target}: out of reach, skipped", flush=True)
                continue
            first = len(settles)
            started = time.monotonic()
            result = motion.approach(Space.DESK, target, precise=True, hover=OVER_CUBES_MM)
            took = time.monotonic() - started
            ended = list(arm._settled) if arm._settled else None
            after = []
            while time.monotonic() - started - took < 4.0:
                pulses = arm._protocol.read_pulses()
                if pulses:
                    after.append(list(pulses))
            after = np.array(after)
            off = np.abs(after - ended).max(axis=0).tolist() if ended else None
            level = np.median(after[-15:], axis=0).tolist()
            rows.append({"target": target, "status": result.status.value, "took": took, "ended": ended,
                         "settles": settles[first:], "after_max_off": off, "later_level": level})
            rounds = "; ".join(f"{s['still_s']} s still: {s['after_travel']:.2f} s after the travel, {s['reads']} reads"
                               for s in settles[first:])
            print(f"{target}: {result.status.value} in {took:.1f} s | {rounds} | ended {ended}; the next 4 s up to "
                  f"{off} off it, level then {level}", flush=True)
    finally:
        try:
            motion.reset()
        finally:
            arm.disconnect()
            (HERE / "settle-new.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
