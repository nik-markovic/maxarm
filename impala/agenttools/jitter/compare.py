#!/usr/bin/env python3
"""The four walks side by side: today's precise move (walk.json) and the nudge three ways (nudge-walk*.json). Where
each move ended against the target's exact pulses -- the mean of the reads 1.5-4 s after it was done, so a late
creep counts -- in mm at the cup and per servo; how long each move took; nudges that moved the servo past its target.

    .venv/bin/python impala/agenttools/jitter/compare.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IMPALA = HERE.parents[1]
sys.path[:0] = [str(IMPALA), str(IMPALA / "calibration")]

from maxarm import MaxArm, Space, geometry, raw                            # noqa: E402
from common.motion import Motion                                           # noqa: E402


def pulses_of(true):
    return np.array(geometry.deg_to_pulse(geometry.inverse(raw.raw_position(tuple(true)))))


def true_of(pulses):
    return np.array(raw.true_position(geometry.forward(geometry.pulse_to_deg(tuple(pulses)))))


def main():
    arm = MaxArm(config_dir=None)
    Motion(arm, IMPALA / "config").use(Space.DESK)
    runs = {"today's precise move": "walk.json", "nudge A: 1 mm past": "nudge-walk.json",
            "nudge B: + servo 2 3.5 mm": "nudge-walk-b.json", "nudge C: + base 2.5 mm": "nudge-walk-c.json"}
    print("ended, mm from the target (median / 90% / worst); per servo |pulses| median (worst); move s (median / "
          "worst); servos ending a pulse or more past after a nudge")
    for name, file in runs.items():
        moves = json.loads((HERE / file).read_text())["moves"]
        mm, servo, took, past = [], [], [], 0
        for index, move in enumerate(moves):
            if index == 0:
                continue                                     # the first move comes from the park
            if "want" in move:
                target, want = np.array(move["target"]), np.array(move["want"])
                done, started = move["done"], move["started"]
                for before, after in zip(move["rounds"], move["rounds"][1:]):
                    miss_before = want - np.array(before["here"])
                    miss_after = want - np.array(after["here"])
                    past += int(np.sum((np.sign(miss_before) != np.sign(miss_after)) & (np.abs(miss_after) >= 1.0)
                                       & (np.abs(miss_before) > 0.75)))
            else:
                target = np.array(arm._spaces.arm.from_desk(arm._space.to_desk(tuple(move["target"]))))
                want = pulses_of(target)
                done, started = move["returned"], move["started"]
            reads = np.array([r for r in move["reads"] if r[1] is not None and done + 1.5 <= r[0] <= done + 4.0], float)
            level = reads[:, 1:].mean(axis=0)
            mm.append(np.linalg.norm(true_of(level) - target))
            servo.append(np.abs(level - want))
            took.append(done - started)
        mm, servo, took = np.array(mm), np.array(servo), np.array(took)
        print(f"  {name:27s} {np.median(mm):.2f} / {np.percentile(mm, 90):.2f} / {mm.max():.2f} mm; "
              + ", ".join(f"{np.median(servo[:, s]):.2f} ({servo[:, s].max():.2f})" for s in range(3))
              + f"; {np.median(took):.1f} / {took.max():.1f} s; past {past}")


if __name__ == "__main__":
    main()
