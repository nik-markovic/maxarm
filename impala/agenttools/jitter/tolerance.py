#!/usr/bin/env python3
"""quadrant.json replayed with other tolerances: each move stops at the first steady state (before a push, or its
end) where every servo is within its tolerance of the target's exact pulses. Where it would have ended and how much
sooner. No arm.

    .venv/bin/python impala/agenttools/jitter/tolerance.py [quadrant.json]
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IMPALA = HERE.parents[1]
sys.path[:0] = [str(IMPALA), str(IMPALA / "calibration")]

from maxarm import MaxArm, Space, arm as arm_module                       # noqa: E402
from common.motion import Motion                                           # noqa: E402


def main():
    data = json.loads((HERE / (sys.argv[1] if len(sys.argv) > 1 else "quadrant.json")).read_text())
    arm = MaxArm(config_dir=None)
    Motion(arm, IMPALA / "config").use(Space.DESK)
    moves = []
    for move in data["moves"]:
        target = np.array(arm._spaces.arm.from_desk(arm._space.to_desk(tuple(move["target"]))))
        want = np.array(arm_module._pulses_of(tuple(target)))
        reads = np.array([r for r in move["reads"] if r[1] is not None], float)
        cmds = move["commands"]
        pushes = [a for a, b in zip(cmds, cmds[1:])
                  if np.linalg.norm(np.array(a[2]) - target) > 0.05 and np.linalg.norm(np.array(b[2]) - target) < 1.0
                  and 0.3 < b[0] - a[0] < 0.8]
        states = []                                   # (time it was known, pulses)
        for push in pushes:
            states.append((push[0], reads[(reads[:, 0] > push[0] - 1.0) & (reads[:, 0] < push[0]), 1:].mean(axis=0)))
        end = move["returned"]
        later = reads[(reads[:, 0] >= end + 1.0), 1:].mean(axis=0)
        states.append((end, later))
        moves.append((move, target, want, states))
    for tolerance in ((0.75, 0.75, 0.75), (0.75, 0.75, 1.0), (0.75, 0.75, 1.25), (1.0, 1.0, 1.0), (0.75, 1.0, 1.25)):
        errors, saved = [], []
        for move, target, want, states in moves:
            for when, pulses in states:
                if np.all(np.abs(pulses - want) <= np.array(tolerance)) or when == states[-1][0]:
                    break
            errors.append(np.linalg.norm(np.array(arm_module._position_of(tuple(pulses))) - target))
            saved.append(states[-1][0] - when)
        errors, saved = np.array(errors), np.array(saved)
        took = np.array([m["returned"] - m["started"] for m, *_ in moves]) - saved
        print(f"tolerance {tolerance}: ended {np.median(errors):.2f} mm median, {np.percentile(errors, 90):.2f} at 90%, "
              f"{errors.max():.2f} worst; moves {np.median(took):.1f} s median ({np.median(saved):.1f} s sooner)")


if __name__ == "__main__":
    main()
