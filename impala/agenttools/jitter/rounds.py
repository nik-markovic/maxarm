#!/usr/bin/env python3
"""quadrant.json's nudge rounds: before each push, where each servo was (the mean of the second of reads before it)
against the target's exact pulses and against the whole pulses the firmware sends (truncated); which servos were
pushed; and whether the round brought them nearer. No arm.

    .venv/bin/python impala/agenttools/jitter/rounds.py [quadrant.json]
"""

import json
import math
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
    pushed_rows, only_truncation, helped, stuck = [], 0, 0, 0
    per_servo = {s: {"moved": 0, "stuck": 0} for s in range(3)}
    for move in data["moves"]:
        target = np.array(arm._spaces.arm.from_desk(arm._space.to_desk(tuple(move["target"]))))
        want = np.array(arm_module._pulses_of(tuple(target)))
        sent = np.floor(want)
        reads = np.array([r for r in move["reads"] if r[1] is not None], float)
        cmds = move["commands"]
        # nudge pairs: a command to somewhere else than the target, then one to the target, 0.4-0.6 s apart
        pairs = [(a, b) for a, b in zip(cmds, cmds[1:])
                 if np.linalg.norm(np.array(a[2]) - target) > 0.05 and np.linalg.norm(np.array(b[2]) - target) < 0.05
                 and 0.3 < b[0] - a[0] < 0.8]
        ends = [move["returned"]]
        for push, back in pairs:
            before = reads[(reads[:, 0] > push[0] - 1.0) & (reads[:, 0] < push[0]), 1:].mean(axis=0)
            nxt = [p for p, _ in pairs if p[0] > back[0]]
            until = nxt[0][0] if nxt else move["returned"]
            after = reads[(reads[:, 0] > until - 1.0) & (reads[:, 0] < until), 1:].mean(axis=0)
            push_pulses = np.array(arm_module._pulses_of(tuple(push[2])))
            pushed = np.abs(push_pulses - want) > 0.5
            miss_exact, miss_sent = np.abs(want - before), np.abs(sent - before)
            if not np.any(miss_sent[pushed] > 0.75):
                only_truncation += 1
            for s in np.nonzero(pushed)[0]:
                key = "moved" if abs(after[s] - before[s]) >= 0.5 else "stuck"
                per_servo[s][key] += 1
            better = np.abs(want - after)[pushed].max() < miss_exact[pushed].max() - 0.25
            helped += int(better)
            stuck += int(not better)
            pushed_rows.append((move["bearing"], move["reach"], move["label"], np.round(before - want, 2),
                                pushed.astype(int), np.round(after - want, 2)))
    print(f"{len(pushed_rows)} nudge rounds: {helped} brought the pushed servos nearer, {stuck} did not; "
          f"{only_truncation} were for servos within 0.75 of the whole pulse the firmware sends")
    for s in range(3):
        print(f"  servo {s + 1} pushed: moved {per_servo[s]['moved']}, did not {per_servo[s]['stuck']}")
    for row in pushed_rows[:40]:
        print(f"  {row[0]:+3d} {row[1]} {row[2]:8s} before {row[3].tolist()} pushed {row[4].tolist()} after {row[5].tolist()}")


if __name__ == "__main__":
    main()
