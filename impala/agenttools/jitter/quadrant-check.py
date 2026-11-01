#!/usr/bin/env python3
"""quadrant.json: for each precise move, where the arm ended against its target, what the library reported, and how
long it took; by kind of move and by place. No arm.

    .venv/bin/python impala/agenttools/jitter/quadrant-check.py [quadrant.json]

Ended: the mean of the reads from 1 s after the move returned to the end of the watch (a late creep counts).
Distances at the cup, mm, from the target's exact pulses through the library's own true-position formulas.
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
    rows = []
    for move in data["moves"]:
        target = np.array(arm._spaces.arm.from_desk(arm._space.to_desk(tuple(move["target"]))))
        want = np.array(arm_module._pulses_of(tuple(target)))
        later = np.array([r for r in move["reads"] if r[1] is not None and r[0] >= move["returned"] + 1.0], float)
        level = later[:, 1:].mean(axis=0)
        ended = np.linalg.norm(np.array(arm_module._position_of(tuple(level))) - target)
        said = move["settled"]
        told = np.linalg.norm(np.array(arm_module._position_of(tuple(said))) - target) if said else np.nan
        told_vs = (np.linalg.norm(np.array(arm_module._position_of(tuple(said))) -
                                  np.array(arm_module._position_of(tuple(level)))) if said else np.nan)
        moves_sent = len(move["commands"])
        rows.append((move["bearing"], move["reach"], move["label"], move["status"], move["returned"] - move["started"],
                     moves_sent, ended, told, told_vs, np.abs(level - want)))
    print(f"{len(rows)} moves; ended (mm from the target, after) / reported (mm from the target) / reported against "
          f"where it ended (mm); time s; commands")
    for label in dict.fromkeys(r[2] for r in rows):
        sel = [r for r in rows if r[2] == label]
        e, t, v, s, c = (np.array([r[i] for r in sel]) for i in (6, 7, 8, 4, 5))
        print(f"  {label:8s} ({len(sel)}): ended {np.median(e):.2f} median, {e.max():.2f} worst; reported "
              f"{np.nanmedian(t):.2f}; reported vs ended {np.nanmedian(v):.2f} ({np.nanmax(v):.2f}); "
              f"{np.median(s):.1f} s ({s.max():.1f}); {np.median(c):.0f} commands ({c.max()})")
    e = np.array([r[6] for r in rows])
    v = np.array([r[8] for r in rows])
    s = np.array([r[4] for r in rows])
    print(f"  all: ended {np.median(e):.2f} median, {np.percentile(e, 90):.2f} at 90%, {e.max():.2f} worst; reported vs "
          f"ended {np.nanmedian(v):.2f} median, {np.nanmax(v):.2f} worst; {np.median(s):.1f} s median, {s.max():.1f} worst")
    print("by place: ended median / worst mm, time median s")
    for place in dict.fromkeys((r[0], r[1]) for r in rows):
        sel = [r for r in rows if (r[0], r[1]) == place]
        e = np.array([r[6] for r in sel]); s = np.array([r[4] for r in sel])
        servo = np.array([r[9] for r in sel])
        print(f"  {place[0]:+3d} deg {place[1]}: {np.median(e):.2f} / {e.max():.2f} mm, {np.median(s):.1f} s; servo |pulses| "
              f"median {np.round(np.median(servo, axis=0), 2).tolist()}, worst {np.round(servo.max(axis=0), 2).tolist()}")
    bad = [r for r in rows if r[3] != "done"]
    if bad:
        print("not done:", [(r[0], r[1], r[2], r[3]) for r in bad])


if __name__ == "__main__":
    main()
