#!/usr/bin/env python3
"""How well each way of turning the servos' reads into one position finds where a servo settles, on recorded read
streams (arm-to-desk/response.json: 20 precise moves to Fine Tune's points, every read for 4 s after each command).

    .venv/bin/python impala/agenttools/jitter/estimators.py [streams.json ...]

Times from the end of the travel the last command gave the servos. A servo's level: the mean of its reads from 2.5 s
on. Each way's answer, servo by servo, against that level, in pulses (0.55 mm for servos 2 and 3 at the cup, 0.6-1.1
for the base).
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LEVEL_FROM_S = 2.5


def streams_from_response(path):
    data = json.loads(path.read_text())
    for run in data["runs"]:
        for read in run["reads"]:
            before = [c for c in run["commands"] if c["at"] < read["called"]]
            end = before[-1]["at"] + before[-1]["ms"] / 1000
            rows = np.array([r for r in read["reads"] if r[1] is not None], float)
            yield rows[:, 0] - end, rows[:, 1:]


def settle(times, values, still_s, cap_s=3.0, first=0.0):
    """The library's rule (maxarm/arm.py _settle): reads until every read of the last `still_s` is within a pulse of
    the others, servo by servo; the index of the read it ends on; or the cap."""
    for i in range(len(times)):
        if times[i] < first:
            continue
        recent = (times >= times[i] - still_s) & (times <= times[i]) & (times >= first)
        if times[i] - max(times[0], first) >= still_s and np.all(values[recent].max(0) - values[recent].min(0) <= 1):
            return i
        if times[i] >= cap_s:
            return i
    return len(times) - 1


def main():
    paths = [Path(p) for p in sys.argv[1:]] or [HERE.parent / "arm-to-desk" / "response.json"]
    streams = [s for p in paths for s in streams_from_response(p)]
    early = [(t, v) for t, v in streams if t[0] < 0.3 and t[-1] > 3.5]     # streams that start as the travel ends
    late = [(t, v) for t, v in streams if t[0] > 3.0]                       # streams long after
    print(f"{len(early)} streams from the end of a travel, {len(late)} long after it")
    # at rest: how the reads flicker
    flick, changes = [], []
    for t, v in late:
        mode = np.array([np.bincount(v[:, s].astype(int) - int(v[:, s].min())).argmax() + v[:, s].min() for s in range(3)])
        flick.append(np.mean(v != mode, axis=0))
        changes.append(np.abs(v - mode).max(axis=0))
    flick, changes = np.array(flick), np.array(changes)
    print("at rest (4 s, well after the move): share of reads off the commonest value, by servo: "
          + ", ".join(f"{100 * np.mean(flick[:, s]):.0f}%" for s in range(3))
          + "; streams where a servo flickers at all: " + ", ".join(f"{np.mean(flick[:, s] > 0) * 100:.0f}%" for s in range(3))
          + "; furthest read from it: " + ", ".join(f"{int(changes[:, s].max())}" for s in range(3)))
    ways = {}
    for t, v in early:
        level = v[t >= LEVEL_FROM_S].mean(axis=0)
        def add(name, value):
            ways.setdefault(name, []).append(value - level)
        for call in (0.4, 0.7, 1.0, 1.5, 2.0):
            at = np.searchsorted(t, call)
            add(f"one read at {call:.1f} s", v[at])
            add(f"median of 5 from {call:.1f} s", np.median(v[at:at + 5], axis=0))
            add(f"mean of 5 from {call:.1f} s", v[at:at + 5].mean(axis=0))
            window = (t >= call - 0.5) & (t <= call)
            if window.sum() >= 3:
                add(f"mean of the 0.5 s up to {call:.1f} s", v[window].mean(axis=0))
        for still in (0.5, 1.25):
            i = settle(t, v, still)
            add(f"library: still {still} s, last read", v[i])
            recent = (t >= t[i] - still) & (t <= t[i])
            add(f"library: still {still} s, mean of the still reads", v[recent].mean(axis=0))
            add(f"library: still {still} s, mean of the last 5", v[max(0, i - 4):i + 1].mean(axis=0))
            ways.setdefault(f"  (library still {still} s ends at, s)", []).append(np.full(3, t[i]))
    print(f"\nagainst the level each servo settles at (mean from {LEVEL_FROM_S} s), pulses: rms / 95% / worst, "
          f"and the share 1 pulse or more off ({len(early)} streams x 3 servos)")
    for name, errors in ways.items():
        e = np.abs(np.array(errors))
        if name.startswith("  "):
            print(f"{name:44s} median {np.median(e[:, 0]):.2f}, 90% {np.percentile(e[:, 0], 90):.2f}")
            continue
        print(f"{name:44s} {np.sqrt(np.mean(e ** 2)):.2f} / {np.percentile(e, 95):.2f} / {e.max():.2f}   "
              f"{np.mean(e >= 0.999) * 100:4.1f}%")


if __name__ == "__main__":
    main()
