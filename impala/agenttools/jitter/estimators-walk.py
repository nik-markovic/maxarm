#!/usr/bin/env python3
"""walk.json: how the servos' reads behave after each precise move's last command, and how well each way of turning
them into one position finds where the servos settle. No arm.

    .venv/bin/python impala/agenttools/jitter/estimators-walk.py

Times from the end of the last command's travel. A servo's level: the mean of its reads over the watch's last 2.5 s
(the move's return plus 1.5-4 s). Errors in pulses (0.55 mm at the cup for servos 2 and 3).
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / "walk.json").read_text())
    rows, curve, flick = [], {}, []
    ways = {}
    for move in data["moves"]:
        last = move["commands"][-1]
        end = last[0] + last[1] / 1000
        reads = np.array([r for r in move["reads"] if r[1] is not None and r[0] >= end - 0.05], float)
        t, v = reads[:, 0] - end, reads[:, 1:]
        returned = move["returned"] - end
        tail = t >= t[-1] - 2.5
        level = v[tail].mean(axis=0)
        settled = np.array(move["settled"], float)
        still_s = move["settles"][-1][1] if move["settles"] else 1.25
        at_end = np.searchsorted(t, returned) - 1                 # the read the library ended on
        window = (t >= t[at_end] - still_s) & (t <= t[at_end])

        def add(name, value):
            ways.setdefault(name, []).append(np.asarray(value, float) - level)

        add("library today: the last read", settled)
        add("library's still reads, their mean", v[window].mean(axis=0))
        add("library's end, mean of the last 5", v[max(0, at_end - 4):at_end + 1].mean(axis=0))
        add("library's end, median of the last 5", np.median(v[max(0, at_end - 4):at_end + 1], axis=0))
        for call in (0.4, 1.0, 1.5):
            i = np.searchsorted(t, call)
            add(f"median of 5 from {call:.1f} s (noon at 0.4)", np.median(v[i:i + 5], axis=0))
            add(f"mean of 5 from {call:.1f} s", v[i:i + 5].mean(axis=0))
        for a, b in ((0.5, 1.5), (1.0, 2.0), (1.5, 2.5)):
            add(f"mean of the reads {a}-{b} s", v[(t >= a) & (t <= b)].mean(axis=0))
        i = np.searchsorted(t, returned + 0.4)
        add("noon's app read: 0.4 s after return, median of 5", np.median(v[i:i + 5], axis=0))
        # a single read's chance to be a pulse or more off the level, by time after the travel
        for lo in np.arange(0.0, 6.0, 0.5):
            sel = (t >= lo) & (t < lo + 0.5)
            if sel.any():
                curve.setdefault(lo, []).extend(list(np.abs(v[sel] - level).max(axis=1) >= 1 - 1e-9))
        mode = np.array([np.bincount(v[tail, s].astype(int) - int(v[tail, s].min())).argmax() + v[tail, s].min()
                         for s in range(3)])
        flick.append(np.mean(v[tail] != mode, axis=0))
        late = np.abs(v[t > returned] - settled).max(axis=0)
        rows.append((move["reach"], move["z"], returned, still_s, settled, level, late))
    print("each move: when the library returned (s after the last travel); what it said; the level after; the "
          "furthest read after it returned, pulses (servos 2, 3)")
    for reach, z, returned, still_s, settled, level, late in rows:
        flag = "  (3 s limit)" if returned >= 2.95 else ""
        print(f"  reach {reach} z {z}: returned {returned:4.2f}{flag:13s} said {settled[1:].astype(int)} level "
              f"{np.round(level[1:], 2)} off by {np.round(settled[1:] - level[1:], 2)}; reads after up to "
              f"{late[1:].astype(int)} from it")
    flick = np.array(flick)
    print(f"\nat rest (the last 2.5 s): reads off the commonest value, servos 1-3: "
          + ", ".join(f"{100 * flick[:, s].mean():.0f}%" for s in range(3)))
    print("a single read a pulse or more off the level (any servo), by time after the travel:")
    print("  " + "  ".join(f"{lo:.1f}-{lo + 0.5:.1f} s {100 * np.mean(x):3.0f}%" for lo, x in sorted(curve.items())))
    print(f"\nagainst the level, servos 2 and 3 ({len(rows)} moves): rms / 95% / worst pulses, share a pulse or more off")
    for name, errors in ways.items():
        e = np.abs(np.array(errors))[:, 1:]
        print(f"  {name:48s} {np.sqrt(np.mean(e ** 2)):.2f} / {np.percentile(e, 95):.2f} / {e.max():.2f}   "
              f"{100 * np.mean(e >= 0.999):4.1f}%")


if __name__ == "__main__":
    main()
