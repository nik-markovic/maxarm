#!/usr/bin/env python3
"""Ways to end a precise move's wait and say where the servos are, tried on recorded reads (walk.json, and
arm-to-desk/response.json's streams that start as a travel ends): when each ends (s after the travel) and how far its
answer is from the level the servos settle at (the mean of the last 2.5 s / of 2.5 s on), servos 2 and 3. No arm.

    .venv/bin/python impala/agenttools/jitter/rules.py
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAP_S = 3.0


def streams():
    for move in json.loads((HERE / "walk.json").read_text())["moves"]:
        last = move["commands"][-1]
        end = last[0] + last[1] / 1000
        reads = np.array([r for r in move["reads"] if r[1] is not None and r[0] >= end - 0.05], float)
        t, v = reads[:, 0] - end, reads[:, 1:]
        yield "walk", t, v, v[t >= t[-1] - 2.5].mean(axis=0)
    data = json.loads((HERE.parent / "arm-to-desk" / "response.json").read_text())
    for run in data["runs"]:
        for read in run["reads"]:
            before = [c for c in run["commands"] if c["at"] < read["called"]]
            end = before[-1]["at"] + before[-1]["ms"] / 1000
            rows = np.array([r for r in read["reads"] if r[1] is not None], float)
            t, v = rows[:, 0] - end, rows[:, 1:]
            if t[0] < 0.3 and t[-1] > 3.5:
                yield "response", t, v, v[t >= 2.5].mean(axis=0)


def still(t, v, still_s, answer):
    """Today's rule: every read of the last still_s within a pulse of the others; `answer` 'last' or 'mean'."""
    for i in range(len(t)):
        window = (t >= t[i] - still_s) & (t <= t[i])
        done = t[i] - t[0] >= still_s and np.all(v[window].max(0) - v[window].min(0) <= 1)
        if done or t[i] >= CAP_S:
            return t[i], (v[i] if answer == "last" else v[window].mean(axis=0))
    return t[-1], v[-1]


def steady(t, v, half_s, within, start_s):
    """The mean of the last half_s against the mean of the half_s before it: done when every servo's two means are
    within `within` pulses; the answer the mean of both halves. Not before start_s after the travel."""
    for i in range(len(t)):
        if t[i] < start_s:
            continue
        recent = (t > t[i] - half_s) & (t <= t[i])
        before = (t > t[i] - 2 * half_s) & (t <= t[i] - half_s)
        if before.sum() >= 3 and np.all(np.abs(v[recent].mean(0) - v[before].mean(0)) <= within):
            return t[i], v[recent | before].mean(axis=0)
        if t[i] >= CAP_S:
            return t[i], v[(t > t[i] - 2 * half_s) & (t <= t[i])].mean(axis=0)
    return t[-1], v[-1]


def main():
    data = list(streams())
    rules = {
        "today: still 1.25 s, the last read": lambda t, v: still(t, v, 1.25, "last"),
        "still 1.25 s, the mean of those reads": lambda t, v: still(t, v, 1.25, "mean"),
        "still 0.5 s, the mean of those reads": lambda t, v: still(t, v, 0.5, "mean"),
        "steady: two 0.5 s means within 0.5, from 1 s": lambda t, v: steady(t, v, 0.5, 0.5, 1.0),
        "steady: two 0.5 s means within 0.35, from 1 s": lambda t, v: steady(t, v, 0.5, 0.35, 1.0),
        "fixed: the mean of the reads 1.0-2.0 s": lambda t, v: (2.0, v[(t >= 1.0) & (t <= 2.0)].mean(axis=0)),
    }
    for source in ("walk", "response"):
        rows = [(t, v, level) for name, t, v, level in data if name == source]
        print(f"{source} ({len(rows)} streams): ends at s after the travel (median / 90% / at the {CAP_S:.0f} s cap); "
              f"servos 2-3 against the level: rms / 95% / worst pulses, share a pulse or more off")
        for name, rule in rules.items():
            ends, errors = [], []
            for t, v, level in rows:
                when, answer = rule(t, v)
                ends.append(when)
                errors.append(np.abs(np.asarray(answer) - level)[1:])
            ends, e = np.array(ends), np.array(errors)
            print(f"  {name:47s} {np.median(ends):.2f} / {np.percentile(ends, 90):.2f} / {np.mean(ends >= CAP_S - 0.05) * 100:3.0f}%"
                  f"   {np.sqrt(np.mean(e ** 2)):.2f} / {np.percentile(e, 95):.2f} / {e.max():.2f}  {np.mean(e >= 0.999) * 100:4.1f}%")


if __name__ == "__main__":
    main()
