#!/usr/bin/env python3
"""What each precise move of settle.py did, in servo pulses: the target's pulses (as `_correct()` wants them), each
round's 5 reads and the correction it sent, the app's read after, and where the servos sat once still (the reads of
the last 3 s: each pulse value and how often).

    .venv/bin/python impala/agenttools/arm-to-desk/precise-rounds.py settle.json:trace.txt settle-more.json:trace-more.txt
"""

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1])]

from maxarm import geometry, raw                                           # noqa: E402

TO = re.compile(r"^to \(([-\d.]+), ([-\d.]+), ([-\d.]+) \[raw")
SENT = re.compile(r"arm\.set_position\(\(([-\d.]+), ([-\d.]+), ([-\d.]+)\), (\d+)\)")
READ = re.compile(r"^(?:#R# )?\[(\d+), (\d+), (\d+)\]")


def pulses_of(true):
    return [int(p) for p in geometry.deg_to_pulse(geometry.inverse(raw.raw_position(true)))]


def main():
    for pair in sys.argv[1:]:
        data_file, trace_file = pair.split(":")
        data = json.loads((HERE / data_file).read_text())
        start = data["trace_start"]
        entries = []
        for line in (HERE / trace_file).read_text().splitlines():
            m = re.match(r"\s+([\d.]+)s (.) (.*)", line)
            if m:
                entries.append((float(m.group(1)), m.group(3)))
        for run in data["runs"]:
            t0, t1, t2 = run["started"] - start, run["done"] - start, run["read"] - start
            legs, reads, last_to = [], [], None
            for t, text in entries:
                if not t0 <= t <= t2:
                    continue
                if (m := TO.search(text)):
                    last_to = tuple(float(v) for v in m.groups())
                elif (m := SENT.search(text)):
                    legs.append((t, last_to, int(m.group(4))))
                elif (m := READ.search(text)):
                    reads.append((t, [int(v) for v in m.groups()]))
            descent = [leg for leg in legs if leg[2] > 300][-1]
            wanted = pulses_of(descent[1])
            fixes = [leg for leg in legs if leg[0] > descent[0]]
            line = f"{str(run['target'][:2]):16s} wanted {wanted}"
            for t, aim, ms in [(None, None, None)] + fixes:
                group = [p for when, p in reads if (t is None or when > t) and when < t1 and
                         (not fixes or t is not None or when < fixes[0][0])]
                if t is not None:
                    later = [f for f in fixes if f[0] > t]
                    group = [p for when, p in reads if t < when < (later[0][0] if later else t1 + 0.01)]
                    line += f" | sent aim {pulses_of(aim)}"
                if group:
                    median = np.median(np.array(group), axis=0).astype(int).tolist()
                    line += f" -> read {median}"
            app = [p for when, p in reads if t1 < when <= t2 + 0.01]
            polls = np.array([p[1:] for p in run["polls"] if p[1] is not None and p[0] >= run["final"] - 3.0], int)
            final = [dict(sorted(collections.Counter(polls[:, i]).items())) for i in range(3)]
            line += f" | app's read {np.median(np.array(app), axis=0).astype(int).tolist() if app else '-'}"
            line += f" | last 3 s: {final}"
            print(line)


if __name__ == "__main__":
    main()
