#!/usr/bin/env python3
"""response.py's log: how soon after a command the servos' reports are final, and whether waiting longer would have
changed what the precise move decided.

    .venv/bin/python impala/agenttools/arm-to-desk/response-check.py [response.json ...]

Times are from the end of the travel the last command gave the servos (its `ms`). A report's final value is the
last report of its read (the owner: trust the last report). For each read: when the reports last strayed 2 or more
pulses from it (`late`), and 1 or more (`exact`); what today's decision saw (the median of the first 5 reads from
400 ms) and the miss the loop would have had at the end instead.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOLERANCE = 1                                   # maxarm PRECISE_TOLERANCE_PULSES


def last_stray(times, values, final, by):
    """The last time a report was `by` or more pulses from `final`; None if never."""
    far = np.any(np.abs(values - final) >= by, axis=1)
    return float(times[far].max()) if far.any() else None


def main():
    files = sys.argv[1:] or ["response.json"]
    kinds = {"descent": [], "correction": [], "after": []}
    changed, outcomes, rows = [], [], []
    for file in files:
        data = json.loads((HERE / file).read_text())
        for run in data["runs"]:
            wanted = np.array(run["wanted"])
            commands = run["commands"]
            descent_at = max(c["at"] for c in commands if c["ms"] > 300)
            for index, read in enumerate(run["reads"]):
                before = [c for c in commands if c["at"] < read["called"]]
                command = before[-1]
                travel_end = command["at"] + command["ms"] / 1000
                rows_ = np.array([r for r in read["reads"] if r[1] is not None], float)
                times, values = rows_[:, 0] - travel_end, rows_[:, 1:]
                final = values[-1]
                kind = ("after" if index == len(run["reads"]) - 1 else
                        "descent" if command["at"] == descent_at else "correction")
                late, exact = last_stray(times, values, final, 2), last_stray(times, values, final, 1)
                start = times[0]
                kinds[kind].append((late if late is not None else start, exact if exact is not None else start,
                                    start))
                today = values[(rows_[:, 0] - read["called"]) >= 0.4][:5]
                decided = np.median(today, axis=0)
                miss_today = np.abs(wanted - decided).max()
                miss_final = np.abs(wanted - final).max()
                if kind != "after" and (miss_today <= TOLERANCE) != (miss_final <= TOLERANCE):
                    changed.append((run["target"][:2], kind, decided.tolist(), final.tolist(), wanted.tolist(), late))
                if kind == "after":
                    outcomes.append(np.abs(final - wanted))
                    rows.append((run["target"][:2], wanted.tolist(), final.tolist(), late, len(run["reads"]) - 1))
    print("from the end of the commanded travel, s: when the reports last strayed 2+ pulses / 1+ pulse from the last "
          "report of the read (the read started at)")
    for kind, values in kinds.items():
        if not values:
            continue
        late, exact, start = np.array(values).T
        print(f"  {kind:10s} {len(values):3d} reads: 2+ pulses until median {np.median(late):.2f}, 90% "
              f"{np.percentile(late, 90):.2f}, max {late.max():.2f}; 1+ pulse until median {np.median(exact):.2f}, "
              f"90% {np.percentile(exact, 90):.2f}, max {exact.max():.2f} (reads from {np.median(start):.2f})")
    print(f"\nreads where waiting would have changed the decision (within {TOLERANCE} pulse of the target or not): "
          f"{len(changed)}")
    for target, kind, decided, final, wanted, late in changed:
        print(f"  {target} {kind}: decided on {decided}, last report {final}, target {wanted}, strayed until "
              f"{late}")
    outcomes = np.array(outcomes)
    print(f"\nwhere the servos ended, from the target, pulses (last report of the app's read), {len(outcomes)} moves:")
    for servo in range(3):
        counts = {int(k): int(v) for k, v in zip(*np.unique(outcomes[:, servo], return_counts=True))}
        print(f"  servo {servo + 1}: {counts}")
    print("  every servo within 1 pulse: "
          f"{int(np.sum(outcomes.max(axis=1) <= 1))} of {len(outcomes)}")
    for target, wanted, final, late, rounds in rows:
        if np.abs(np.array(final) - wanted).max() > 1:
            print(f"    {target}: target {wanted}, ended {final}, {rounds} round(s)")


if __name__ == "__main__":
    main()
