#!/usr/bin/env python3
"""GRID-PAPER-jackal's 25 desk points (3 Nov evening: the firmware's readings, and where the paper puts the cup by its
rim) through raw.py as it is today, and through other turn terms, each laid on the paper by one turn and shift
(least squares): how far each point is from the paper, mm.

    PYTHONPATH=src ../.venv/bin/python3 agenttools/fitcheck/paper-now.py
"""
import math, re
from pathlib import Path
import numpy as np
from maxarm import raw

rows = []
for line in Path("../work/GRID-PAPER-jackal.md").read_text().splitlines():
    m = re.match(r"\| (-?\d+), (-?\d+) \| (-?[\d.]+), (-?[\d.]+), (-?[\d.]+) \| (-?[\d.]+), (-?[\d.]+) \| "
                 r"(-?[\d.]+), (-?[\d.]+) \|", line)
    if m:
        v = [float(g) for g in m.groups()]
        rows.append((v[2:5], v[7:9]))
firmware = np.array([r[0] for r in rows])
paper = np.array([r[1] for r in rows])


def rigid(a, b, scale=False):
    """a laid on b by a turn and shift (and a scale): the leftovers."""
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    u, s, vt = np.linalg.svd((b - mb).T @ (a - ma))
    rot = u @ vt
    k = s.sum() / ((a - ma) ** 2).sum() if scale else 1.0
    return b - (mb + k * (a - ma) @ rot.T), k


def library(sin_deg, cos_deg, desk_reach=raw.REACH_AT_DESK_MM, per_mm=raw.REACH_LESS_PER_MM):
    out = []
    for x, y, z in firmware:
        t = raw.true_position((x, y, z))                       # today's z, so today's height above the desk
        h = max(t[2] - raw.DESK_Z_MM, 0.0)
        r, b = math.hypot(x, y), math.atan2(x, -y)
        r += desk_reach - per_mm * min(h, 44.0)
        b += math.radians(sin_deg * math.sin(2 * b) + cos_deg * (math.cos(2 * b) - 1))
        out.append((r * math.sin(b), -r * math.cos(b)))
    return np.array(out)


def show(label, xy):
    left, _ = rigid(xy, paper)
    d = np.linalg.norm(left, axis=1)
    _, k = rigid(xy, paper, scale=True)
    print(f"  {label:58} {np.median(d):4.2f} typical, {d.max():4.2f} worst; free scale {k:.4f}")


print(f"{len(rows)} points, heights over the desk {min(raw.true_position(tuple(f))[2] for f in firmware) - 44:.1f}"
      f" to {max(raw.true_position(tuple(f))[2] for f in firmware) - 44:.1f} mm")
show("raw.py as on 3 Nov evening: reach +1.6, turn (-1.36, +0.33)", library(-1.36, 0.33, 1.6, 0.0))
show("raw.py today: reach +6.26 - 0.110 h, turn (-1.36, +0.33)", library(-1.36, 0.33))
show("today's reach, turn sin part only (-1.36, 0)", library(-1.36, 0.0))
show("today's reach, no turn correction", library(0.0, 0.0))
best = min(((float(np.median(np.linalg.norm(rigid(library(a, c), paper)[0], axis=1))), a, c)
            for a in np.arange(-2.6, -0.59, 0.05) for c in np.arange(-0.4, 0.81, 0.05)))
show(f"today's reach, the best turn for the paper ({best[1]:+.2f}, {best[2]:+.2f})", library(best[1], best[2]))
for c in (0.0,):
    sub = min(((float(np.median(np.linalg.norm(rigid(library(a, c), paper)[0], axis=1))), a)
               for a in np.arange(-2.6, -0.59, 0.05)))
    show(f"today's reach, cos part 0, the best sin part ({sub[1]:+.2f})", library(sub[1], c))
