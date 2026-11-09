#!/usr/bin/env python3
"""Where the cup starts to seal, against the true-z formula, and whether another shape fits it better. No arm.

    cd jackal
    ../.venv/bin/python agenttools/mechanics/edge_fit.py runs/edge-*.jsonl

The edge at a place is the raw height that best splits the seals (at or under it) from the leaks (above it), the
middle of the best range. The same is worked out for the 5 Nov find-desk runs (files/true-z/), so both days are
measured one way. Each edge is then put through `raw.py` as it is now: its height over the formula's desk. Were
the formula right, that height would be the same everywhere.

Shapes fitted to the heights, each scored by leaving each place out in turn:
- level:  one number, the formula as it is;
- tilt:   plus a plane in x and y, the arm standing tilted on the desk;
- reach:  plus a straight line in the firmware's reach;
- both:   tilt and reach.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
JACKAL = HERE.parents[1]
sys.path.insert(0, str(JACKAL / "src"))

from maxarm import raw          # noqa: E402

THEN = [JACKAL / "files/true-z/2026-11-05-run2-letter.jsonl", JACKAL / "files/true-z/2026-11-05-run3-affirm.jsonl"]

Try = Tuple[Tuple[float, float, float], str]       # raw position, sealed / loose


def edge(tries: List[Try]) -> Optional[Tuple[float, float, float, int, int]]:
    """(raw x, raw y, raw z of the edge, tries misjudged by it, tries) or None without both a seal and a leak."""
    judged = [(p, s) for p, s in tries if s in ("sealed", "loose")]
    if not any(s == "sealed" for _, s in judged) or not any(s == "loose" for _, s in judged):
        return None
    heights = sorted({p[2] for p, _ in judged})
    cuts = [(a + b) / 2.0 for a, b in zip(heights, heights[1:])] + [heights[0] - 0.01, heights[-1] + 0.01]

    def wrong(cut: float) -> int:
        return sum((p[2] <= cut) != (s == "sealed") for p, s in judged)

    fewest = min(wrong(cut) for cut in cuts)
    best = [cut for cut in cuts if wrong(cut) == fewest]
    return (mean(p[0] for p, _ in judged), mean(p[1] for p, _ in judged), (min(best) + max(best)) / 2.0, fewest,
            len(judged))


def load_then() -> Dict[str, List[Try]]:
    places = defaultdict(list)
    for path in THEN:
        for line in path.open():
            entry = json.loads(line)
            if entry.get("kind") == "try":
                name = "ref" if entry["point"].startswith("ref") else entry["point"]
                places[name].append((tuple(entry["read_raw"]), entry["state"]))
    return places


def load_now(paths: List[str]) -> Dict[str, List[Try]]:
    places = defaultdict(list)
    for path in paths:
        for line in open(path):
            entry = json.loads(line)
            if entry.get("kind") == "edge" and "raw" in entry:
                places[entry["label"].rsplit(" f", 1)[0]].append((tuple(entry["raw"]), entry.get("seal")))
    return places


def over_desk(x: float, y: float, z: float) -> float:
    return raw.true_position((x, y, z))[2] - raw.DESK_Z_MM


SHAPES = {
    "level": lambda x, y, r: [1.0],
    "tilt": lambda x, y, r: [1.0, x, y],
    "reach": lambda x, y, r: [1.0, r - raw.LEVEL_REACH_MM],
    "both": lambda x, y, r: [1.0, x, y, r - raw.LEVEL_REACH_MM],
}


def fit(rows: List[Tuple[float, float, float, float]]) -> None:
    """rows: raw x, y, reach, height over the formula's desk."""
    for name, columns in SHAPES.items():
        a = np.array([columns(x, y, r) for x, y, r, _ in rows])
        h = np.array([h for *_, h in rows])
        coef = np.linalg.lstsq(a, h, rcond=None)[0]
        misses = []
        for i in range(len(rows)):
            keep = [j for j in range(len(rows)) if j != i]
            c = np.linalg.lstsq(a[keep], h[keep], rcond=None)[0]
            misses.append(h[i] - a[i] @ c)
        rms = math.sqrt(mean(m * m for m in misses))
        terms = {"level": "", "tilt": "  x %+.4f/mm y %+.4f/mm", "reach": "  reach %+.4f/mm",
                 "both": "  x %+.4f/mm y %+.4f/mm reach %+.4f/mm"}[name]
        print(f"  {name:6s} left-out miss rms {rms:.2f} mm, worst {max(abs(m) for m in misses):.2f};"
              f" offset {coef[0]:+.2f}" + (terms % tuple(coef[1:]) if terms else ""))


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    then, now = load_then(), load_now(paths)
    print("place        5 Nov over desk   today over desk   shift   (edge over the formula's desk, mm; misjudged/tries)")
    rows_now, rows_then = [], []
    for name in sorted(now, key=lambda n: (n != "ref", n)):
        b = edge(now[name])
        a = edge(then.get(name, []))
        if b:
            rows_now.append((b[0], b[1], math.hypot(b[0], b[1]), over_desk(*b[:3])))
        if a:
            rows_then.append((a[0], a[1], math.hypot(a[0], a[1]), over_desk(*a[:3])))
        text_a = f"{over_desk(*a[:3]):+.2f} ({a[3]}/{a[4]:2d})" if a else "      -     "
        text_b = f"{over_desk(*b[:3]):+.2f} ({b[3]}/{b[4]:2d})" if b else "      -     "
        shift = f"{over_desk(*b[:3]) - over_desk(*a[:3]):+.2f}" if a and b else "  -"
        print(f"{name:11s}  {text_a:15s}   {text_b:15s}   {shift}")
    print(f"\ntoday, {len(rows_now)} places:")
    fit(rows_now)
    print(f"5 Nov, the same places, {len(rows_then)}:")
    fit(rows_then)


if __name__ == "__main__":
    main()
