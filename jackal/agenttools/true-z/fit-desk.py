#!/usr/bin/env python3
"""The true z formula from the desk the cup's seal found (find-desk.py's desk-seal.jsonl). No arm.

    cd jackal
    ../.venv/bin/python agenttools/true-z/fit-desk.py [path/to/desk-seal.jsonl]

The desk at each point is MARGIN_MM under the highest height the cup seals at (the owner's): true z 44 there. That
height is taken half-way between the highest reading that sealed and the lowest that did not. Its raw z against the
firmware's reach is fitted as a line, as `maxarm/raw.py` has it, and as curves, each with the desk's tilt across
(x) beside it, which is desk.json's to take, not the formula's; the simplest that fits about as well as the best is
proposed. Tilted outward (-y) as well only as a check: in front of the arm a tilt outward and the reach term change
heights much alike, and these points tell them apart poorly. Each point is predicted from all the others, and those
that miss it by more than OFF_FIT_MM are listed, with the points where no seal was found.

The reference place, found again through the run, shows how far the arm's readings drifted: a line through its
visits, and past DRIFT_MM over the run the points are corrected by it. One visit alone is no measure: each edge is
found only to within find-desk's RESOLUTION_MM or so.
"""

import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Callable, Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from maxarm import raw, seal      # noqa: E402

MARGIN_MM = 0.5
DRIFT_MM = 0.5
OFF_FIT_MM = 0.25
# The proposal: the simplest shape whose points, each predicted from the others, miss by no more than this over
# the best shape's.
CLOSE_MM = 0.02
SHOWN_AT_MM = (120.0, 145.0, 170.0, 195.0, 220.0, 245.0, 265.0)

Columns = Callable[[np.ndarray, np.ndarray, np.ndarray], List[np.ndarray]]
# The reach is counted from raw.py's level reach; a desk tilted across ends the columns with x, both ways x and y.
SHAPES: Dict[str, Columns] = {
    "line": lambda r, x, y: [np.ones_like(r), r, x],
    "curve, squared": lambda r, x, y: [np.ones_like(r), r, r ** 2, x],
    "curve, cubed": lambda r, x, y: [np.ones_like(r), r, r ** 2, r ** 3, x],
}
CHECKS: Dict[str, Columns] = {
    "line, the desk level": lambda r, x, y: [np.ones_like(r), r],
    "line, the desk tilted both ways": lambda r, x, y: [np.ones_like(r), r, x, y],
}


def load(path: Path) -> Tuple[List[dict], List[dict]]:
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return ([entry for entry in entries if entry["kind"] == "point"],
            [entry for entry in entries if entry["kind"] == "try"])


def ring_of(label: str) -> str:
    return label.split()[0]


def edge_raw(entry: dict) -> float:
    """The highest the cup seals at, raw z: half-way between the highest seal and the lowest miss, which on the desk
    lie the same way apart in raw z as in true z."""
    return entry["top_raw"][2] + (entry["bottom"] - entry["top"]) / 2.0


def drift(points: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """The reference's edge, raw z, at each time it was found: times, and the edges less the first."""
    found = sorted((entry["time"], edge_raw(entry)) for entry in points
                   if ring_of(entry["point"]) == "ref" and entry["found"])
    times = np.array([when for when, _ in found])
    edges = np.array([edge for _, edge in found])
    return times, (edges - edges[0] if len(edges) else edges)


def fit(columns: List[np.ndarray], z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coefficients, what is left at each point, and each point predicted from all the others."""
    matrix = np.column_stack(columns)
    coef, *_ = np.linalg.lstsq(matrix, z, rcond=None)
    out = []
    for index in range(len(z)):
        keep = np.arange(len(z)) != index
        one, *_ = np.linalg.lstsq(matrix[keep], z[keep], rcond=None)
        out.append(z[index] - matrix[index] @ one)
    return coef, z - matrix @ coef, np.array(out)


def typical(values: np.ndarray) -> float:
    return math.sqrt(float(np.mean(values ** 2)))


def spread(values: np.ndarray) -> str:
    return f"typically {typical(values):.2f}, worst {float(np.max(np.abs(values))):.2f}"


def desk_term(coef: np.ndarray, degree: int, reach: float) -> float:
    """What true z adds to raw z on the desk at `reach`, by the fitted shape: 44 less the desk's raw z there."""
    out = reach - raw.LEVEL_REACH_MM
    return raw.DESK_Z_MM - sum(coef[power] * out ** power for power in range(degree + 1))


def today_term(reach: float) -> float:
    return raw.TRUE_Z_PER_MM_OF_REACH * (reach - raw.LEVEL_REACH_MM)


def by_ring(points: List[dict], tries: List[dict], today: Dict[str, float], left: Dict[str, float]) -> None:
    rings = sorted({ring_of(entry["point"]) for entry in points if ring_of(entry["point"]) != "ref"},
                   key=lambda ring: float(ring[1:]))
    print("\nby reach            desk by today's    left by the    stops short    seal by dip    loose dip")
    print("                    formula (44 ideal)  proposal      coming down    up to          from")
    for ring in rings:
        labels = [entry["point"] for entry in points if ring_of(entry["point"]) == ring and entry["found"]]
        mine = [entry for entry in tries if ring_of(entry["point"]) == ring]
        short = [entry["read"][2] - entry["sent"] for entry in mine if entry["state"] == "loose"]
        by_dip = [entry["dip"] for entry in mine if entry["state"] == "sealed" and entry["dip"] is not None
                  and (entry["pulled"] or 0.0) < seal.PULLED_MM]
        loose = [entry["dip"] for entry in mine if entry["state"] == "loose" and entry["dip"] is not None]
        cells = [f"{np.mean([today[label] for label in labels]):6.2f}" if labels else "     -",
                 f"{np.mean([left[label] for label in labels]):+6.2f}" if labels else "     -",
                 f"{median(short):5.2f}" if short else "    -",
                 f"{max(by_dip):5.1f}" if by_dip else "    -",
                 f"{min(loose):5.1f}" if loose else "    -"]
        print(f"  {ring[1:]:>4s} mm, {len(labels):2d} points   {cells[0]}            {cells[1]}        "
              f"{cells[2]}          {cells[3]}          {cells[4]}")
    print(f"  (the library calls a seal under a dip of {seal.SEALED_BELOW:g}, loose over {seal.LOOSE_ABOVE:g})")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "desk-seal.jsonl"
    points, tries = load(path)
    grid = [entry for entry in points if ring_of(entry["point"]) != "ref" and entry["found"]]
    missing = [entry for entry in points if not entry["found"]]
    print(f"{len(grid)} points found, {len(missing)} not, from {path}; the desk {MARGIN_MM} mm under the highest seal")
    for entry in missing:
        print(f"  no edge at {entry['point']} ({entry['x']:.0f}, {entry['y']:.0f}): {entry.get('why', '')}")
    overlaps = [entry["point"] for entry in grid if entry["is_overlap"]]
    if overlaps:
        print(f"  a seal read higher than a miss at: {', '.join(overlaps)}")

    times, moved = drift(points)
    correction = np.zeros(len(grid))
    if len(moved):
        print(f"the reference place found {len(moved)} times over {(times[-1] - times[0]) / 60:.0f} min: moved "
              + ", ".join(f"{value:+.2f}" for value in moved))
        if len(moved) >= 3:
            per_s = np.polyfit(times - times[0], moved, 1)[0]
            over_run = per_s * (times[-1] - times[0])
            print(f"  a line through them: {over_run:+.2f} mm over the run")
            if abs(over_run) > DRIFT_MM:
                correction = per_s * (np.array([entry["time"] for entry in grid]) - times[0])
                print(f"  past {DRIFT_MM} mm: each point corrected by the line at its time")

    raw_xyz = np.array([entry["top_raw"] for entry in grid], dtype=float)
    desk = np.array([edge_raw(entry) for entry in grid]) - correction - MARGIN_MM
    reach = np.hypot(raw_xyz[:, 0], raw_xyz[:, 1])
    out = reach - raw.LEVEL_REACH_MM
    x, y = raw_xyz[:, 0] - raw_xyz[:, 0].mean(), raw_xyz[:, 1] - raw_xyz[:, 1].mean()
    today = np.array([raw.true_position((px, py, pz))[2] for (px, py, _), pz in zip(raw_xyz, desk)])
    print(f"\ntoday's formula puts the desk at {today.mean():.2f} on average, {today.min():.2f} to {today.max():.2f}"
          f" (true z 44 is the desk)")

    results = {}
    for name, columns in {**SHAPES, **CHECKS}.items():
        coef, left, each = fit(columns(out, x, y), desk)
        results[name] = (coef, left, each)
        line = f"  {name}: {spread(left)}; each from the others {spread(each)}"
        if name in SHAPES:
            line += f"; the desk rises {coef[-1] * np.ptp(x):+.2f} mm toward +x over the points"
        if name.endswith("both ways"):
            line += (f"; the desk rises {coef[-2] * np.ptp(x):+.2f} toward +x, {-coef[-1] * np.ptp(y):+.2f} outward,"
                     f" the line {-coef[1]:.4f} per mm")
        print(line)

    best = min(typical(results[name][2]) for name in SHAPES)
    proposed = next(name for name in SHAPES if typical(results[name][2]) <= best + CLOSE_MM)
    coef, left, each = results[proposed]
    degree = len(coef) - 2
    print(f"\nproposed: the {proposed}. On the desk, true z = raw z + "
          + " ".join(f"{-coef[power] if power else raw.DESK_Z_MM - coef[0]:+.6g}"
                     + ("" if not power else f" x (reach - {raw.LEVEL_REACH_MM:.0f})" + ("" if power == 1 else
                                                                                         f"^{power}"))
                     for power in range(degree + 1)))
    if degree == 1:
        slope = -coef[1]
        print(f"    as raw.py has it: TRUE_Z_PER_MM_OF_REACH = {slope:.4f} (now {raw.TRUE_Z_PER_MM_OF_REACH}), "
              f"LEVEL_REACH_MM = {raw.LEVEL_REACH_MM + (coef[0] - raw.DESK_Z_MM) / slope:.1f} "
              f"(now {raw.LEVEL_REACH_MM})")
    print("    against today's: " + ", ".join(
        f"{desk_term(coef, degree, at) - today_term(at):+.2f} at {at:.0f}" for at in SHOWN_AT_MM))

    labels = [entry["point"] for entry in grid]
    off = [(abs(value), label, reach[index], math.degrees(math.atan2(raw_xyz[index, 0], -raw_xyz[index, 1])), value)
           for index, (label, value) in enumerate(zip(labels, each)) if abs(value) > OFF_FIT_MM]
    print(f"\npoints the others put more than {OFF_FIT_MM} mm off: " + (", ".join(
        f"{label} (reach {at:.0f}, bearing {bearing:+.0f}: {value:+.2f})" for _, label, at, bearing, value
        in sorted(off, reverse=True)) if off else "none"))
    bearing = np.degrees(np.arctan2(raw_xyz[:, 0], -raw_xyz[:, 1]))
    sides = [("toward -x", bearing < -15.0), ("centre", np.abs(bearing) <= 15.0), ("toward +x", bearing > 15.0)]
    print("left by the proposal by bearing: " + ", ".join(
        f"{name} {left[mask].mean():+.2f} ({mask.sum()})" for name, mask in sides if mask.any()))

    by_ring(points, tries, dict(zip(labels, today)), dict(zip(labels, left)))


if __name__ == "__main__":
    main()
