#!/usr/bin/env python3
"""The true z formula from the desk the cup's seal found (find-desk.py's runs). No arm.

    cd jackal
    ../.venv/bin/python agenttools/true-z/fit-desk.py [run.jsonl ...]    # by default desk-seal*.jsonl but -1

The desk at each point is the highest height the arm held the cup at where it sealed, a whole step of its servos
(the owner's: a full step on the edge), less MARGIN_MM: true z 44 there. Each point is worked out again from its
tries, a seal kept only if it pulled the arm down (`seal_edge.judged()`); every run's points are pooled, a point
found in two runs counting twice. Its raw z against the firmware's reach is fitted as a line, as `maxarm/raw.py` has
it, and as a shallow arc, each with the arm's mounting tilt beside it -- a plane, desk.json's to take, not the
formula's. The ring found all round gives the tilt on its own: round one ring the reach does not change. The
formula is then hung on the band ANCHOR_REACH_MM, near the arm, where it moves surest and a tilt moves the desk
least (the owner's): it reads 44 there on average. Each point is predicted from the others too; those that miss
by more than OFF_FIT_MM are listed, with the points where no seal was found. Last, the owner's touch points
(config/desk-points.json, taken by eye) against the highest seal the proposal puts at each of them.

The reference place, found again through each run, shows how far the arm's readings drifted: a line through its
visits, and past DRIFT_MM over the run that run's points are corrected by it.
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

from maxarm import SealState, raw, seal      # noqa: E402

from seal_edge import SAME_MM, held_heights, judged, rebuild   # noqa: E402

OWNER_POINTS = HERE.parents[1] / "config" / "desk-points.json"
# The desk is this far under the highest seal: the owner's margin, deeper, so that a landing there seals for sure.
MARGIN_MM = 0.5
# Points this many times the typical miss off the table are left out of it, and named.
TABLE_OFF_TIMES = 3.0
DRIFT_MM = 0.5
# A reference visit this far from the run's others is left out of its drift: a run's first pump has read like a
# seal in the air.
REF_OFF_MM = 0.5
OFF_FIT_MM = 0.25
# A third of the way out to half way across the work area (the owner's), the firmware's reach.
ANCHOR_REACH_MM = (150.0, 190.0)
ALL_ROUND = "r175"
# The proposal: the line unless the arc predicts the points better by more than this.
CLOSE_MM = 0.02
SHOWN_AT_MM = (115.0, 130.0, 145.0, 160.0, 175.0, 190.0, 205.0, 220.0, 235.0, 250.0, 265.0)

Columns = Callable[[np.ndarray, np.ndarray, np.ndarray], List[np.ndarray]]
# The reach counted from raw.py's level reach; the arm's mounting tilt the last two columns, x and y.
SHAPES: Dict[str, Columns] = {
    "line": lambda r, x, y: [np.ones_like(r), r, x, y],
    "shallow arc": lambda r, x, y: [np.ones_like(r), r, r ** 2, x, y],
}
CHECKS: Dict[str, Columns] = {
    "line, no tilt": lambda r, x, y: [np.ones_like(r), r],
}


def load(path: Path, run: int) -> Tuple[List[dict], List[dict]]:
    """A run's points, worked out again, and its tries, each with its seal judged again; all tagged with `run`."""
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    tries = [dict(entry, run=run, state=judged(SealState(entry.get("library", entry["state"])), entry["pulled"]).value)
             for entry in entries if entry["kind"] == "try"]
    return [dict(entry, run=run) for entry in rebuild(entries)], tries


def ring_of(label: str) -> str:
    return label.split()[0]


def edge_raw(entry: dict) -> float:
    """The highest the cup was held at and sealed, raw z."""
    return entry["top_raw"][2]


def step_over(entry: dict) -> float:
    """How far over the edge the next height the arm can hold there is."""
    over = held_heights(entry["x"], entry["y"], entry["top"] + SAME_MM, entry["top"] + 2.0)
    return over[0][0] - entry["top"] if over else float("nan")


def drift(points: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """The reference's edge, raw z, at each time it was found: times, and the edges less the first."""
    found = sorted((entry["time"], edge_raw(entry)) for entry in points
                   if ring_of(entry["point"]) == "ref" and entry["found"])
    middle = median(edge for _, edge in found) if found else 0.0
    odd = [edge for _, edge in found if abs(edge - middle) > REF_OFF_MM]
    if odd:
        print("  left out of the drift, too far from the other reference visits: "
              + ", ".join(f"{edge - middle:+.2f}" for edge in odd))
    found = [(when, edge) for when, edge in found if abs(edge - middle) <= REF_OFF_MM]
    times = np.array([when for when, _ in found])
    edges = np.array([edge for _, edge in found])
    return times, (edges - edges[0] if len(edges) else edges)


def drift_correction(points: List[dict], grid: List[dict]) -> np.ndarray:
    """What to take off each of `grid`'s heights for the drift its run's reference saw; zero under DRIFT_MM."""
    times, moved = drift(points)
    if not len(moved):
        return np.zeros(len(grid))
    print(f"  the reference found {len(moved)} times over {(times[-1] - times[0]) / 60:.0f} min: moved "
          + ", ".join(f"{value:+.2f}" for value in moved))
    if len(moved) < 3:
        return np.zeros(len(grid))
    per_s = np.polyfit(times - times[0], moved, 1)[0]
    over_run = per_s * (times[-1] - times[0])
    print(f"  a line through them: {over_run:+.2f} mm over the run"
          + (f"; past {DRIFT_MM} mm, its points corrected by it" if abs(over_run) > DRIFT_MM else ""))
    if abs(over_run) <= DRIFT_MM:
        return np.zeros(len(grid))
    return per_s * (np.array([entry["time"] for entry in grid]) - times[0])


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
    """What raw.py as it is adds to raw z on the desk at a firmware reach: under the desk the height term is 0."""
    under = 30.0
    return raw.true_position((0.0, -reach, under))[2] - under


def line_term(reach: float) -> float:
    return raw.TRUE_Z_PER_MM_OF_REACH * (reach - raw.LEVEL_REACH_MM)


def wave_table(points: List[dict], desk: np.ndarray, reach: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    """The desk followed ring by ring: a table of the firmware's reach and what true z adds on top of raw.py's line
    there, straight between the rings, the tilt fitted beside it; printed as raw.py takes it."""
    rings = sorted({ring_of(entry["point"]) for entry in points}, key=lambda ring: float(ring[1:]))
    ring = np.array([ring_of(entry["point"]) for entry in points])
    nodes = np.array([np.median(reach[ring == name]) for name in rings])
    hats = [np.interp(reach, nodes, row) for row in np.eye(len(nodes))]
    keep = np.ones(len(desk), dtype=bool)
    for _ in range(2):
        matrix = np.column_stack(hats + [x, y])
        coef, *_ = np.linalg.lstsq(matrix[keep], desk[keep], rcond=None)
        left = desk - matrix @ coef
        keep = np.abs(left) <= TABLE_OFF_TIMES * typical(left[keep])
    dropped = [f"{entry['point']} run {entry['run']} ({value:+.2f})" for entry, value, kept in zip(points, left, keep)
               if not kept]
    table = [(float(node), raw.DESK_Z_MM - value - line_term(node)) for node, value in zip(nodes, coef[:len(nodes)])]
    print(f"\nthe desk ring by ring, a table straight between the rings, the tilt beside it ({keep.sum()} points; the "
          f"points kept miss it {spread(left[keep])}" + (f"; left out: {', '.join(dropped)}" if dropped else "") + ")")
    print("    raw.py: DESK_WAVE_MM = (" + ", ".join(f"({node:.1f}, {value:+.2f})" for node, value in table) + ")")
    print("    against today's: " + ", ".join(f"{value + line_term(node) - today_term(node):+.2f} at {node:.0f}"
                                            for node, value in table))


def tilt_round_one_ring(points: List[dict], today: np.ndarray, raw_xyz: np.ndarray) -> None:
    """The arm's mounting tilt from the ring found all round alone: a plane through its points."""
    mine = [index for index, entry in enumerate(points) if ring_of(entry["point"]) == ALL_ROUND]
    bearings = [math.degrees(math.atan2(raw_xyz[index, 0], -raw_xyz[index, 1])) for index in mine]
    if len(mine) < 5 or max(bearings) - min(bearings) < 120.0:
        return
    x, y = raw_xyz[mine, 0], raw_xyz[mine, 1]
    coef, left, each = fit([np.ones_like(x), x, y], today[mine])
    slope = math.hypot(coef[1], coef[2])
    downhill = math.degrees(math.atan2(-coef[1], coef[2]))     # the bearing the desk falls toward, from the arm
    print(f"\nthe arm's mounting tilt, round the {ALL_ROUND[1:]} ring alone ({len(mine)} points, bearings "
          f"{min(bearings):+.0f} to {max(bearings):+.0f}): {slope * 100:.2f} mm per 100 mm, the desk reading lower "
          f"toward bearing {downhill:+.0f}; toward +x {coef[1] * 100:+.2f}, outward (-y) {-coef[2] * 100:+.2f} per 100 "
          f"mm; the points off that plane {spread(left)}, each from the others {spread(each)}")


def repeats(points: List[dict]) -> None:
    """Points found in more than one run: how far apart their highest seals read."""
    by_label: Dict[str, List[dict]] = {}
    for entry in points:
        by_label.setdefault(entry["point"], []).append(entry)
    pairs = [(label, found) for label, found in by_label.items() if len(found) > 1]
    if not pairs:
        return
    apart = np.array([found[-1]["top"] - found[0]["top"] for _, found in pairs])
    far = [(label, value) for (label, _), value in zip(pairs, apart) if abs(value) > OFF_FIT_MM]
    print(f"\nfound in two runs ({len(pairs)} points), the later less the earlier: {spread(apart)}, on average "
          f"{apart.mean():+.2f}" + (": " + ", ".join(f"{label} {value:+.2f}" for label, value in far) if far else ""))


def against_owner(coef: np.ndarray, degree: int, mean_xy: Tuple[float, float], reach_span: Tuple[float, float],
                  bearing_max: float) -> None:
    """The owner's touches against the highest seal the proposal puts there: by how much the seal is higher."""
    if not OWNER_POINTS.exists():
        return
    content = json.loads(OWNER_POINTS.read_text())
    if content.get("heights") != "raw":
        print(f"\nyour touch points: {OWNER_POINTS} does not hold raw heights; not compared")
        return
    inside = []
    for point in content["points"]:
        reach = math.hypot(point["x"], point["y"])
        bearing = math.degrees(math.atan2(point["x"], -point["y"]))
        seal_z = (raw.DESK_Z_MM - desk_term(coef, degree, reach) + coef[-2] * (point["x"] - mean_xy[0])
                  + coef[-1] * (point["y"] - mean_xy[1]) + MARGIN_MM)
        if reach_span[0] <= reach <= reach_span[1] and abs(bearing) <= bearing_max:
            inside.append(seal_z - point["z"])
    if inside:
        print(f"\nyour touch points ({OWNER_POINTS.name}) within the points found ({len(inside)}): the highest seal "
              f"higher by {np.mean(inside):+.2f} on average, {min(inside):+.2f} to {max(inside):+.2f}")


def by_ring(points: List[dict], tries: List[dict], today: np.ndarray, level: np.ndarray, left: np.ndarray) -> None:
    rings = sorted({ring_of(entry["point"]) for entry in points}, key=lambda ring: float(ring[1:]))
    print("\nby reach            desk by today's    tilt          left by the    stops short    seal by dip    loose"
          " dip    a seal        next step")
    print("                    formula (44 ideal)  taken out     proposal      coming down    up to          from     "
          "    pulls         over the edge")
    for ring in rings:
        mine = [index for index, entry in enumerate(points) if ring_of(entry["point"]) == ring]
        steps = [step_over(points[index]) for index in mine]
        ring_tries = [entry for entry in tries if ring_of(entry["point"]) == ring]
        short = [entry["read"][2] - entry["sent"] for entry in ring_tries if entry["state"] == "loose"]
        by_dip = [entry["dip"] for entry in ring_tries if entry["state"] == "sealed" and entry["dip"] is not None
                  and (entry["pulled"] or 0.0) < seal.PULLED_MM]
        loose = [entry["dip"] for entry in ring_tries if entry["state"] == "loose" and entry["dip"] is not None]
        pulls = [entry["pulled"] for entry in ring_tries if entry["state"] == "sealed" and entry["pulled"] is not None]
        cells = [f"{today[mine].mean():6.2f}", f"{level[mine].mean():6.2f}", f"{left[mine].mean():+6.2f}",
                 f"{median(short):5.2f}" if short else "    -",
                 f"{max(by_dip):5.1f}" if by_dip else "    -",
                 f"{min(loose):5.1f}" if loose else "    -",
                 f"{min(pulls):.2f}-{max(pulls):.2f}" if pulls else "-",
                 f"{np.nanmedian(steps):.2f}" if steps else "-"]
        print(f"  {ring[1:]:>4s} mm, {len(mine):2d} points   {cells[0]}            {cells[1]}        {cells[2]}        "
              f"{cells[3]}          {cells[4]}          {cells[5]}        {cells[6]:9s}     {cells[7]}")
    print(f"  (the library calls a seal under a dip of {seal.SEALED_BELOW:g}, loose over {seal.LOOSE_ABOVE:g})")
    needs = [raw.DESK_Z_MM - float(np.median(level[[ring_of(entry["point"]) == ring for entry in points]]))
             for ring in rings]
    print("as a table by reach, the tilt taken out, what each ring needs against today's formula for its highest seal "
          "to read 44 (the median): " + ", ".join(f"{need:+.2f} at {ring[1:]}" for need, ring in zip(needs, rings)))


def main() -> None:
    paths = ([Path(arg) for arg in sys.argv[1:]]
             or [path for path in HERE.glob("desk-seal*.jsonl") if not path.stem.endswith("-1")])
    paths.sort(key=lambda path: json.loads(path.read_text().splitlines()[0])["time"])     # in the order run
    points, tries, corrections = [], [], []
    for run, path in enumerate(paths):
        found, tried = load(path, run)
        grid = [entry for entry in found if ring_of(entry["point"]) != "ref" and entry["found"]]
        print(f"{path.name} (run {run}): {len(grid)} points found, {sum(1 for entry in found if not entry['found'])}"
              f" not; {sum(1 for entry in grid if entry['is_resolved'])} with no height held between the highest seal"
              f" and the lowest miss")
        for entry in found:
            if not entry["found"]:
                print(f"  no edge at {entry['point']} ({entry['x']:.0f}, {entry['y']:.0f}): {entry.get('why', '')}")
        corrections.append(drift_correction(found, grid))
        points += grid
        tries += tried
    repeats(points)

    correction = np.concatenate(corrections)
    raw_xyz = np.array([entry["top_raw"] for entry in points], dtype=float)
    desk = np.array([edge_raw(entry) for entry in points]) - correction - MARGIN_MM
    reach = np.hypot(raw_xyz[:, 0], raw_xyz[:, 1])
    out = reach - raw.LEVEL_REACH_MM
    mean_xy = (raw_xyz[:, 0].mean(), raw_xyz[:, 1].mean())
    x, y = raw_xyz[:, 0] - mean_xy[0], raw_xyz[:, 1] - mean_xy[1]
    today = np.array([raw.true_position((px, py, pz))[2] for (px, py, _), pz in zip(raw_xyz, desk)])
    print(f"\ntoday's formula puts the desk at {today.mean():.2f} on average, {today.min():.2f} to {today.max():.2f}"
          f" (true z 44 is the desk), {len(points)} points")
    tilt_round_one_ring(points, today, raw_xyz)

    results = {}
    print("\nfitted to every point:")
    for name, columns in {**SHAPES, **CHECKS}.items():
        coef, left, each = fit(columns(out, x, y), desk)
        results[name] = (coef, left, each)
        line = f"  {name}: {spread(left)}; each from the others {spread(each)}"
        if name in SHAPES:
            line += f"; the tilt toward +x {coef[-2] * 100:+.2f}, outward {-coef[-1] * 100:+.2f} per 100 mm"
        if name == "shallow arc":
            ends = ((115.0 - raw.LEVEL_REACH_MM) ** 2 + (265.0 - raw.LEVEL_REACH_MM) ** 2) / 2.0
            line += (f"; at 190 the arc lies {coef[2] * ((190.0 - raw.LEVEL_REACH_MM) ** 2 - ends):+.2f} off the line"
                     f" through its ends at 115 and 265")
        print(line)

    is_arc = typical(results["shallow arc"][2]) < typical(results["line"][2]) - CLOSE_MM
    proposed = "shallow arc" if is_arc else "line"
    coef, left, each = results[proposed]
    degree = len(coef) - 3
    in_band = (reach >= ANCHOR_REACH_MM[0]) & (reach <= ANCHOR_REACH_MM[1])
    coef = coef.copy()
    coef[0] += left[in_band].mean()                 # hung on the band: its points read 44 on average
    left = left - left[in_band].mean()
    print(f"\nproposed: the {proposed}, hung on reach {ANCHOR_REACH_MM[0]:.0f}-{ANCHOR_REACH_MM[1]:.0f} "
          f"({in_band.sum()} points). On the desk, true z = raw z "
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

    off = [(abs(value), entry["point"], entry["run"], value)
           for entry, value in zip(points, each) if abs(value) > OFF_FIT_MM]
    print(f"\npoints the others put more than {OFF_FIT_MM} mm off: " + (", ".join(
        f"{label} run {run} ({value:+.2f})" for _, label, run, value in sorted(off, reverse=True)) if off else
        "none"))

    level = today - coef[-2] * x - coef[-1] * y      # the tilt taken out
    by_ring(points, tries, today, level, left)
    wave_table(points, desk, reach, x, y)
    bearing = np.degrees(np.arctan2(raw_xyz[:, 0], -raw_xyz[:, 1]))
    against_owner(coef, degree, mean_xy, (reach.min(), reach.max()), float(np.max(np.abs(bearing))))


if __name__ == "__main__":
    main()
