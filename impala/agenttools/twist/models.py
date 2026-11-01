#!/usr/bin/env python3
"""Which systematic error explains Fine Tune's misses? Each model is fitted to the points as SAVE lays the arm (a
turn, shift and stretch: any affine map, six numbers) plus its own few numbers, and judged by leave-one-out: each
point's miss predicted from the others. Reads twist-<name>.json (residuals.py) and the desk.json of its day.

    .venv/bin/python impala/agenttools/twist/models.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT)]

from maxarm import raw                                              # noqa: E402

SETS = {"now": ROOT / "config", "old1030": ROOT.parent / "work" / "config-impala-1030"}


def load(name):
    rows = json.loads((HERE / f"twist-{name}.json").read_text())["rows"]
    plane = json.loads((SETS[name] / "desk.json").read_text())["desk_to_arm"][2]
    arm = np.array([row["arm"] for row in rows])
    seen = np.array([row["seen"] for row in rows])
    z = np.array([row["z"] + plane[0] * row["arm"][0] + plane[1] * row["arm"][1] + plane[3] for row in rows])
    firmware = np.array([raw.raw_position((x, y, zz)) for (x, y), zz in zip(arm, z)])
    return [row["id"] for row in rows], arm, seen, firmware, plane


def with_constants(firmware, **constants):
    """The arm's x, y from the firmware's positions, raw.py's constants changed as given."""
    kept = {name: getattr(raw, name) for name in constants}
    try:
        for name, value in constants.items():
            setattr(raw, name, value)
        return np.array([raw.true_position(tuple(p))[:2] for p in firmware])
    finally:
        for name, value in kept.items():
            setattr(raw, name, value)


def fields(arm, terms):
    """Displacement fields, one per number, at each point: (n, 2) each."""
    reach = np.hypot(arm[:, 0], arm[:, 1])
    bearing = np.arctan2(arm[:, 0], -arm[:, 1])
    out = np.column_stack((np.sin(bearing), -np.cos(bearing)))
    round_ = np.column_stack((np.cos(bearing), np.sin(bearing)))
    make = {"radial": out, "lateral": round_,
            "wave_sin": round_ * (reach * np.sin(2 * bearing))[:, None] * math.pi / 180,
            "wave_cos": round_ * (reach * (np.cos(2 * bearing) - 1))[:, None] * math.pi / 180,
            "reach_scale_sq": out * ((reach - 200) ** 2 / 100)[:, None]}
    return [make[term] for term in terms]


def design(arm, seen, terms, projective):
    """Rows: arm = affine(seen) [+ projective] - sum p_k field_k; unknowns affine 6, projective 2, fields."""
    n = len(arm)
    columns = []
    for axis in range(2):
        block = np.zeros((n, 6))
        block[:, 3 * axis:3 * axis + 3] = np.column_stack((seen, np.ones(n)))
        columns.append(block)
    rows = np.vstack(columns)
    if projective:
        rows = np.hstack((rows, np.vstack([-seen * arm[:, [axis]] for axis in range(2)])))
    for field in fields(arm, terms):
        rows = np.hstack((rows, -np.concatenate((field[:, 0], field[:, 1]))[:, None]))
    return rows, np.concatenate((arm[:, 0], arm[:, 1]))


def fit(arm, seen, terms=(), projective=False):
    rows, target = design(arm, seen, terms, projective)
    numbers = np.linalg.lstsq(rows, target, rcond=None)[0]
    misses = (target - rows @ numbers).reshape(2, -1).T
    return numbers, misses


def loo(arm, seen, terms=(), projective=False):
    out = []
    for i in range(len(arm)):
        keep = np.arange(len(arm)) != i
        numbers, _ = fit(arm[keep], seen[keep], terms, projective)
        rows, target = design(arm[[i]], seen[[i]], terms, projective)
        out.append(np.hypot(*(target - rows @ numbers)))
    return np.array(out)


def summary(arm, seen, terms=(), projective=False):
    numbers, misses = fit(arm, seen, terms, projective)
    across = np.hypot(misses[:, 0], misses[:, 1])
    held = loo(arm, seen, terms, projective)
    extra = numbers[6 + 2 * projective:]
    return across, held, numbers[6:8] if projective else None, extra


def main():
    for name in SETS:
        ids, arm, seen, firmware, plane = load(name)
        print(f"\n== {name}: {len(ids)} points; rms / median / max of the misses, 'held' each point left out "
              f"(green: <= 1 mm)")
        variants = {
            "as now": arm,
            "no reach correction (6.6 - 0.118 h)": with_constants(firmware, REACH_AT_DESK_MM=0.0, REACH_LESS_PER_MM=0.0),
            "no bearing wave": with_constants(firmware, WAVE_SIN_DEG=0.0, WAVE_COS_DEG=0.0),
            "no reach correction, no wave": with_constants(firmware, REACH_AT_DESK_MM=0.0, REACH_LESS_PER_MM=0.0,
                                                           WAVE_SIN_DEG=0.0, WAVE_COS_DEG=0.0),
            "no true-z formula (h from raw z)": with_constants(firmware, TRUE_Z_PER_MM_OF_REACH=0.0,
                                                               TALL_AT_STACK_MM=0.0, TALL_PER_MM_OF_REACH=0.0),
        }
        flat_z = np.array([row for row in firmware])
        for label, xy in variants.items():
            across, held, _, _ = summary(xy, seen)
            print(f"  {label:40s} fit rms {math.sqrt(np.mean(across ** 2)):.2f} med {np.median(across):.2f} "
                  f"max {across.max():.2f} green {np.sum(across <= 1)}/{len(across)} | held rms "
                  f"{math.sqrt(np.mean(held ** 2)):.2f}")
        print("  models on the arm as now:")
        for label, terms, projective in (
                ("affine (SAVE today)", (), False),
                ("+ reach offset", ("radial",), False),
                ("+ sideways offset", ("lateral",), False),
                ("+ reach and sideways offsets", ("radial", "lateral"), False),
                ("+ bearing wave change", ("wave_sin", "wave_cos"), False),
                ("+ reach offset + wave", ("radial", "wave_sin", "wave_cos"), False),
                ("+ offsets + wave", ("radial", "lateral", "wave_sin", "wave_cos"), False),
                ("camera perspective (homography)", (), True),
                ("perspective + reach offset", ("radial",), True),
                ("perspective + offsets", ("radial", "lateral"), True)):
            across, held, persp, extra = summary(arm, seen, terms, projective)
            numbers = ", ".join(f"{term} {value:+.2f}" for term, value in zip(terms, extra))
            print(f"    {label:34s} fit rms {math.sqrt(np.mean(across ** 2)):.2f} green {np.sum(across <= 1):2d}"
                  f" | held rms {math.sqrt(np.mean(held ** 2)):.2f} green {np.sum(held <= 1):2d}  {numbers}"
                  + (f" persp {persp[0] * 1e4:+.2f} {persp[1] * 1e4:+.2f} e-4/mm" if persp is not None else ""))


if __name__ == "__main__":
    main()
