#!/usr/bin/env python3
"""Both days' Fine Tune points at once: each day its own SAVE (any affine map), a model's numbers shared. Is the
leftover pattern the arm's (by its bearing and reach) or the camera's (by where in the frame)? Judged leaving each
point out in turn. Reads twist-now.json and twist-old1030.json (residuals.py).

    .venv/bin/python impala/agenttools/twist/joint.py
"""

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name):
    data = json.loads((HERE / f"twist-{name}.json").read_text())
    rows = data["rows"]
    arm = np.array([row["arm"] for row in rows])
    seen = np.array([row["seen"] for row in rows])
    pixel = np.array([row["pixel"] for row in rows])
    camera = np.array(data["true_to_pixel"])
    to_true = np.array(data["desk_to_true"])
    return {"ids": [row["id"] for row in rows], "arm": arm, "seen": seen, "pixel": pixel, "camera": camera,
            "to_true": to_true}


def desk_of_pixel(day, pixel):
    """Desk x, y (the arm's, through the day's SAVE) of the desk at a pixel."""
    homography = day["camera"][:, [0, 1, 3]]
    x, y, w = np.linalg.solve(homography, (*pixel, 1.0))
    return np.linalg.solve(day["to_true"], (x / w, y / w))


def fields(day, terms):
    arm = day["arm"]
    reach = np.hypot(arm[:, 0], arm[:, 1])
    bearing = np.arctan2(arm[:, 0], -arm[:, 1])
    out = np.column_stack((np.sin(bearing), -np.cos(bearing)))
    round_ = np.column_stack((np.cos(bearing), np.sin(bearing)))
    result = []
    for term in terms:
        kind, _, k = term.partition(":")
        if kind in ("tan_sin", "tan_cos", "rad_sin", "rad_cos"):
            angle = (np.sin if kind.endswith("sin") else np.cos)(int(k) * bearing)
            result.append((round_ * reach[:, None] * math.pi / 180 if kind.startswith("tan") else out) * angle[:, None])
        elif kind == "radial":
            result.append(out)
        elif kind == "lateral":
            result.append(round_)
        elif kind in ("lens2", "lens4"):
            # the desk moved as a pixel is, by a radial lens bend of 1 (rho^2 or rho^4), rho from the frame's middle
            # in units of the focal length (~1670 px)
            moved = []
            for pixel in day["pixel"]:
                offset = pixel - np.array([1152.0, 768.0])
                rho2 = (offset @ offset) / 1670.0 ** 2
                bend = rho2 if kind == "lens2" else rho2 ** 2
                moved.append(desk_of_pixel(day, pixel + offset * bend * 0.01) - desk_of_pixel(day, pixel))
            result.append(np.array(moved) * 100)
        else:
            raise ValueError(term)
    return result


def design(days, terms):
    blocks, targets = [], []
    for index, day in enumerate(days):
        n = len(day["arm"])
        rows = np.zeros((2 * n, 6 * len(days)))
        for axis in range(2):
            rows[axis * n:(axis + 1) * n, 6 * index + 3 * axis:6 * index + 3 * axis + 3] = \
                np.column_stack((day["seen"], np.ones(n)))
        extra = [-np.concatenate((field[:, 0], field[:, 1]))[:, None] for field in fields(day, terms)]
        blocks.append(np.hstack([rows, *extra]) if extra else rows)
        targets.append(np.concatenate((day["arm"][:, 0], day["arm"][:, 1])))
    return np.vstack(blocks), np.concatenate(targets)


def run(days, terms):
    rows, target = design(days, terms)
    numbers = np.linalg.lstsq(rows, target, rcond=None)[0]
    fitted = target - rows @ numbers
    total = sum(len(day["arm"]) for day in days)
    # pair each point's x and y rows
    starts = np.cumsum([0] + [len(day["arm"]) for day in days])
    pairs = []
    for index, day in enumerate(days):
        n = len(day["arm"])
        base = 2 * starts[index]
        for i in range(n):
            pairs.append((base + i, base + n + i))
    across = np.array([math.hypot(fitted[a], fitted[b]) for a, b in pairs])
    held = []
    for a, b in pairs:
        keep = np.ones(len(target), bool)
        keep[[a, b]] = False
        trial = np.linalg.lstsq(rows[keep], target[keep], rcond=None)[0]
        miss = target[[a, b]] - rows[[a, b]] @ trial
        held.append(math.hypot(*miss))
    held = np.array(held)
    return numbers[6 * len(days):], across, held, total


def main():
    days = [load("now"), load("old1030")]
    print("both days together; rms of the misses fitted / each point left out; green = <= 1 mm (of 38)")
    for label, terms in (
            ("SAVE today (affine per day)", ()),
            ("arm: wave 2b (as raw.py, changed)", ("tan_sin:2", "tan_cos:2")),
            ("arm: wave 2b + 3b", ("tan_sin:2", "tan_cos:2", "tan_sin:3", "tan_cos:3")),
            ("arm: wave 2b + reach offset", ("tan_sin:2", "tan_cos:2", "radial")),
            ("arm: reach + sideways offsets", ("radial", "lateral")),
            ("arm: wave 2b,3b + reach offset", ("tan_sin:2", "tan_cos:2", "tan_sin:3", "tan_cos:3", "radial")),
            ("arm: reach error by bearing 2b", ("rad_sin:2", "rad_cos:2")),
            ("camera: lens bend rho^2", ("lens2",)),
            ("camera: lens bend rho^2 + rho^4", ("lens2", "lens4")),
            ("camera lens + arm wave 2b", ("lens2", "tan_sin:2", "tan_cos:2"))):
        numbers, across, held, total = run(days, terms)
        print(f"  {label:36s} fit {math.sqrt(np.mean(across ** 2)):.2f} green {np.sum(across <= 1):2d} | held "
              f"{math.sqrt(np.mean(held ** 2)):.2f} green {np.sum(held <= 1):2d} | "
              + ", ".join(f"{term} {value:+.3f}" for term, value in zip(terms, numbers)))
    for name, day in (("now", days[0]), ("old1030", days[1])):
        numbers, across, held, total = run([day], ("tan_sin:2", "tan_cos:2"))
        print(f"  {name} alone, wave 2b: {', '.join(f'{v:+.2f}' for v in numbers)}")


if __name__ == "__main__":
    main()
