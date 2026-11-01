#!/usr/bin/env python3
"""A camera fitted to the grid paper alone, against SAVE's (fitted to the Fine Tune cubes' outlines): does the paper's
camera lay the arm on the Fine Tune points better? No arm.

    .venv/bin/python impala/agenttools/cube-grid/paper.py frame.png true_to_pixel     # writes lines.json first
    .venv/bin/python impala/agenttools/cube-grid/paper-camera.py

The paper camera: focal length, principal point (held loosely to the frame's middle, as SAVE holds it), rotation and
shift, so that every line point found lies on its line of a perfect 5 mm grid, the grid's place and turn free. Then
each day's Fine Tune points: their cubes placed (40 mm, from their edges) through each camera, the arm laid on them
(turn, shift, stretch), and the misses.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera import store                                                   # noqa: E402
from camera.fit import CENTRE_SPREAD, link_arm, stretch_of                 # noqa: E402
from camera.model import least_squares                                     # noqa: E402
from camera.refine import _numbers, _pinhole, refine                       # noqa: E402
from camera.sighting import in_frame, measure, misses, place, start       # noqa: E402

SQUARE_MM = 5.0
DAYS = {"today": ROOT / "config", "noon": ROOT.parent / "work" / "config-impala-1030"}


def on_plane(matrix, pixels):
    plane = matrix[:, [0, 1, 3]]
    h = np.linalg.solve(plane, np.column_stack((pixels, np.ones(len(pixels)))).T).T
    return h[:, :2] / h[:, 2:]


def describe(name, numbers):
    camera = _pinhole(numbers)
    axis = camera.matrix[2, :3] / np.linalg.norm(camera.matrix[2, :3])
    print(f"  {name}: focal {numbers[0]:.1f} px, principal point {numbers[1]:.0f},{numbers[2]:.0f}, "
          f"{camera.position[2]:.1f} mm above the desk, axis {math.degrees(math.acos(abs(axis[2]))):.2f} deg from "
          f"straight down")


def main():
    lines = json.loads((HERE / "lines.json").read_text())
    pixels = np.array([p["pixel"] for p in lines["points"]])
    family = np.array([p["family"] for p in lines["points"]])
    centre, u, v = map(np.array, (lines["centre"], lines["u"], lines["v"]))
    sheet = np.array([p["sheet"] for p in lines["points"]])
    pitch = lines["pitch"]
    coord = np.where(family == 0, sheet[:, 0], sheet[:, 1])
    shift0 = np.array([np.median((coord - np.round(coord / pitch) * pitch)[family == f]) for f in (0, 1)])
    number = np.round((coord - shift0[family]) / pitch)
    saved = store.load_camera(ROOT / "config")
    width, height = saved.frame_size
    middle = np.array([width / 2, height / 2])
    first = _numbers(saved.true_to_pixel)

    def grid_misses(params):
        numbers, (s0, s1, turn) = params[:9], params[9:]
        true = on_plane(_pinhole(numbers).matrix, pixels) - centre
        along = np.column_stack((true @ u, true @ v))
        cos, sin = math.cos(turn), math.sin(turn)
        along = along @ np.array([[cos, -sin], [sin, cos]])
        mine = np.where(family == 0, along[:, 0] - s0, along[:, 1] - s1)
        return np.concatenate((mine - number * SQUARE_MM, (numbers[1:3] - middle) / (CENTRE_SPREAD * width)))

    start_params = np.concatenate((first, [shift0[0], shift0[1], 0.0]))
    steps = np.concatenate((np.array([0.01, 0.01, 0.01, 1e-5, 1e-5, 1e-5, 0.01, 0.01, 0.01]), [0.01, 0.01, 1e-5]))
    held = least_squares(grid_misses, start_params, steps[[0] * 0 + list(range(12))], huber=0.3, iterations=1)[0]
    # SAVE's camera with only the grid's place free: how the paper sits through it
    fixed = lambda place_params: grid_misses(np.concatenate((first, place_params)))[:-2]
    place_only = least_squares(fixed, start_params[9:], steps[9:], huber=0.3, iterations=50)[0]
    before = fixed(place_only)
    solution, residuals, _ = least_squares(grid_misses, start_params, steps, huber=0.3, iterations=200)
    after = residuals[:-2]
    print(f"the paper's {len(pixels)} line points from a perfect {SQUARE_MM} mm grid: through SAVE's camera rms "
          f"{math.sqrt(np.mean(before ** 2)):.3f} mm (worst {np.abs(before).max():.2f}); through the paper's own "
          f"camera {math.sqrt(np.mean(after ** 2)):.3f} (worst {np.abs(after).max():.2f})")
    for label, values in (("SAVE's camera", before), ("paper camera", after)):
        cells = []
        for lo, hi in ((0, 600), (600, 1100), (1100, 1700), (1700, 2304)):
            sel = (pixels[:, 0] >= lo) & (pixels[:, 0] < hi)
            cells.append(f"u {lo}-{hi}: {math.sqrt(np.mean(values[sel] ** 2)):.2f} ({sel.sum()})" if sel.sum() else f"u {lo}-{hi}: -")
        print(f"  {label} by frame column, rms mm (points): " + "; ".join(cells))
    describe("SAVE's camera (cubes)", first)
    describe("paper camera         ", solution[:9])
    paper_camera = _pinhole(solution[:9]).matrix
    json.dump({"true_to_pixel": paper_camera.tolist()}, open(HERE / "paper-camera.json", "w"))
    per_point = {}
    for day, config in DAYS.items():
        fit = store.load_camera(config)
        points = store.load_refine_points(config)
        saved_day = refine(fit, [point.sighting for point in points])
        for name, camera in (("SAVE's camera", saved_day.true_to_pixel), ("paper camera ", paper_camera)):
            seen, heights, edge = [], [], []
            for point in points:
                sighting = point.sighting
                matrix = in_frame(camera, tuple(fit.frame_size), sighting)
                first_cube = start(matrix, sighting)
                cube = place(matrix, sighting, first_cube[:3])
                seen.append(cube[:2])
                free = measure(matrix, sighting, first_cube)
                heights.append(free[3] - 40.0)
                edge.append(math.sqrt(np.mean(misses(matrix, sighting, free) ** 2)))
            arm = np.array([(point.x, point.y) for point in points])
            turn, shift, desk_to_true = link_arm(arm, np.array(seen))
            back = np.linalg.solve(desk_to_true, np.linalg.solve(turn, (np.array(seen) - shift).T)).T
            off = np.hypot(*(back - arm).T)
            low, high, skew = stretch_of(desk_to_true)
            per_point.setdefault(day, []).append({point.id: (round(float(o), 2), point.sighting.pixel[0]) for point, o in zip(points, off)})
            print(f"  {day:5s} {name}: misses rms {math.sqrt(np.mean(off ** 2)):.2f} mm, median {np.median(off):.2f}, "
                  f"worst {off.max():.2f}, green {np.sum(off <= 1)}/{len(off)}; cubes' tops from 40 "
                  f"{np.median(np.abs(heights)):.2f} typ, {np.max(np.abs(heights)):.2f} worst; edges "
                  f"{np.median(edge):.2f} px; the arm stretched {low:.4f}-{high:.4f}, {skew:.2f} deg")


if __name__ == "__main__":
    main()


def compare():
    pass
