#!/usr/bin/env python3
"""SAVE's camera fit with the grid paper added: one camera through which today's Fine Tune cubes' edges lie along
those found (as SAVE fits it) and the paper's printed lines on a perfect 5 mm grid, together. The paper's snapshot
may have the camera a pixel or two from the cubes' (a free shift of its pixels). Then both days' points: their cubes
placed through it, the arm laid on them, the misses. No arm.

    .venv/bin/python impala/agenttools/cube-grid/joint-camera.py [paper-weight-px-per-mm]
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
from camera.fit import CENTRE_SPREAD, EDGE_HUBER_PX, STEPS, link_arm, stretch_of   # noqa: E402
from camera.model import CUBE_MM, least_squares                            # noqa: E402
from camera.refine import _numbers, _pinhole, refine                       # noqa: E402
from camera.sighting import in_frame, measure, misses, place, start       # noqa: E402

SQUARE_MM = 5.0
DAYS = {"today": ROOT / "config", "noon": ROOT.parent / "work" / "config-impala-1030"}


def on_plane(matrix, pixels):
    h = np.linalg.solve(matrix[:, [0, 1, 3]], np.column_stack((pixels, np.ones(len(pixels)))).T).T
    return h[:, :2] / h[:, 2:]


def lay(camera, points, frame_size):
    """Each point's cube placed through `camera`, the arm laid on them: misses (mm), heights from 40, edges (px)."""
    seen, heights, edges = [], [], []
    for point in points:
        sighting = point.sighting
        matrix = in_frame(camera, frame_size, sighting)
        first = start(matrix, sighting)
        seen.append(place(matrix, sighting, first[:3])[:2])
        free = measure(matrix, sighting, first)
        heights.append(free[3] - CUBE_MM)
        edges.append(math.sqrt(np.mean(misses(matrix, sighting, free) ** 2)))
    arm = np.array([(point.x, point.y) for point in points])
    turn, shift, desk_to_true = link_arm(arm, np.array(seen))
    back = np.linalg.solve(desk_to_true, np.linalg.solve(turn, (np.array(seen) - shift).T)).T
    return np.hypot(*(back - arm).T), np.array(heights), np.array(edges), stretch_of(desk_to_true)


def main():
    weight = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    lines = json.loads((HERE / "lines.json").read_text())
    pixels = np.array([p["pixel"] for p in lines["points"]])
    family = np.array([p["family"] for p in lines["points"]])
    centre, u, v = map(np.array, (lines["centre"], lines["u"], lines["v"]))
    sheet = np.array([p["sheet"] for p in lines["points"]])
    coord = np.where(family == 0, sheet[:, 0], sheet[:, 1])
    shift0 = np.array([np.median((coord - np.round(coord / lines["pitch"]) * lines["pitch"])[family == f])
                       for f in (0, 1)])
    number = np.round((coord - shift0[family]) / lines["pitch"])
    fit = store.load_camera(DAYS["today"])
    points = store.load_refine_points(DAYS["today"])
    frame_size = tuple(fit.frame_size)
    width = frame_size[0]
    middle = np.array(frame_size) / 2
    saved = refine(fit, [point.sighting for point in points])
    first_camera = _numbers(saved.true_to_pixel)
    cubes0 = [place(in_frame(saved.true_to_pixel, frame_size, p.sighting), p.sighting,
                    start(in_frame(saved.true_to_pixel, frame_size, p.sighting), p.sighting)[:3]) for p in points]
    count = len(points)

    def residuals(params):
        camera_numbers, paper_shift, paper_place = params[:9], params[9:11], params[11:14]
        matrix = _pinhole(camera_numbers).matrix
        parts = []
        for index, point in enumerate(points):
            cube = params[14 + 3 * index:17 + 3 * index]
            parts.append(misses(in_frame(matrix, frame_size, point.sighting), point.sighting, np.append(cube, CUBE_MM)))
        true = on_plane(matrix, pixels + paper_shift) - centre
        along = np.column_stack((true @ u, true @ v))
        cos, sin = math.cos(paper_place[2]), math.sin(paper_place[2])
        along = along @ np.array([[cos, -sin], [sin, cos]])
        mine = np.where(family == 0, along[:, 0] - paper_place[0], along[:, 1] - paper_place[1])
        parts.append(weight * (mine - number * SQUARE_MM))
        parts.append((camera_numbers[1:3] - middle) / (CENTRE_SPREAD * width))
        return np.concatenate(parts)

    # the camera moved this much from the cubes' snapshots to the paper's (phase correlation of the desk round them)
    moved = np.array([float(value) for value in sys.argv[2].split(",")]) if len(sys.argv) > 2 else np.array([-2.0, -1.2])
    params = np.concatenate((first_camera, moved, [shift0[0], shift0[1], 0.0], np.concatenate(cubes0)))
    steps = np.concatenate((STEPS[:9], [0.05, 0.05], [0.01, 0.01, 1e-5], np.tile([0.01, 0.01, 1e-5], count)))
    # the paper's place first, the camera held
    params, _, _ = least_squares(residuals, params, steps, huber=EDGE_HUBER_PX, free=[11, 12, 13], iterations=20)
    edge_count = sum(len(p.sighting.edges[2]) for p in points)
    print(f"{edge_count} cube edge points ({count} cubes), {len(pixels)} paper points weighted {weight} px per mm")
    free = [index for index in range(len(params)) if index not in (9, 10)]
    solution, res, _ = least_squares(residuals, params, steps, huber=EDGE_HUBER_PX, free=free, iterations=60)
    cube_part = res[:edge_count] if False else None
    camera = _pinhole(solution[:9]).matrix
    paper_res = res[-(len(pixels) + 2):-2] / weight
    print(f"joint camera: focal {solution[0]:.1f} px, principal point {solution[1]:.0f},{solution[2]:.0f}; the paper's "
          f"pixels shifted {solution[9]:+.2f},{solution[10]:+.2f}; the paper's lines from a perfect grid rms "
          f"{math.sqrt(np.mean(paper_res ** 2)):.3f} mm")
    position = _pinhole(solution[:9]).position
    axis = camera[2, :3] / np.linalg.norm(camera[2, :3])
    print(f"  {position[2]:.1f} mm above the desk, axis {math.degrees(math.acos(abs(axis[2]))):.2f} deg from straight "
          f"down (SAVE's: focal {first_camera[0]:.1f})")
    json.dump({"true_to_pixel": camera.tolist()}, open(HERE / f"joint-camera-{weight:g}.json", "w"))
    for day, config in DAYS.items():
        day_points = store.load_refine_points(config)
        day_fit = store.load_camera(config)
        day_saved = refine(day_fit, [point.sighting for point in day_points])
        for name, matrix in (("SAVE's camera", day_saved.true_to_pixel), ("joint camera ", camera)):
            off, heights, edges, (low, high, skew) = lay(matrix, day_points, tuple(day_fit.frame_size))
            print(f"  {day:5s} {name}: misses rms {math.sqrt(np.mean(off ** 2)):.2f} mm, median {np.median(off):.2f},"
                  f" worst {off.max():.2f}, green {np.sum(off <= 1)}/{len(off)}; tops from 40 "
                  f"{np.median(np.abs(heights)):.2f} typ, {np.abs(heights).max():.2f} worst; edges {np.median(edges):.2f}"
                  f" px; arm stretched {low:.4f}-{high:.4f}, {skew:.2f} deg")


if __name__ == "__main__":
    main()
