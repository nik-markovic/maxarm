#!/usr/bin/env python3
"""Are the cubes' edges found where the cube's are? Through SAVE's camera (fitted to every point's edges, 40 mm
cubes), each point's cube placed by its edges; every edge point found, signed outward from the cube's middle in
the picture (+ the found edge lies outside the drawn cube), by kind of edge: the top's far outline, the uprights'
outline, the bottom's near outline, and the inner edges (top to side, side to side). Pixels, and mm at the cube.

    edge-bias.py [config_dir]
"""
import math
import sys
from pathlib import Path

import numpy as np

JACKAL = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(JACKAL / "src"), str(JACKAL / "src" / "calibration")]
from applib.camera import store  # noqa: E402
from applib.camera.model import camera_position, cube_corners, project, visible_edges  # noqa: E402
from applib.camera.sighting import in_frame, place, start  # noqa: E402
from camera import refine  # noqa: E402

config = Path(sys.argv[1]) if len(sys.argv) > 1 else JACKAL.parent / "work" / "config-jackal-1104"
fit = store.load_camera(config)
stored = [p for p in store.load_refine_points(config, fit) if p.edges]
points = [p.sighting for p in stored]
matrix = refine._camera(fit.cube_fit(), points)
lens = camera_position(matrix)
focal = refine._numbers(matrix)[0]
KINDS = ("top outline", "upright outline", "bottom outline", "top inner", "upright inner")


def kind(a, b, is_outline):
    lo, hi = sorted((a, b))
    where = "top" if lo >= 4 else "bottom" if hi < 4 else "upright"
    return f"{where} {'outline' if is_outline else 'inner'}"


print(f"camera: lens {np.round(lens, 1)}, focal {focal:.0f}")
print(" id  dist  px/mm |" + "|".join(f"{k:>16s}" for k in KINDS) + "|  (mean signed px, + outside; n)")
totals = {k: [] for k in KINDS}
for p, s in zip(stored, points):
    m = in_frame(matrix, fit.frame_size, s)
    cube = place(m, s, start(m, s)[:3])
    corners3 = cube_corners(cube[:2], cube[2])
    corners = project(m, corners3)
    middle = project(m, np.array([[cube[0], cube[1], 20.0]]))[0]
    outline = {tuple(sorted((e.start, e.end))): e.is_outline for e in visible_edges(cube[:2], cube[2], lens)}
    starts, ends, found = s.edges
    distance = float(np.linalg.norm(lens - np.array([cube[0], cube[1], 20.0])))
    px_per_mm = focal / distance
    by = {k: [] for k in KINDS}
    for a, b, q in zip(starts, ends, found):
        key = tuple(sorted((int(a), int(b))))
        if key not in outline:
            continue
        side = corners[b] - corners[a]
        normal = np.array([-side[1], side[0]]) / np.linalg.norm(side)
        if (middle - corners[a]) @ normal > 0:
            normal = -normal
        d = (q - corners[a]) @ normal
        if abs(d) < 4:
            by[kind(a, b, outline[key])].append(d)
    cells = []
    for k in KINDS:
        v = by[k]
        totals[k] += v
        cells.append(f"{np.mean(v):+6.2f} ({len(v):3d})" if v else f"{'':>16s}")
    print(f"{p.id:3d} {distance:5.0f} {px_per_mm:6.2f} |" + "|".join(f"{c:>16s}" for c in cells) + "|")
print("all          |" + "|".join(f"{np.mean(v):+6.2f} ({len(v):3d})" for v in totals.values()) + "|")
