#!/usr/bin/env python3
"""5 Nov night, the owner: the edges not facing the camera bleed out, the ones on the desk most. Each point's found
edge points against its 40 mm cube placed by them through CALIBRATE's camera, by kind of edge: outline on the desk
(the base), outline at the top, outline upright, and the inner edges (between two faces seen), top and upright.
Outward + (away from the cube for an outline; for an inner edge, away from the top face or from the cube's middle).
Then the cubes placed again without the base outline, and the pairs at one place compared.

    nov5-bleed.py config_dir
"""
import sys
from pathlib import Path

import numpy as np

from applib.camera import edges, store
from applib.camera.fit import EDGE_HUBER_PX, HANDEDNESS
from applib.camera.model import CUBE_MM, cube_corners, least_squares, project
from applib.camera.sighting import Sighting, in_frame, place, start

CONFIG = Path(sys.argv[1])
fit = store.load_camera(CONFIG)
camera = fit.cube_fit().true_to_pixel
KINDS = ("base outline", "top outline", "upright outline", "top inner", "upright inner")


def kinds_and_outward(matrix, s, numbers):
    starts, ends, found = s.edges
    corners = project(matrix, cube_corners(numbers[:2], numbers[2]))
    middle = corners.mean(axis=0)
    top_middle = corners[4:].mean(axis=0)
    hull = set()
    import cv2
    for i in cv2.convexHull(corners.astype(np.float32), returnPoints=False).ravel():
        hull.add(int(i))
    side = corners[ends] - corners[starts]
    normal = np.column_stack((-side[:, 1], side[:, 0])) / np.linalg.norm(side, axis=1)[:, None]
    signed = np.sum((found - corners[starts]) * normal, axis=1)
    mid = (corners[starts] + corners[ends]) / 2
    kinds, outward = [], []
    for k, (a, b) in enumerate(zip(starts, ends)):
        outline = a in hull and b in hull and not (a < 4 and b >= 4 and False)
        is_base, is_top = a < 4 and b < 4, a >= 4 and b >= 4
        level = "base" if is_base else "top" if is_top else "upright"
        # an outline edge has the cube on one side only: its middle's side away from the cube's middle
        away = mid[k] - (top_middle if (is_top and not outline) else middle)
        sign = 1.0 if normal[k] @ away > 0 else -1.0
        is_outline = outline and _on_outline(corners, a, b)
        kinds.append(f"{level} {'outline' if is_outline else 'inner'}")
        outward.append(sign * signed[k])
    return np.array(kinds), np.array(outward)


def _on_outline(corners, a, b):
    """Every other corner on one side of the line a-b: the edge is on the silhouette."""
    side = corners[b] - corners[a]
    normal = np.array([-side[1], side[0]])
    others = [(corners[k] - corners[a]) @ normal for k in range(8) if k not in (a, b)]
    return all(o >= -1e-6 for o in others) or all(o <= 1e-6 for o in others)


def px_per_mm(matrix, numbers):
    c = np.array([[numbers[0], numbers[1], CUBE_MM / 2], [numbers[0] + 1, numbers[1], CUBE_MM / 2],
                  [numbers[0], numbers[1] + 1, CUBE_MM / 2]])
    p = project(matrix, c)
    return float((np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[0])) / 2)


points = [p for p in store.load_refine_points(CONFIG, fit) if p.edges]
table, placed, placed_nobase = {k: [] for k in KINDS}, {}, {}
for point in points:
    s = point.sighting
    matrix = in_frame(camera, fit.frame_size, s)
    numbers = place(matrix, s, start(matrix, s)[:3])
    kinds, outward = kinds_and_outward(matrix, s, numbers)
    scale = px_per_mm(matrix, numbers)
    row = []
    for k in KINDS:
        sel = kinds == k
        if sel.any():
            table[k].append(np.mean(outward[sel]) / scale)
            row.append(f"{k} {np.mean(outward[sel]):+.1f} px ({sel.sum()})")
    print(f"#{point.id:2} {point.cube or 'red*':6} {scale:.1f} px/mm | " + "; ".join(row))
    keep = kinds != "base outline"
    starts, ends, found = s.edges
    without = Sighting(s.x, s.y, s.pixel, s.frame_size, s.turn_deg, (starts[keep], ends[keep], found[keep]))
    placed[point.id] = HANDEDNESS @ numbers[:2]
    placed_nobase[point.id] = HANDEDNESS @ place(matrix, without, numbers)[:2]
print("\nmean outward, mm, over the points:")
for k in KINDS:
    if table[k]:
        print(f"   {k:16} {np.mean(table[k]):+.2f} mm (spread {np.std(table[k]):.2f}, {len(table[k])} points)")
print("\neach cube placed without its base outline: how far it moves, and its miss against its reading")
for point in points:
    a, b = placed[point.id], placed_nobase[point.id]
    r = np.array([point.x, point.y])
    print(f"#{point.id:2} moves {np.linalg.norm(b - a):.2f} mm {np.round(b - a, 2).tolist()}; miss "
          f"{np.linalg.norm(a - r):.2f} -> {np.linalg.norm(b - r):.2f}")
by_id = {p.id: p for p in points}
for first, second in ((3, 4), (2, 5), (1, 6)):
    if first in by_id and second in by_id:
        r = np.array([by_id[second].x - by_id[first].x, by_id[second].y - by_id[first].y])
        for name, cubes in (("all edges", placed), ("no base outline", placed_nobase)):
            d = cubes[second] - cubes[first] - r
            print(f"#{first} vs #{second}, {name}: disagree {np.linalg.norm(d):.2f} mm")

# the pairs again, each cube placed with its size free (one scale for all its sides), and from the inner edges and
# the base alone (no top or upright outline)
def corners_sized(matrix, n):
    c = cube_corners(n[:2], n[2])
    middle = np.array([n[0], n[1], 0.0])
    return project(matrix, middle + (c - middle) * n[3])


sized, inner = {}, {}
for point in points:
    s = point.sighting
    matrix = in_frame(camera, fit.frame_size, s)
    starts, ends, found = s.edges
    first = place(matrix, s, start(matrix, s)[:3])
    n = least_squares(lambda t: edges.signed_distances(corners_sized(matrix, t), starts, ends, found),
                      np.append(first, 1.0), np.array([0.01, 0.01, 1e-5, 1e-5]), huber=EDGE_HUBER_PX,
                      iterations=40)[0]
    sized[point.id] = HANDEDNESS @ n[:2]
    kinds, _ = kinds_and_outward(matrix, s, first)
    keep = (kinds != "top outline") & (kinds != "upright outline")
    only = Sighting(s.x, s.y, s.pixel, s.frame_size, s.turn_deg, (starts[keep], ends[keep], found[keep]))
    inner[point.id] = HANDEDNESS @ place(matrix, only, first)[:2]
for first, second in ((3, 4), (2, 5), (1, 6)):
    if first in by_id and second in by_id:
        r = np.array([by_id[second].x - by_id[first].x, by_id[second].y - by_id[first].y])
        for name, cubes in (("size free", sized), ("inner + base only", inner)):
            d = cubes[second] - cubes[first] - r
            print(f"#{first} vs #{second}, {name}: disagree {np.linalg.norm(d):.2f} mm")
