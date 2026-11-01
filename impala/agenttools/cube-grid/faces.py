#!/usr/bin/env python3
"""Each cube's faces measured apart, through camera.json: the top square on the plane 40 mm up (its size free), the
cube's height, where its bottom sits; and where each way of fitting puts the cube. No arm.

    .venv/bin/python impala/agenttools/cube-grid/faces.py frame.png [green box x0,y0,x1,y1]

The cubes' edges are found as FIND RED CUBE finds them (`locate.py`), the green cube's blob cut out by its colour in
the box given when the cube finder misses it. Fits, each through the camera, Huber 1 px, on the edges' points:
  box     the 40 mm cube standing on the desk, all its seen edges (as CALIBRATE, SAVE and FIND RED CUBE fit it)
  tall    the same, its height free (Fine Tune's Z)
  top     the top square alone, 40 mm up, 40 mm wide: where the top is
  top s   the same, its width free: how wide the top looks on the plane 40 mm up
  lifted  the 40 mm cube, its bottom free in height: where it stands, by all its edges
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera import cubes, edges, locate, store                              # noqa: E402
from camera.fit import EDGE_HUBER_PX                                       # noqa: E402
from camera.model import CUBE_MM, _SQUARE, least_squares, project          # noqa: E402


def corners_of(x, y, turn, side=CUBE_MM, bottom=0.0, top=None):
    top = bottom + CUBE_MM if top is None else top
    c, s = math.cos(turn), math.sin(turn)
    return np.array([(x + side * (c * dx - s * dy), y + side * (s * dx + c * dy), z)
                     for z in (bottom, top) for dx, dy in _SQUARE])


def fits(matrix, found, start):
    """Each way of fitting, from `start` (x, y, turn): its numbers and the edges' misses, px."""
    starts, ends, targets = found
    top = (starts >= 4) & (ends >= 4)
    bottom = (starts < 4) & (ends < 4)

    def misses(corners, keep=slice(None)):
        return edges.signed_distances(project(matrix, corners), starts[keep], ends[keep], targets[keep])

    ways = {
        "box": (lambda p: misses(corners_of(*p)), list(start)),
        "tall": (lambda p: misses(corners_of(p[0], p[1], p[2], top=p[3])), list(start) + [CUBE_MM]),
        "top": (lambda p: misses(corners_of(*p, bottom=0.0), top), list(start)),
        "top s": (lambda p: misses(corners_of(p[0], p[1], p[2], side=p[3]), top), list(start) + [CUBE_MM]),
        "lifted": (lambda p: misses(corners_of(p[0], p[1], p[2], bottom=p[3])), list(start) + [0.0]),
    }
    result = {}
    for name, (residuals, first) in ways.items():
        steps = np.array([0.01, 0.01, 1e-5, 0.01][:len(first)])
        numbers, left, _ = least_squares(residuals, np.array(first, float), steps, huber=EDGE_HUBER_PX, iterations=40)
        result[name] = (numbers, math.sqrt(np.mean(left ** 2)), int(top.sum()), int(bottom.sum()), len(targets))
    return result


def green_blob(frame, box):
    x0, y0, x1, y1 = box
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2], np.uint8)
    part = hsv[y0:y1, x0:x1]
    mask[y0:y1, x0:x1] = ((part[..., 0] > 65) & (part[..., 0] < 100) & (part[..., 1] > 60) & (part[..., 2] > 25)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    chroma = cubes.chroma_from_desk(frame, cubes.desk_chroma(frame))
    return cubes._blob((labels == biggest).astype(np.uint8) * 255, chroma)


def main():
    frame = cv2.imread(str(HERE / sys.argv[1]))
    fit = store.load_camera(ROOT / "config")
    matrix = fit.true_to_pixel
    blobs = {("red" if cubes.is_warm(blob) else "blue"): blob for blob in cubes.find_blobs(frame)}
    if len(sys.argv) > 2:
        blobs["green"] = green_blob(frame, [int(v) for v in sys.argv[2].split(",")])
    print(f"{sys.argv[1]}: each cube, true mm; 'moves' is how far a way puts the cube from 'box'")
    for name, blob in blobs.items():
        overlap, params, distances, found = locate._fit(frame, matrix, blob)
        result = fits(matrix, found, params)
        box = result["box"][0]
        n_top, n_bottom, n_all = result["box"][2:]
        cells = [f"{name:5s} at ({box[0]:6.1f},{box[1]:6.1f}), edges {result['box'][1]:.2f} px "
                 f"({n_all} points: top {n_top}, bottom {n_bottom})"]
        cells.append(f"tall {result['tall'][0][3]:.2f} mm")
        cells.append(f"top looks {result['top s'][0][3]:.2f} mm wide 40 mm up")
        cells.append(f"top alone moves {math.dist(result['top'][0][:2], box[:2]):.2f} mm "
                     f"(edges {result['top'][1]:.2f} px)")
        cells.append(f"lifted: bottom at {result['lifted'][0][3]:+.2f} mm, moves "
                     f"{math.dist(result['lifted'][0][:2], box[:2]):.2f}")
        print("  " + "; ".join(cells))


if __name__ == "__main__":
    main()
