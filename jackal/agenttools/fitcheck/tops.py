#!/usr/bin/env python3
"""Model-free check of the points: each cube's top face from its found edges -- four lines, their corners, the
diagonals' crossing -- against the arm's reading, through one plane-to-image map (a homography), no lens, no focal
length, no principal point, no 3-D cube. A pinhole camera without distortion maps the plane 40 mm up (the cubes'
tops) to the picture by exactly such a map; so if the arm's true x, y are real and each cube is centred under the
cup, one homography fits every top centre to the found edges' precision.

    PYTHONPATH=src:src/calibration ../.venv/bin/python3 agenttools/fitcheck/tops.py [refine-points.json]
"""
import itertools, json, math, sys
from pathlib import Path

import numpy as np

from applib.camera.fit import HANDEDNESS
from applib.camera.model import least_squares

POINTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/camera-refine-points.json")
TOP = ((4, 5), (5, 6), (6, 7), (7, 4))


def line(points):
    """Total least squares: a point on the line, its direction, and the points' rms distance from it."""
    middle = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - middle)
    direction = vt[0]
    normal = np.array((-direction[1], direction[0]))
    return middle, direction, float(np.sqrt(np.mean(((points - middle) @ normal) ** 2)))


def crossing(a, b):
    (p, d), (q, e) = a, b
    t = np.linalg.solve(np.column_stack((d, -e)), q - p)[0]
    return p + t * d


def top_face(point):
    """The top's four corners (4, 5, 6, 7), pixels: each where the found lines through it cross, least squares (two
    top edges and the upright, as found); and how straight each edge's points lie."""
    lines, spread = {}, {}
    for entry in point["edges"]:
        a, b = sorted(entry["corners"])
        found = np.array(entry["found"], float).reshape(-1, 2)
        middle, direction, rms = line(found)
        lines[(a, b)] = (middle, direction)
        spread[(a, b)] = (len(found), rms)
    corners = {}
    for corner in (4, 5, 6, 7):
        through = [lines[e] for e in lines if corner in e]
        normals = np.array([(-d[1], d[0]) for _, d in through])
        offsets = np.array([n @ m for n, (m, _) in zip(normals, through)])
        corners[corner] = np.linalg.lstsq(normals, offsets, rcond=None)[0]
    return corners, spread


def centre_of(corners):
    """The diagonals' crossing: the top's centre, whatever the perspective."""
    d1 = (corners[4], corners[6] - corners[4])
    d2 = (corners[5], corners[7] - corners[5])
    return crossing(d1, d2)


def apply(h, xy):
    v = np.column_stack((xy, np.ones(len(xy)))) @ h.T
    return v[:, :2] / v[:, 2:]


def homography(src, dst):
    """src -> dst, least squares in dst (pixels): DLT, then refined."""
    def normaliser(p):
        m, s = p.mean(axis=0), np.sqrt(2) / np.mean(np.linalg.norm(p - p.mean(axis=0), axis=1))
        return np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])
    ts, td = normaliser(src), normaliser(dst)
    s, d = apply(ts, src), apply(td, dst)
    rows = []
    for (x, y), (u, v) in zip(s, d):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    h = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    h = np.linalg.inv(td) @ h @ ts
    h = h / h[2, 2]
    if len(src) > 4:
        start = h.ravel()[:8]
        steps = np.maximum(np.abs(start), 1e-6) * 1e-6
        h = least_squares(lambda p: (apply(np.append(p, 1).reshape(3, 3), src) - dst).ravel(), start, steps,
                          huber=1e9, iterations=200)[0]
        h = np.append(h, 1).reshape(3, 3)
    return h


def mm_off(h, pixel, reading):
    """Where the homography puts the pixel on the plane, against the reading, desk mm (mirror undone)."""
    seen = apply(np.linalg.inv(h), pixel[None])[0]
    return HANDEDNESS @ (seen - reading)


def main():
    stored = json.loads(POINTS.read_text())["points"]
    readings, centres = [], []
    print(f"{POINTS}: the top faces from the found edges")
    for p in stored:
        corners, spread = top_face(p)
        c = centre_of(corners)
        readings.append(HANDEDNESS @ np.array((p["x"], p["y"])))
        centres.append(c)
        sides = [np.linalg.norm(corners[b] - corners[a]) for a, b in TOP]
        print(f"  #{p['id']} {p.get('cube') or 'red':5} {p['taken']:15} top centre ({c[0]:7.1f}, {c[1]:6.1f})  "
              f"sides {' '.join(f'{s:5.1f}' for s in sides)} px; edge points (count, rms px) "
              + " ".join(f"{n}/{r:.2f}" for n, r in spread.values()))
    readings, centres = np.array(readings), np.array(centres)
    n = len(readings)

    h = homography(readings, centres)
    print(f"\nOne homography through all {n} top centres (8 numbers, {2 * n} equations):")
    for k in range(n):
        px = apply(h, readings[k:k + 1])[0] - centres[k]
        d = mm_off(h, centres[k], readings[k])
        print(f"  #{stored[k]['id']}: {np.linalg.norm(px):5.2f} px, {np.linalg.norm(d):5.2f} mm (dx {d[0]:+5.2f}, "
              f"dy {d[1]:+5.2f})")

    if n >= 5:
        print(f"\nEach point left out, the homography of the other {n - 1} predicting it:")
        for k in range(n):
            keep = [j for j in range(n) if j != k]
            hk = homography(readings[keep], centres[keep])
            d = mm_off(hk, centres[k], readings[k])
            print(f"  #{stored[k]['id']}: {np.linalg.norm(d):5.2f} mm (dx {d[0]:+5.2f}, dy {d[1]:+5.2f})")

    print("\nEvery four points (a homography exactly through them) predicting the rest, mm:")
    worst = []
    for four in itertools.combinations(range(n), 4):
        try:
            hk = homography(readings[list(four)], centres[list(four)])
        except np.linalg.LinAlgError:
            continue
        rest = {k: float(np.linalg.norm(mm_off(hk, centres[k], readings[k]))) for k in range(n) if k not in four}
        worst += list(rest.values())
        print(f"  {'+'.join(str(stored[k]['id']) for k in four):8} -> "
              + "  ".join(f"#{stored[k]['id']} {v:5.2f}" for k, v in rest.items()))
    print(f"  median {np.median(worst):.2f}, worst {max(worst):.2f}")


if __name__ == "__main__":
    main()
