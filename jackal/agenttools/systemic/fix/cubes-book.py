#!/usr/bin/env python3
"""The cubes in the book frame through the camera from the sheets alone (book-camera.py): each cube found and its
edges followed as the app does (`locate._fit_outline`, `_fit_edges`), standing on the desk or on the book (40 mm);
then, from its found edges, how tall it is and where its base is; and where the found edges sit off a 40 mm cube, by
kind of edge (+ outside the cube drawn). The book's height moves the camera, so each at 39.8, 40.0 and 40.2 mm.

    cubes-book.py
"""
import math
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
JACKAL = HERE.parents[2]
sys.path[:0] = [str(JACKAL / "src"), str(JACKAL / "src" / "calibration")]
from applib.camera import cubes, edges, locate  # noqa: E402
from applib.camera.model import camera_position, cube_corners, least_squares, project, visible_edges  # noqa: E402

frame = cv2.imread(str(HERE / "out" / "book.png"))
state = pickle.load(open("/tmp/book-camera.pkl", "rb"))
KINDS = ("top outline", "upright outline", "bottom outline", "top inner", "upright inner")


def matrix(cam):
    K = np.array([[cam[0], 0, cam[1]], [0, cam[0], cam[2]], [0, 0, 1.0]])
    return K @ np.c_[cv2.Rodrigues(np.array(cam[3:6], float))[0], cam[6:9]]


def lifted(T, z):
    M = T.copy()
    M[:, 3] = T[:, 3] + z * T[:, 2]
    return M


def corners(M, c, turn, top=40.0, base=0.0):
    c3 = cube_corners(c, turn, top)
    c3[:, 2] += base
    return project(M, c3)


def by_kind(M, params, found):
    starts, ends, targets = found
    pix = corners(M, params[:2], params[2])
    middle = project(M, np.array([[params[0], params[1], 20.0]]))[0]
    outline = {tuple(sorted((e.start, e.end))): e.is_outline
               for e in visible_edges(params[:2], params[2], camera_position(M))}
    rows = {k: [] for k in KINDS}
    for a, b, f in zip(starts, ends, targets):
        key = tuple(sorted((int(a), int(b))))
        if key not in outline:
            continue
        side = pix[b] - pix[a]
        normal = np.array([-side[1], side[0]]) / np.linalg.norm(side)
        if (middle - pix[a]) @ normal > 0:
            normal = -normal
        lo, hi = key
        kind = ("top" if lo >= 4 else "bottom" if hi < 4 else "upright") + (" outline" if outline[key] else " inner")
        rows[kind].append((f - pix[a]) @ normal)
    return rows


def main():
    blobs = cubes.find_blobs(frame)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    print("blobs:")
    for i, b in enumerate(blobs):
        m = b.mask > 0
        a, bb = lab[..., 1][m].mean() - 128, lab[..., 2][m].mean() - 128
        print(f"  {i}: centroid ({b.centroid[0]:.0f}, {b.centroid[1]:.0f}), area {b.area:.0f}, "
              f"hue {math.degrees(math.atan2(bb, a)):+.0f}, solidity {b.solidity:.2f}")
    chosen = {}
    for b in blobs:
        x, y = b.centroid
        if 1500 < x < 2100 and 150 < y < 650:
            chosen.setdefault("blue, on the book", (b, 40.0))
        if 1950 < x < 2250 and 600 < y < 950:
            chosen.setdefault("red, on the desk", (b, 0.0))
    for h in (40.0, 39.8, 40.2):
        T = matrix(state["fits"][(h, False, False)][:9])
        print(f"\nbook at {h:.1f} mm: camera f {state['fits'][(h, False, False)][0]:.1f}")
        for name, (blob, z) in chosen.items():
            M = lifted(T, h if z else 0.0)
            params = locate._fit_outline(M, blob, locate._start(M, blob))
            params, distances, found = locate._fit_edges(frame, M, blob, params)
            starts, ends, targets = found

            def miss(p):
                return edges.signed_distances(corners(M, p[:2], p[2], p[3], p[4]), starts, ends, targets)

            tall = least_squares(lambda p: miss(np.r_[p, 0.0]), np.r_[params, 40.0], np.array([0.01, 0.01, 1e-5, 0.01]),
                                 huber=1.5, iterations=40)[0]
            both = least_squares(miss, np.r_[params, 40.0, 0.0], np.array([0.01, 0.01, 1e-5, 0.01, 0.01]),
                                 huber=1.5, iterations=40)[0]
            rows = by_kind(M, params, found)
            print(f"  {name}: {len(targets)} edge points, {np.median(np.abs(distances)):.2f} px typ as a 40 mm cube; "
                  f"height from its edges {tall[3]:.2f} mm (base held); base {both[4]:+.2f} mm, top "
                  f"{both[3] + both[4]:.2f} mm (both free)")
            print("     " + " | ".join(f"{k} {np.mean(v):+.2f} ({len(v)})" for k, v in rows.items() if v))


if __name__ == "__main__":
    main()
