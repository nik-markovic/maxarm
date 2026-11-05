#!/usr/bin/env python3
"""Cube faces in the book's plane, the book frame. The red cube stands on the desk beside the book, its top level with
the book's (both 40 mm, the owner), only the top showing; the blue cube stands on the book, so its base is in the
book's plane. Their edges are found as the app finds edges (`edges.search`: the strongest colour gradient across a
side), each side a straight line, the corners where the lines meet, and the corners laid in the book's plane through
the sheets' camera -- the plane the book's sheet itself measured, so neither the book's exact height nor the camera's
focal length matters much. Then each face's sides and diagonals, mm, against a 40 mm square.

    red-top.py
"""
import math
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
JACKAL = HERE.parents[2]
sys.path[:0] = [str(JACKAL / "src"), str(JACKAL / "src" / "calibration")]
from applib.camera import cubes, edges, locate  # noqa: E402
from applib.camera.model import Edge, camera_position, cube_corners, project, visible_edges  # noqa: E402

frame = cv2.imread(str(HERE / "out" / "book.png"))
fits = pickle.load(open("/tmp/book-camera.pkl", "rb"))["fits"]


def matrix(cam):
    K = np.array([[cam[0], 0, cam[1]], [0, cam[0], cam[2]], [0, 0, 1.0]])
    return K @ np.c_[cv2.Rodrigues(np.array(cam[3:6], float))[0], cam[6:9]]


def to_plane(T, z, pix):
    w = np.linalg.solve(np.c_[T[:, 0], T[:, 1], z * T[:, 2] + T[:, 3]], np.c_[pix, np.ones(len(pix))].T)
    return (w[:2] / w[2]).T


def line(points):
    c = points.mean(0)
    d = np.linalg.svd(points - c)[2][0]
    return c, d


def meet(l1, l2):
    (c1, d1), (c2, d2) = l1, l2
    t = np.linalg.solve(np.c_[d1, -d2], c2 - c1)
    return c1 + t[0] * d1


def found_sides(corners, sides, reach_px):
    box = np.r_[corners.min(0) - 40, corners.max(0) + 40].astype(int)
    contour = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]], np.int32)
    size = float(np.linalg.norm(corners.max(0) - corners.min(0)))
    blob = SimpleNamespace(contour=contour.reshape(-1, 1, 2), size=size)
    crop = edges.crop(frame, blob, cubes.desk_chroma(frame))
    return edges.search(crop, corners, sides, reach_px / size)


def square_misses(points_by_side, centre_turn, way):
    """Each side's points against a 40 mm square's side: signed, + outside; the sides round the square `way`."""
    cx, cy, turn = centre_turn
    out = []
    for k, pts in enumerate(points_by_side):
        a = turn + way * k * math.pi / 2
        n = np.array([math.cos(a), math.sin(a)])
        out.append((pts - (cx, cy)) @ n - 20.0)
    return out


def report_face(name, plane_corners, points_by_side, labels):
    sides = [np.linalg.norm(plane_corners[(k + 1) % 4] - plane_corners[k]) for k in range(4)]
    diag = [np.linalg.norm(plane_corners[2] - plane_corners[0]), np.linalg.norm(plane_corners[3] - plane_corners[1])]
    angles = []
    for k in range(4):
        u = plane_corners[(k + 1) % 4] - plane_corners[k]
        v = plane_corners[k - 1] - plane_corners[k]
        angles.append(math.degrees(math.acos(u @ v / np.linalg.norm(u) / np.linalg.norm(v))))
    print(f"  {name}: sides {', '.join(f'{s:.2f}' for s in sides)} mm; diagonals {diag[0]:.2f}, {diag[1]:.2f} "
          f"(a 40 mm square: 56.57); corners {', '.join(f'{a:.1f}' for a in angles)} deg")
    # the 40 mm square that fits the sides' points best: each side's mean offset
    centre = plane_corners.mean(0)
    mid = (plane_corners[0] + plane_corners[1]) / 2 - centre
    side = plane_corners[1] - plane_corners[0]
    out0 = mid - (mid @ side) / (side @ side) * side
    u, v = plane_corners[1] - plane_corners[0], plane_corners[2] - plane_corners[1]
    way = 1.0 if u[0] * v[1] - u[1] * v[0] > 0 else -1.0
    x = np.array([*centre, math.atan2(out0[1], out0[0])])
    for _ in range(30):
        r = np.concatenate(square_misses(points_by_side, x, way))
        J = np.column_stack([(np.concatenate(square_misses(points_by_side, x + np.eye(3)[i] * 1e-6, way)) - r) / 1e-6
                             for i in range(3)])
        x = x - np.linalg.lstsq(J, r, rcond=None)[0]
    r = square_misses(points_by_side, x, way)
    print("     each side against a 40 mm square, + outside: " +
          ", ".join(f"{lab} {np.mean(v):+.2f} mm" for lab, v in zip(labels, r)))


def red_top(T, z):
    first = np.array([(2052, 665), (2151, 719), (2139, 895), (2033, 853)], float)
    sides = [Edge(0, 1, True), Edge(1, 2, True), Edge(2, 3, False), Edge(3, 0, True)]
    corners = first
    for reach in (14.0, 6.0, 4.0):
        starts, ends, found = found_sides(corners, sides, reach)
        lines = [line(found[(starts == e.start) & (ends == e.end)]) for e in sides]
        corners = np.array([meet(lines[k - 1], lines[k]) for k in range(4)])
    by_side = [found[(starts == e.start) & (ends == e.end)] for e in sides]
    plane = to_plane(T, z, corners)
    pts = [to_plane(T, z, p) for p in by_side]
    print(f"  red top: {[len(p) for p in by_side]} edge points on its far, right, near and left sides; corners px "
          f"{np.round(corners).astype(int).tolist()}")
    report_face("red top in the book's plane", plane, pts, ("far", "right", "near (top|side)", "left"))
    return corners, by_side


def blue_base(T, z):
    blob = cubes.find_blobs(frame)[0]
    M = T.copy(); M[:, 3] = T[:, 3] + z * T[:, 2]
    params = locate._fit_outline(M, blob, locate._start(M, blob))
    params, _, (starts, ends, found) = locate._fit_edges(frame, M, blob, params)
    seen = visible_edges(params[:2], params[2], camera_position(M))
    pix = project(M, cube_corners(params[:2], params[2]))
    groups = {}
    for e in seen:
        sel = ((starts == e.start) & (ends == e.end)) | ((starts == e.end) & (ends == e.start))
        if sel.sum() >= 6:
            groups[(e.start, e.end)] = (e, found[sel])
    bottom = [(k, v) for k, v in groups.items() if k[0] < 4 and k[1] < 4]
    uprights = [(k, v) for k, v in groups.items() if (k[0] < 4) != (k[1] < 4) and v[0].is_outline]
    print(f"  blue: bottom edges seen {[k for k, _ in bottom]}, upright outlines {[k for k, _ in uprights]}")
    if len(bottom) != 2 or len(uprights) != 2:
        print("  blue base: not two bottom edges and two upright outlines, left out")
        return
    lb = {k: line(v[1]) for k, v in bottom}
    lu = {k: line(v[1]) for k, v in uprights}
    (k1, l1), (k2, l2) = lb.items()
    near = meet(l1, l2)
    ends_px = []
    for kb, l in lb.items():
        # the upright that shares a corner with this bottom edge, away from the near corner
        for ku, u in lu.items():
            shared = set(kb) & set(ku)
            if shared:
                ends_px.append(meet(l, u))
    if len(ends_px) != 2:
        print("  blue base: corners not found, left out")
        return
    plane = to_plane(T, z, np.array([ends_px[0], near, ends_px[1]]))
    a, b = np.linalg.norm(plane[1] - plane[0]), np.linalg.norm(plane[2] - plane[1])
    u, v = plane[0] - plane[1], plane[2] - plane[1]
    angle = math.degrees(math.acos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
    print(f"  blue base in the book's plane: its two seen bottom sides {a:.2f} and {b:.2f} mm, at {angle:.1f} deg "
          f"(the corners where the upright outlines meet them)")


def main():
    for key, label in (((40.0, False, False), "book 40.0 mm, f 1781"), ((39.8, False, False), "book 39.8 mm"),
                       ((40.2, False, False), "book 40.2 mm")):
        T = matrix(fits[key][:9])
        print(f"{label}:")
        red_top(T, key[0])
        blue_base(T, key[0])


if __name__ == "__main__":
    main()
