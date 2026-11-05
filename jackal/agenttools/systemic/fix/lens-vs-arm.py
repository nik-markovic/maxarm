#!/usr/bin/env python3
"""Camera or arm? Camera models fitted to every point's cube edges alone (the arm has no say), then the arm's
readings laid on the cubes so placed by a turn and a shift: what each point is left with, out / round (+ toward
+x), mm, and each point left out of the turn and shift.

    lens-vs-arm.py [config_dir]        # default work/config-jackal-1104
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

JACKAL = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(JACKAL / "src"), str(JACKAL / "src" / "calibration")]
from applib.camera import edges, store  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tie import link_arm  # noqa: E402
from applib.camera.model import CUBE_MM, cube_corners  # noqa: E402
from applib.camera.sighting import start  # noqa: E402
from camera import refine  # noqa: E402

config = Path(sys.argv[1]) if len(sys.argv) > 1 else JACKAL.parent / "work" / "config-jackal-1104"
fit = store.load_camera(config)
stored = [p for p in store.load_refine_points(config, fit) if p.edges]
points = [p.sighting for p in stored]
arm = np.array([(p.x, p.y) for p in points])
width, height = fit.frame_size
first = refine._numbers(fit.true_to_pixel)


def project(camera, xyz):
    focal, aspect, cx, cy, rx, ry, rz, tx, ty, tz, k1, k2, p1, p2 = camera
    seen = xyz @ cv2.Rodrigues(np.array([rx, ry, rz]))[0].T + (tx, ty, tz)
    u, v = seen[:, 0] / seen[:, 2], seen[:, 1] / seen[:, 2]
    r2 = u * u + v * v
    bend = 1 + k1 * r2 + k2 * r2 * r2
    du, dv = 2 * p1 * u * v + p2 * (r2 + 2 * u * u), p1 * (r2 + 2 * v * v) + 2 * p2 * u * v
    return np.column_stack((focal * (u * bend + du) + cx, focal * aspect * (v * bend + dv) + cy))


def misses(camera, point, cube):
    corners = project(camera, cube_corners(cube[:2], cube[2], CUBE_MM))
    return edges.signed_distances(corners, *point.edges)


def solve(residuals, x, free, iterations=40):
    x = x.astype(float).copy()
    for _ in range(iterations):
        r = residuals(x)
        w = np.where(np.abs(r) <= 1.0, 1.0, 1.0 / np.maximum(np.abs(r), 1e-12))
        J = np.empty((len(r), len(free)))
        for c, i in enumerate(free):
            h = 1e-6 * max(1.0, abs(x[i]))
            xx = x.copy(); xx[i] += h
            J[:, c] = (residuals(xx) - r) / h
        A = J.T @ (J * w[:, None])
        step = np.linalg.lstsq(A + 1e-6 * np.diag(np.diag(A)), -(J.T @ (w * r)), rcond=None)[0]
        x[free] += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return x


def fit_camera(free_camera, keep=None):
    keep = range(len(points)) if keep is None else keep
    sel = [points[i] for i in keep]
    camera = np.array([first[0], 1.0, *first[1:9], 0, 0, 0, 0])
    cubes = [start(fit.true_to_pixel, p)[:3] for p in sel]
    x = np.concatenate([camera, np.concatenate(cubes)])
    free = list(free_camera) + list(range(14, 14 + 3 * len(sel)))

    def residuals(x):
        parts = [misses(x[:14], p, x[14 + 3 * i: 17 + 3 * i]) for i, p in enumerate(sel)]
        parts.append(np.array([x[2] - width / 2, x[3] - height / 2]) / (0.1 * width))
        return np.concatenate(parts)

    x = solve(residuals, x, free)
    r = residuals(x)[:-2]
    return x[:14], x[14:].reshape(-1, 3)[:, :2], float(np.sqrt(np.mean(np.minimum(np.abs(r), 3) ** 2)))


def polar(xy, d):
    b = math.atan2(xy[0], -xy[1])
    return d @ (math.sin(b), -math.cos(b)), d @ (math.cos(b), math.sin(b))


def report(name, free_camera):
    camera, cubes, edge_rms = fit_camera(free_camera)
    turn, shift, mirror = link_arm(arm, cubes)
    laid = (turn @ mirror @ arm.T).T + shift
    back = np.linalg.solve(turn @ mirror, (cubes - shift).T).T          # each cube in the arm's desk mm
    rows = [polar(a, c - a) for a, c in zip(arm, back)]
    lo = []
    for i in range(len(points)):
        rest = [j for j in range(len(points)) if j != i]
        t, s, m = link_arm(arm[rest], cubes[rest])
        lo.append(np.linalg.norm(np.linalg.solve(t @ m, cubes[i] - s) - arm[i]))
    rotation = cv2.Rodrigues(camera[4:7])[0]
    where = -rotation.T @ camera[7:10]
    print(f"\n{name}: edges {edge_rms:.2f} px; focal {camera[0]:.0f} aspect {camera[1]:.4f} k1 {camera[10]:+.4f} "
          f"k2 {camera[11]:+.4f} p {camera[12]:+.4f} {camera[13]:+.4f}; lens {where[2]:.0f} up")
    print(f"  arm on cubes, turn+shift: {np.median(np.linalg.norm(laid - cubes, axis=1)):.2f} typ, "
          f"{np.max(np.linalg.norm(laid - cubes, axis=1)):.2f} worst; left out {np.median(lo):.2f} typ, "
          f"{max(lo):.2f} worst")
    print("   id   bear  reach   out  round   leftout")
    for p, a, (o, r), l in sorted(zip(stored, arm, rows, lo), key=lambda t: math.atan2(t[1][0], -t[1][1])):
        print(f"  {p.id:3d} {math.degrees(math.atan2(a[0], -a[1])):+6.1f} {np.hypot(*a):6.1f} {o:+5.2f} {r:+5.2f}"
              f"    {l:5.2f}")
    return camera, cubes


if __name__ == "__main__":
    base = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    report("square pixels, no bend (the app's)", base)
    report("lens bends (k1)", base + [10])
    report("lens bends (k1, k2)", base + [10, 11])
    report("k1, k2, tangential", base + [10, 11, 12, 13])
    report("k1, aspect", base + [10, 1])
