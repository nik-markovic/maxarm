#!/usr/bin/env python3
"""Which camera model puts a 40 mm cube on every point's edges: the app's (square pixels, no bend), pixels
not square, the lens bending (k1, k2), the red cube another size. Each cube free; the picture alone.

    ../.venv/bin/python agenttools/systemic/camera-models.py [config_dir]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "calibration")]

from applib.camera import edges, store                                              # noqa: E402
from applib.camera.model import CUBE_MM, cube_corners                               # noqa: E402
from applib.camera.sighting import start                                            # noqa: E402
from camera import refine                                                           # noqa: E402

config = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "config"
fit = store.load_camera(config)
alone = fit.cube_fit()
points = [p.sighting for p in store.load_refine_points(config, fit) if p.sighting.edges is not None]
colours = [p.cube for p in store.load_refine_points(config, fit) if p.sighting.edges is not None]
width, height = fit.frame_size
first = refine._numbers(fit.true_to_pixel)


def project(camera, points3):
    focal, aspect, cx, cy, rx, ry, rz, tx, ty, tz, k1, k2 = camera
    rotation = cv2.Rodrigues(np.array([rx, ry, rz]))[0]
    seen = points3 @ rotation.T + (tx, ty, tz)
    u, v = seen[:, 0] / seen[:, 2], seen[:, 1] / seen[:, 2]
    r2 = (u * u + v * v)
    bend = 1 + k1 * r2 + k2 * r2 * r2
    return np.column_stack((focal * u * bend + cx, focal * aspect * v * bend + cy))


def misses(camera, point, cube, size):
    starts, ends, found = point.edges
    corners = project(camera, cube_corners(cube[:2], cube[2], size))
    return edges.signed_distances(corners, starts, ends, found)


def solve(residuals, x, free, iterations=30):
    x = x.astype(float).copy()
    for _ in range(iterations):
        r = residuals(x)
        w = np.where(np.abs(r) <= 1.0, 1.0, 1.0 / np.maximum(np.abs(r), 1e-12))
        J = np.empty((len(r), len(free)))
        for c, i in enumerate(free):
            h = 1e-6 * max(1.0, abs(x[i]))
            xx = x.copy(); xx[i] += h
            J[:, c] = (residuals(xx) - r) / h
        A = J.T @ (J * w[:, None]); g = J.T @ (w * r)
        step = np.linalg.lstsq(A + 1e-6 * np.diag(np.diag(A)), -g, rcond=None)[0]
        x[free] += step
        if np.max(np.abs(step)) < 1e-7:
            break
    return x


def run(name, free_camera, red_size_free=False):
    camera = np.array([first[0], 1.0, *first[1:9], 0.0, 0.0])
    cubes = [start(fit.true_to_pixel, p)[:3] for p in points]
    n = len(points)
    x = np.concatenate([camera, np.concatenate(cubes), [CUBE_MM]])
    free = list(free_camera) + list(range(12, 12 + 3 * n)) + ([12 + 3 * n] if red_size_free else [])

    def residuals(x):
        cam = x[:12]
        parts = []
        for i, p in enumerate(points):
            size = x[-1] if colours[i] in (None, "red") else CUBE_MM
            parts.append(misses(cam, p, x[12 + 3 * i: 15 + 3 * i], size))
        parts.append(np.array([(x[2] - width / 2), (x[3] - height / 2)]) / (0.1 * width))
        return np.concatenate(parts)

    x = solve(residuals, x, free)
    cam = x[:12]
    rotation = cv2.Rodrigues(cam[4:7])[0]
    position = -rotation.T @ cam[7:10]
    per = []
    for i, p in enumerate(points):
        size = x[-1] if colours[i] in (None, "red") else CUBE_MM
        per.append(np.sqrt(np.mean(misses(cam, p, x[12 + 3 * i: 15 + 3 * i], size) ** 2)))
    allr = residuals(x)[:-2]
    print(f"{name:34s} edges {np.sqrt(np.mean(allr ** 2)):.2f} px (per cube {' '.join(f'{v:.1f}' for v in per)}); "
          f"focal {cam[0]:.0f} aspect {cam[1]:.4f} k1 {cam[10]:+.3f} k2 {cam[11]:+.3f} red {x[-1]:.2f} mm; "
          f"camera {position[2]:.1f} mm up")
    return x


base = [0, 2, 3, 4, 5, 6, 7, 8, 9]
run("square pixels, no bend (the app's)", base)
run("pixels not square", base + [1])
run("lens bends (k1)", base + [10])
run("lens bends (k1, k2)", base + [10, 11])
run("red cube's size free", base, red_size_free=True)
run("all of them", base + [1, 10, 11], red_size_free=True)
