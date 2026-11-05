#!/usr/bin/env python3
"""One square-pixel camera for the cubes and the paper (3 Nov): the three cubes' edges in the 12:18 frame and the
grid paper's lines in the 20:24 frame, as one camera that has not moved. Fitted to the cubes alone, to the paper
alone (its pose and the cubes' heights then free of it), and to both. For each: the cubes' edges (px), the paper's
lines (mm), how the camera's desk mm measure the paper, and how far apart two of these cameras put desk points
across the work area.

    joint-paper-cubes.py
"""
import contextlib
import importlib.util
import io
import math
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
JACKAL = HERE.parents[2]
sys.path[:0] = [str(JACKAL / "src"), str(JACKAL / "src" / "calibration")]
from applib.camera import edges  # noqa: E402
from applib.camera.model import cube_corners  # noqa: E402
from camera import refine  # noqa: E402

spec = importlib.util.spec_from_file_location("paperlens", HERE / "paper-lens.py")
with contextlib.redirect_stdout(io.StringIO()):
    paperlens = importlib.util.module_from_spec(spec)
    sys.argv = sys.argv[:1]
    spec.loader.exec_module(paperlens)
pix, mm, is_x = paperlens.pix, paperlens.mm, paperlens.is_x

fit = pickle.load(open("/tmp/fit1218.pkl", "rb"))
T0 = fit["true_to_pixel"]
width, height = fit["frame_size"]
colours = list(fit["cube_edges"])
first = refine._numbers(T0)


def plane_point(T, p):
    w = np.linalg.solve(T[:, [0, 1, 3]], np.array([p[0], p[1], 1.0]))
    return w[:2] / w[2]


cube0 = []
for c in colours:
    turn, _ = fit["cube_edges"][c]
    cube0 += [*plane_point(T0, fit["cubes"][c]["pixel"]), math.radians(turn)]


def matrix(cam):
    K = np.array([[cam[0], 0, cam[1]], [0, cam[0], cam[2]], [0, 0, 1.0]])
    R = cv2.Rodrigues(cam[3:6])[0]
    return K @ np.c_[R, cam[6:9]]


def edge_res(x):
    T = matrix(x[:9])
    out = []
    for i, c in enumerate(colours):
        cx, cy, turn = x[9 + 3 * i: 12 + 3 * i]
        corners = T @ np.c_[cube_corners((cx, cy), turn), np.ones(8)].T
        corners = (corners[:2] / corners[2]).T
        out.append(edges.signed_distances(corners, *fit["cube_edges"][c][1]))
    return np.concatenate(out)


def paper_res(x, flip):
    T = matrix(x[:9])
    theta, ox, oy = x[18:21]
    w = np.linalg.solve(T[:, [0, 1, 3]], np.c_[pix, np.ones(len(pix))].T)
    P = (w[:2] / w[2]).T - (ox, oy)
    c, s = math.cos(theta), math.sin(theta)
    Q = np.c_[c * P[:, 0] + s * P[:, 1], flip * (-s * P[:, 0] + c * P[:, 1])]
    return np.where(is_x, Q[:, 0] - mm, Q[:, 1] - mm)


def residuals(x, use, flip):
    parts = []
    if "cubes" in use:
        parts.append(edge_res(x) / 1.0)
    if "paper" in use:
        parts.append(paper_res(x, flip) / 0.15)
    parts.append((x[1:3] - (width / 2, height / 2)) / (0.1 * width))
    return np.concatenate(parts)


def solve(fun, x, free, iterations=60):
    x = x.astype(float).copy()
    for _ in range(iterations):
        r = fun(x)
        w = np.where(np.abs(r) <= 1.5, 1.0, 1.5 / np.maximum(np.abs(r), 1e-12))
        J = np.empty((len(r), len(free)))
        for k, i in enumerate(free):
            h = 1e-6 * max(1.0, abs(x[i]))
            xx = x.copy(); xx[i] += h
            J[:, k] = (fun(xx) - r) / h
        A = J.T @ (J * w[:, None])
        step = np.linalg.lstsq(A + 1e-6 * np.diag(np.diag(A)), -(J.T @ (w * r)), rcond=None)[0]
        x[free] += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return x


def paper_start(x, flip):
    """The paper's pose on the camera's desk, for the camera in x: a similarity's turn and shift (scale left out)."""
    T = matrix(x[:9])
    best = None
    for theta in np.radians(np.arange(0, 360, 5)):
        x2 = x.copy(); x2[18:21] = (theta, 0, 0)
        w = np.linalg.solve(T[:, [0, 1, 3]], np.c_[pix, np.ones(len(pix))].T)
        P = (w[:2] / w[2]).T
        x2 = solve(lambda y: paper_res(y, flip), x2, [18, 19, 20], 30)
        r = np.median(np.abs(paper_res(x2, flip)))
        if best is None or r < best[0]:
            best = (r, x2[18:21].copy())
    return best[1]


x0 = np.concatenate([first, cube0, [0.0, 0.0, 0.0]])
results = {}
for flip in (1.0, -1.0):
    x = x0.copy()
    x[18:21] = paper_start(x, flip)
    results[flip] = np.median(np.abs(paper_res(x, flip)))
flip = min(results, key=results.get)
x0[18:21] = paper_start(x0, flip)

fits = {}
cam_free = list(range(9))
fits["cubes alone"] = solve(lambda y: residuals(y, ("cubes",), flip), x0, cam_free + list(range(9, 18)))
fits["cubes alone"][18:21] = paper_start(fits["cubes alone"], flip)
fits["cubes alone"] = solve(lambda y: paper_res(y, flip), fits["cubes alone"], [18, 19, 20])
fits["paper alone"] = solve(lambda y: residuals(y, ("paper",), flip), fits["cubes alone"], cam_free + [18, 19, 20])
fits["paper alone"] = solve(lambda y: residuals(y, ("cubes",), flip), fits["paper alone"], list(range(9, 18)))
fits["both"] = solve(lambda y: residuals(y, ("cubes", "paper"), flip), fits["cubes alone"], list(range(21)))

grid = np.array([(r * math.sin(math.radians(b)), -r * math.cos(math.radians(b)))
                 for r in (120, 180, 240) for b in range(-90, 91, 30)])
for name, x in fits.items():
    e, p = edge_res(x), paper_res(x, flip)
    T = matrix(x[:9])
    lens = -cv2.Rodrigues(x[3:6])[0].T @ x[6:9]
    print(f"{name:12s}: cube edges {np.median(np.abs(e)):.2f} px typ ({np.sqrt(np.mean(np.minimum(np.abs(e), 4) ** 2)):.2f} rms); "
          f"paper lines {np.median(np.abs(p)):.3f} mm typ, {np.percentile(np.abs(p), 95):.2f} at 95%; focal {x[0]:.0f}, "
          f"lens {lens[2]:.0f} up")
print("\nhow far the cubes-alone camera puts desk points from where the paper-and-cubes camera does (same pixel), mm,"
      " through each one's own desk plane:")
Ta, Tb = matrix(fits["cubes alone"][:9]), matrix(fits["both"][:9])
# compare in the paper's mm: a pixel -> each camera's desk -> paper coordinates through each fit's paper pose
def to_paper(x, P):
    theta, ox, oy = x[18:21]
    c, s = math.cos(theta), math.sin(theta)
    Q = P - (ox, oy)
    return np.array([c * Q[0] + s * Q[1], flip * (-s * Q[0] + c * Q[1])])
pts_b = []
for name in ("cubes alone",):
    x = fits[name]
    T = matrix(x[:9])
    diffs = []
    for g in grid:
        # a desk point by the joint camera, in its true frame -> pixel -> the other camera's desk -> paper mm in each
        q = Tb @ np.array([*g, 0, 1.0]); q = q[:2] / q[2]
        pa = to_paper(x, plane_point(T, q))
        pb = to_paper(fits["both"], plane_point(Tb, q))
        diffs.append(np.linalg.norm(pa - pb))
    diffs = np.array(diffs)
    print(f"   {name}: {np.median(diffs):.2f} typ, {diffs.max():.2f} worst over reach 120-240, bearings +-90 (the"
          f" joint camera's frame taken as the desk, roughly)")
