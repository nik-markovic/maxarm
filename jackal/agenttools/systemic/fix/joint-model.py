#!/usr/bin/env python3
"""Which camera model fits reality: the 3 Nov cubes' edges (12:18 frame) and the grid paper's lines (20:24 frame)
by one camera, with terms added to the square-pixel pinhole one at a time -- the pixels' aspect, a radial bend k1,
a bend along one frame axis only. If a term brings the cubes' edges back to what they fit alone while the paper stays
at 0.15 mm, the app's camera lacks it, and three cubes alone cannot see it.

    joint-model.py          (needs /tmp/fit1218.pkl, made by the 12:18 refit)
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
PIX, MM, IS_X = paperlens.pix, paperlens.mm, paperlens.is_x
FIT = pickle.load(open("/tmp/fit1218.pkl", "rb"))
WIDTH, HEIGHT = FIT["frame_size"]
COLOURS = list(FIT["cube_edges"])
NAMES = ("f", "cx", "cy", "rx", "ry", "rz", "tx", "ty", "tz", "aspect", "k1", "kx", "ky")
CAM = len(NAMES)


def project(cam, xyz):
    f, cx, cy, rx, ry, rz, tx, ty, tz, aspect, k1, kx, ky = cam
    seen = xyz @ cv2.Rodrigues(np.array([rx, ry, rz]))[0].T + (tx, ty, tz)
    u, v = seen[:, 0] / seen[:, 2], seen[:, 1] / seen[:, 2]
    bend = 1 + k1 * (u * u + v * v)
    return np.c_[f * u * bend * (1 + kx * u * u) + cx, f * aspect * v * bend * (1 + ky * v * v) + cy]


def on_desk(cam, p):
    """Desk (X, Y) under pixels p: the pinhole part solved, then Newton on the bends."""
    f, cx, cy, rx, ry, rz, tx, ty, tz, aspect = cam[:10]
    T = np.array([[f, 0, cx], [0, f * aspect, cy], [0, 0, 1.0]]) @ np.c_[cv2.Rodrigues(np.array([rx, ry, rz]))[0],
                                                                         (tx, ty, tz)]
    w = np.linalg.solve(T[:, [0, 1, 3]], np.c_[p, np.ones(len(p))].T)
    X = (w[:2] / w[2]).T
    if not np.any(cam[10:]):
        return X
    for _ in range(5):
        q = project(cam, np.c_[X, np.zeros(len(X))])
        qx = project(cam, np.c_[X + (1e-3, 0), np.zeros(len(X))])
        qy = project(cam, np.c_[X + (0, 1e-3), np.zeros(len(X))])
        jx, jy, e = (qx - q) / 1e-3, (qy - q) / 1e-3, p - q
        det = jx[:, 0] * jy[:, 1] - jx[:, 1] * jy[:, 0]
        X = X + np.c_[(e[:, 0] * jy[:, 1] - e[:, 1] * jy[:, 0]) / det, (jx[:, 0] * e[:, 1] - jx[:, 1] * e[:, 0]) / det]
    return X


def edge_res(x):
    out = []
    for i, c in enumerate(COLOURS):
        cx, cy, turn = x[CAM + 3 * i: CAM + 3 * i + 3]
        out.append(edges.signed_distances(project(x[:CAM], cube_corners((cx, cy), turn)), *FIT["cube_edges"][c][1]))
    return np.concatenate(out)


PAPER = CAM + 3 * len(COLOURS)                      # the paper's turn and shift on the desk, then its handedness


def paper_res(x, flip):
    P = on_desk(x[:CAM], PIX) - x[PAPER + 1: PAPER + 3]
    c, s = math.cos(x[PAPER]), math.sin(x[PAPER])
    Q = np.c_[c * P[:, 0] + s * P[:, 1], flip * (-s * P[:, 0] + c * P[:, 1])]
    return np.where(IS_X, Q[:, 0] - MM, Q[:, 1] - MM)


def solve(fun, x, free, iterations=50):
    x = x.astype(float).copy()
    for _ in range(iterations):
        r = fun(x)
        w = np.where(np.abs(r) <= 1.5, 1.0, 1.5 / np.maximum(np.abs(r), 1e-12))
        J = np.empty((len(r), len(free)))
        for k, i in enumerate(free):
            h = 1e-6 * max(1.0, abs(x[i])) if i >= CAM or NAMES[i] not in ("k1", "kx", "ky") else 1e-5
            xx = x.copy(); xx[i] += h
            J[:, k] = (fun(xx) - r) / h
        A = J.T @ (J * w[:, None])
        step = np.linalg.lstsq(A + 1e-6 * np.diag(np.diag(A)), -(J.T @ (w * r)), rcond=None)[0]
        x[free] += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return x


def held(x):
    return (x[1:3] - (WIDTH / 2, HEIGHT / 2)) / (0.1 * WIDTH)


def main():
    T0 = FIT["true_to_pixel"]
    first = refine._numbers(T0)
    cubes = []
    for c in COLOURS:
        w = np.linalg.solve(T0[:, [0, 1, 3]], np.array([*FIT["cubes"][c]["pixel"], 1.0]))
        cubes += [w[0] / w[2], w[1] / w[2], math.radians(FIT["cube_edges"][c][0])]
    x = np.concatenate([first, [1.0, 0.0, 0.0, 0.0], cubes, [0.0, 0.0, 0.0]])
    pose = list(range(9))
    cube_ids = list(range(CAM, PAPER))
    paper_ids = [PAPER, PAPER + 1, PAPER + 2]
    x = solve(lambda y: np.concatenate([edge_res(y), held(y)]), x, pose + cube_ids)
    best = None
    for flip in (1.0, -1.0):
        for theta in np.radians(np.arange(0, 360, 10)):
            y = x.copy(); y[PAPER:PAPER + 3] = (theta, 0.0, 0.0)
            y = solve(lambda z: paper_res(z, flip), y, paper_ids, 25)
            r = np.median(np.abs(paper_res(y, flip)))
            if best is None or r < best[0]:
                best = (r, y, flip)
    _, x, flip = best
    alone = x.copy()
    print(f"cubes alone, square pixels: edges {np.median(np.abs(edge_res(x))):.2f} px typ; the paper through it "
          f"{np.median(np.abs(paper_res(x, flip))):.3f} mm typ, {np.percentile(np.abs(paper_res(x, flip)), 95):.2f} at 95%")
    for terms in ((), ("aspect",), ("k1",), ("kx",), ("ky",), ("aspect", "k1"), ("k1", "kx", "ky")):
        free = pose + [NAMES.index(t) for t in terms] + cube_ids + paper_ids
        y = solve(lambda z: np.concatenate([edge_res(z), paper_res(z, flip) / 0.15, held(z)]), alone, free)
        e, p = edge_res(y), paper_res(y, flip)
        shown = ", ".join(f"{t} {y[NAMES.index(t)]:+.4f}" for t in terms) or "square pixels, no bend"
        print(f"both, {shown:40s}: edges {np.median(np.abs(e)):.2f} px typ; paper {np.median(np.abs(p)):.3f} mm typ, "
              f"{np.percentile(np.abs(p), 95):.2f} at 95%; focal {y[0]:.0f}", flush=True)


if __name__ == "__main__":
    main()
