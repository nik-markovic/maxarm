#!/usr/bin/env python3
"""Today's frame: the two grid sheets (each its own place on one desk plane) and the three cubes (40 mm), as one
square-pixel camera. Fitted to the sheets alone (the lens's middle held loosely to the frame's), the cubes alone,
both; and the 4 Nov camera.json as it is. For each: how it reads each sheet along desk x and y, the sheets' lines
(mm), the cubes' edges (px), the focal length and where the lens stands.

    today-joint.py
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
from applib.camera import edges, store  # noqa: E402
from applib.camera.model import cube_corners  # noqa: E402
from camera import refine  # noqa: E402

S = np.load(HERE / "out" / "sheets.npz")
SHEETS = [(S[f"{n}_pix"], S[f"{n}_mm"], S[f"{n}_is_x"]) for n in ("left", "right")]
FIT = pickle.load(open("/tmp/fit-today.pkl", "rb"))
COLOURS = list(FIT["cube_edges"])
WIDTH, HEIGHT = FIT["frame_size"]
CAM = 9                                         # f, cx, cy, rotation 3, shift 3
CUBES = CAM                                     # 3 per cube: x, y, turn
SHEET0 = CUBES + 3 * len(COLOURS)               # 3 per sheet: turn, shift x, y (in the desk plane)


def matrix(cam):
    K = np.array([[cam[0], 0, cam[1]], [0, cam[0], cam[2]], [0, 0, 1.0]])
    return K @ np.c_[cv2.Rodrigues(cam[3:6])[0], cam[6:9]]


def edge_res(x):
    T = matrix(x[:CAM])
    out = []
    for i, c in enumerate(COLOURS):
        cx, cy, turn = x[CUBES + 3 * i: CUBES + 3 * i + 3]
        p = T @ np.c_[cube_corners((cx, cy), turn), np.ones(8)].T
        out.append(edges.signed_distances((p[:2] / p[2]).T, *FIT["cube_edges"][c][1]))
    return np.concatenate(out)


def sheet_res(x, k, flip=1.0):
    pix, mm, is_x = SHEETS[k]
    T = matrix(x[:CAM])
    w = np.linalg.solve(T[:, [0, 1, 3]], np.c_[pix, np.ones(len(pix))].T)
    P = (w[:2] / w[2]).T - x[SHEET0 + 3 * k + 1: SHEET0 + 3 * k + 3]
    c, s = math.cos(x[SHEET0 + 3 * k]), math.sin(x[SHEET0 + 3 * k])
    Q = np.c_[c * P[:, 0] + s * P[:, 1], flip * (-s * P[:, 0] + c * P[:, 1])]
    return np.where(is_x, Q[:, 0] - mm, Q[:, 1] - mm)


FLIP = [1.0, 1.0]


def all_sheets(x):
    return np.concatenate([sheet_res(x, k, FLIP[k]) for k in range(len(SHEETS))])


def held(x):
    return (x[1:3] - (WIDTH / 2, HEIGHT / 2)) / (0.1 * WIDTH)


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


def sheet_poses(x):
    """Each sheet's turn and shift on the camera's desk plane, and its handedness, for the camera in x."""
    for k in range(len(SHEETS)):
        best = None
        for flip in (1.0, -1.0):
            for theta in np.radians(np.arange(0, 360, 10)):
                y = x.copy(); y[SHEET0 + 3 * k: SHEET0 + 3 * k + 3] = (theta, 0, 0)
                y = solve(lambda z: sheet_res(z, k, flip), y, [SHEET0 + 3 * k + j for j in range(3)], 20)
                r = np.median(np.abs(sheet_res(y, k, flip)))
                if best is None or r < best[0]:
                    best = (r, y[SHEET0 + 3 * k: SHEET0 + 3 * k + 3].copy(), flip)
        x[SHEET0 + 3 * k: SHEET0 + 3 * k + 3], FLIP[k] = best[1], best[2]
    return x


def place_cubes(x):
    return solve(lambda z: edge_res(z), x, list(range(CUBES, SHEET0)))


def reads(x, k):
    """How the camera's desk mm read sheet k's mm along the sheet's two axes (close to desk x and y)."""
    pix, mm, is_x = SHEETS[k]
    T = matrix(x[:CAM])
    out = []
    for axis in (True, False):
        sel = is_x == axis
        w = np.linalg.solve(T[:, [0, 1, 3]], np.c_[pix[sel], np.ones(sel.sum())].T)
        P = (w[:2] / w[2]).T
        # the camera's mm across the lines against the lines' own mm: slope of desk position along the sheet axis
        theta = x[SHEET0 + 3 * k]
        axis_dir = np.array([math.cos(theta), math.sin(theta)]) if axis else \
            FLIP[k] * np.array([-math.sin(theta), math.cos(theta)])
        along = P @ axis_dir
        slope = np.polyfit(mm[sel], along, 1)[0]
        out.append(abs(slope))
    return out


def report(name, x):
    T = matrix(x[:CAM])
    lens = -cv2.Rodrigues(x[3:6])[0].T @ x[6:9]
    e = edge_res(x)
    s = [sheet_res(x, k, FLIP[k]) for k in range(len(SHEETS))]
    r = [reads(x, k) for k in range(len(SHEETS))]
    print(f"{name:26s} focal {x[0]:6.0f}, lens {lens[2]:5.0f} up | cubes' edges {np.median(np.abs(e)):.2f} px | "
          f"sheets' lines {np.median(np.abs(s[0])):.3f} / {np.median(np.abs(s[1])):.3f} mm | reads left sheet x "
          f"{r[0][0]:.4f} y {r[0][1]:.4f}, right sheet x {r[1][0]:.4f} y {r[1][1]:.4f}", flush=True)


def main():
    T0 = FIT["true_to_pixel"]
    x = np.zeros(SHEET0 + 3 * len(SHEETS))
    x[:CAM] = refine._numbers(T0)
    for i, c in enumerate(COLOURS):
        w = np.linalg.solve(T0[:, [0, 1, 3]], np.array([*FIT["cubes"][c]["pixel"], 1.0]))
        x[CUBES + 3 * i: CUBES + 3 * i + 3] = (w[0] / w[2], w[1] / w[2], math.radians(FIT["cube_edges"][c][0]))
    pose = list(range(CAM))
    sheets = list(range(SHEET0, len(x)))
    cubes_alone = sheet_poses(solve(lambda z: np.concatenate([edge_res(z), held(z)]), x, pose + list(range(CUBES, SHEET0))))
    cubes_alone = sheet_poses(cubes_alone)
    report("cubes alone (today's 3)", cubes_alone)
    sheets_alone = solve(lambda z: np.concatenate([all_sheets(z) / 0.1, held(z)]), cubes_alone, pose + sheets)
    sheets_alone = place_cubes(sheets_alone)
    report("sheets alone", sheets_alone)
    both = solve(lambda z: np.concatenate([edge_res(z), all_sheets(z) / 0.1, held(z)]), sheets_alone,
                 list(range(len(x))))
    report("sheets and cubes", both)
    # the 4 Nov camera.json as it is (moved 4 px since): its camera alone, cubes placed through it
    saved = store.load_camera(JACKAL / "config")
    y = both.copy()
    y[:CAM] = refine._numbers(saved.true_to_pixel)
    y = place_cubes(sheet_poses(y))
    report("4 Nov camera.json (12 pts)", y)
    pickle.dump({"both": both, "sheets_alone": sheets_alone, "cubes_alone": cubes_alone, "saved": y, "flip": FLIP},
                open("/tmp/today-joint.pkl", "wb"))


if __name__ == "__main__":
    main()
