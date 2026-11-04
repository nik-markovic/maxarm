#!/usr/bin/env python3
"""The owner's way, tried on 3 Nov's data: the arm's error as a fixed model in the library; the camera alone
calibrated, from CALIBRATE's three cubes; each cube's base tied to where the arm really is. Judged by the 24 desk
points' cups (analysis.json), which the calibration never sees.

Library models (the firmware's raw reading -> real x, y; the wave as raw.py has it):
  now      reach + 1.6 at every height
  +down    reach + 6.26 - 0.110 min(h, 44): 1.6 at the cubes' notch as now, the way down measured on 3 Nov
  +stretch the same, the reach stretched by k about a middle arc R0 (nothing moves there), k from the paper
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JACKAL = HERE.parents[2]
sys.path.insert(0, str(JACKAL / "src"))
from maxarm import raw  # noqa: E402
from maxarm.spaces import Space, Spaces  # noqa: E402

WAVE = (raw.WAVE_SIN_DEG, raw.WAVE_COS_DEG)
R0 = 180.0


def real_xy(raw_point, model, k=1.0):
    x, y, z = raw_point
    h = min(max(raw.true_position(raw_point)[2] - raw.DESK_Z_MM, 0.0), 44.0)
    reach, b = math.hypot(x, y), math.atan2(x, -y)
    reach += 1.6 if model == "now" else 6.26 - 0.110 * h
    if model == "+stretch":
        reach = R0 + k * (reach - R0)
    b += math.radians(WAVE[0] * math.sin(2 * b) + WAVE[1] * (math.cos(2 * b) - 1))
    return np.array([-reach * math.sin(b), -reach * math.cos(b)])      # the board's mirror: x negated, as the camera's world


def rigid(A, B, scale=False):
    a, b = A - A.mean(0), B - B.mean(0)
    U, sig, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, d])
    R = (U @ D @ Vt).T
    s = (sig * np.diag(D)).sum() / (a ** 2).sum() if scale else 1.0
    return lambda P: s * (P - A.mean(0)) @ R.T + B.mean(0), s


def affine(A, B):
    X, *_ = np.linalg.lstsq(np.c_[A, np.ones(len(A))], B, rcond=None)
    return lambda P: np.c_[P, np.ones(len(P))] @ X


def main() -> None:
    cam = json.loads((JACKAL / "config" / "camera.json").read_text())
    D2T = np.array(cam["desk_to_true"])
    spaces = Spaces(JACKAL / "config")
    a = json.loads((HERE / "analysis.json").read_text())
    desk = [p for p in a["points"] if math.hypot(*p["desk"][:2]) >= 109]
    desk_raw = [tuple(p["raw"]) for p in desk]
    cups_true = np.array([D2T @ np.array(p["rim_desk"]) for p in desk])     # the camera's real mm
    paper = np.array([p["paper"] for p in desk]) * [-1, 1]                 # mirrored too, to compare like with like
    cubes = json.loads((JACKAL / "config" / "camera-points.json").read_text())["cubes"]
    cube_raw = [raw.raw_position(spaces.convert((c["x"], c["y"], c["z"]), Space.DESK, Space.ARM)) for c in
                cubes.values()]
    cubes_true = np.array([D2T @ np.array([c["x"], c["y"]]) for c in cubes.values()])  # where the camera saw them

    # k: the reach stretch about R0 that lays the desk points on the paper best, by turn and shift alone
    best = None
    for k in np.arange(0.97, 1.03, 0.0005):
        arm = np.array([real_xy(p, "+stretch", k) for p in desk_raw])
        place, _ = rigid(arm, paper)
        d = np.hypot(*(place(arm) - paper).T)
        if best is None or np.median(d) < best[1]:
            best = (k, float(np.median(d)), float(d.max()))
    k = best[0]
    print(f"reach stretch about {R0:.0f}: k {k:.4f}; the paper by turn and shift {best[1]:.2f} typ, {best[2]:.2f} worst")
    for model in ("now", "+down", "+stretch"):
        arm = np.array([real_xy(p, model, k) for p in desk_raw])
        place, _ = rigid(arm, paper)
        d = np.hypot(*(place(arm) - paper).T)
        print(f"  {model:8s} on the paper: {np.median(d):.2f} typ, {d.max():.2f} worst")
    print("CALIBRATE's three cubes tie the arm to the camera; the 24 desk cups predicted (camera mm):")
    for model in ("now", "+down", "+stretch"):
        cube_arm = np.array([real_xy(p, model, k) for p in cube_raw])
        desk_arm = np.array([real_xy(p, model, k) for p in desk_raw])
        for tie_name, tie in (("turn, shift", lambda A, B: rigid(A, B)[0]),
                              ("turn, shift, scale", lambda A, B: rigid(A, B, True)[0]),
                              ("turn, shift, stretch (now)", affine)):
            t = tie(cube_arm, cubes_true)
            miss = t(desk_arm) - cups_true
            d = np.hypot(*miss.T)
            on_cubes = np.hypot(*(t(cube_arm) - cubes_true).T)
            out = np.array([m @ (q / np.linalg.norm(q)) for m, q in zip(-miss, desk_arm)])
            print(f"  {model:8s} {tie_name:27s}: cups {np.median(d):.2f} typ, {d.max():.2f} worst, out {out.mean():+.2f}"
                  f" | the cubes themselves {', '.join(f'{v:.2f}' for v in on_cubes)}")


if __name__ == "__main__":
    main()
