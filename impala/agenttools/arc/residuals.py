"""Each point's miss after the best rigid fit, split along and across the arm's line. No arm."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from points import fit_affine, load  # noqa: E402

fit, rows = load()
arm = np.array([(r["x"], r["y"]) for r in rows])
cam = np.array([r["cam"] for r in rows])
for kind in ("rigid", "affine"):
    mapped, params = fit_affine(arm, cam, kind)
    # express the misses in the arm's frame: map the camera points back through the inverse of the fit
    if kind == "rigid":
        mirror, scale, turn = params
        R = np.array([[math.cos(math.radians(turn)), -math.sin(math.radians(turn))],
                      [math.sin(math.radians(turn)), math.cos(math.radians(turn))]])
        lin = R @ np.diag([-1 if mirror else 1, 1]) * scale
    else:
        lin = params[:2].T
    miss = (cam - mapped) @ np.linalg.inv(lin).T          # arm mm: where the camera sees it, minus the arm's
    print(f"\n{kind}: miss = camera - arm, in the arm's frame; out = away from the base axis, round = toward +bearing")
    print(" pt   reach bearing    dx     dy   |  out   round")
    for r, (dx, dy) in zip(rows, miss):
        b = math.atan2(r["x"], -r["y"])
        out = dx * math.sin(b) - dy * math.cos(b)
        rnd = dx * math.cos(b) + dy * math.sin(b)
        print(f"{r['name']:>3} {math.hypot(r['x'], r['y']):6.1f} {math.degrees(b):7.1f}  {dx:5.2f}  {dy:5.2f}  | {out:5.2f}  {rnd:5.2f}")
