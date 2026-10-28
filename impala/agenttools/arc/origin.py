"""Where each model fitted to the owner's 17 points puts the arm's axis, desk (0, 0), in the camera's mm. No arm.

The arcs on the pages are drawn round desk (0, 0) through camera.json's 6-number fit; each model below is fitted
to the same points, and its axis compared with that one.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from floor import fit_polar, polar_model  # noqa: E402
from points import fit_affine, load  # noqa: E402

fit, rows = load()
arm = np.array([(r["x"], r["y"]) for r in rows])
raw = np.array([r["raw"] for r in rows])
cam = np.array([r["cam"] for r in rows])


def rigid_map(src, dst, kind):
    mapped, prm = fit_affine(src, dst, kind)
    mirror, scale, turn = prm
    s = src * ([-1, 1] if mirror else [1, 1])
    R = np.array([[math.cos(math.radians(turn)), -math.sin(math.radians(turn))],
                  [math.sin(math.radians(turn)), math.cos(math.radians(turn))]]) * scale
    t = (mapped - s @ R.T).mean(0)
    return lambda p: (np.atleast_2d(p) * ([-1, 1] if mirror else [1, 1])) @ R.T + t


current = fit.desk_to_true @ np.zeros(2)          # the refined camera.json: desk (0,0) is true (0,0)
ref = np.zeros(2)
models = {}
_, sol = fit_affine(arm, cam, "affine")
models["6 numbers (camera.json's refinement), library x,y"] = np.array([0.0, 0, 1]) @ sol
for kind in ("rigid", "similarity"):
    models[f"{kind}, library x,y"] = rigid_map(arm, cam, kind)(np.zeros(2))[0]
    models[f"{kind}, firmware x,y"] = rigid_map(raw, cam, kind)(np.zeros(2))[0]
_, sol = fit_affine(raw, cam, "affine")
models["6 numbers, firmware x,y"] = np.array([0.0, 0, 1]) @ sol
for free, label in (([1, 2, 4], "wave + reach offset"), ([0, 1, 2, 3, 4], "bearing scale+wave, reach scale+offset")):
    _, full = fit_polar(raw, cam, free)
    corrected = polar_model(raw, full)
    models[f"firmware, polar: {label}"] = rigid_map(corrected, cam, "rigid")(np.zeros(2))[0]
cubes_alone = fit.cube_fit()
# the cubes alone: their axis in the refined camera's true mm. Both share the camera; express via pixels.
from camera.model import project  # noqa: E402
pix = project(cubes_alone.desk_to_pixel, np.array([[0.0, 0.0, 0.0]]))[0]
H = fit.true_to_pixel[:, [0, 1, 3]]
q = np.linalg.solve(H, (*pix, 1.0))
models["the three cubes alone (CALIBRATE)"] = q[:2] / q[2]
print("where the axis falls, against camera.json's (refined) desk (0, 0), mm on the desk (camera's frame)")
for label, p in models.items():
    d = np.linalg.solve(fit.desk_to_true, p)       # in the arm's directions: x right, y back
    print(f"  {label:58s} {np.hypot(*d):5.2f} mm   (x {d[0]:+5.2f}, y {d[1]:+5.2f} in the arm's directions)")
