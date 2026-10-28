"""Gecko's eleven pencil circles (desk level) against where each calibration says the arm put the cup, and the
axis the seven at reach 200 fit, against each calibration's desk (0, 0). The calibrations: the evening's as saved
(6 numbers, refined on its 16 points), and the same points refined by the new code (a turn and a shift). No arm.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration"), str(Path(__file__).resolve().parent)]
from camera import store  # noqa: E402
from camera.fit import CameraFit  # noqa: E402
from camera.refine import refine  # noqa: E402
from maxarm import raw  # noqa: E402

OLD = ROOT.parent / "work" / "config-old"
data = np.load(Path(__file__).resolve().parent / "circles-true.npy", allow_pickle=True).item()
found, axis_true = data["found"], data["axis_true"]
MARKS = {"A": (-201, -1, 46), "B": (-160, -161, 45), "C": (0, -201, 44), "D": (160, -160, 44), "1": (-181, -80, 44),
         "2": (-80, -181, 44), "3": (79, -180, 44), "4": (181, -81, 44), "5": (201, 1, 44), "6": (-159, -1, 44),
         "7": (-260, -1, 44)}
saved = CameraFit.from_json(json.loads((OLD / "camera.json").read_text()))   # the circles are in its true mm
new = refine(saved.cube_fit(), [p.sighting for p in store.load_refine_points(OLD)])


def in_saved_true(fit, desk_xy):
    """A desk point, through `fit`, as the saved calibration's true mm have it (both share the camera)."""
    px = fit.desk_to_pixel @ [desk_xy[0], desk_xy[1], 0.0, 1.0]
    q = np.linalg.solve(saved.true_to_pixel[:, [0, 1, 3]], px)
    return q[:2] / q[2]


for label, fit in (("the evening's, as saved (6 numbers)", saved), ("its points, refined by the new code", new)):
    misses = []
    for name, (x, y, z) in MARKS.items():
        lib = np.array(raw._true_xy(x, y, max(z - 44.0, 0.0)))
        misses.append(np.linalg.norm(in_saved_true(fit, lib) - found[name]))
    axis = in_saved_true(fit, (0.0, 0.0))
    print(f"{label}: circles from where the arm put the cup, typ {np.median(misses):.2f}, worst {max(misses):.2f} mm;"
          f" its arcs' centre {np.linalg.norm(axis - axis_true):.2f} mm from the axis the circles fit")
    print("    " + ", ".join(f"{n} {m:.1f}" for n, m in zip(MARKS, misses)))

print("\nthe circles' misses, the new code's calibration, split: out = away from the axis, round = towards +bearing")
for name, (x, y, z) in MARKS.items():
    lib = np.array(raw._true_xy(x, y, max(z - 44.0, 0.0)))
    predicted = in_saved_true(new, lib)
    miss_true = found[name] - predicted
    # directions in the saved true frame: radial from the circles' axis
    radial = predicted - in_saved_true(new, (0.0, 0.0))
    radial /= np.linalg.norm(radial)
    tangent = np.array([-radial[1], radial[0]])
    # which sense of tangent is +bearing: compare with a point a little further round in desk terms
    b = math.atan2(lib[0], -lib[1])
    ahead = in_saved_true(new, (math.hypot(*lib) * math.sin(b + 0.01), -math.hypot(*lib) * math.cos(b + 0.01)))
    if (ahead - predicted) @ tangent < 0:
        tangent = -tangent
    print(f"  {name}: bearing {math.degrees(b):6.1f}, reach {math.hypot(*lib):5.1f}: out {miss_true @ radial:+5.2f}, "
          f"round {miss_true @ tangent:+5.2f} mm")
