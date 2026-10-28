"""The evening's camera fitted to its cubes' edges and to the owner's ruler at once: the five distances between
gecko's pencil circles (their pixels in the calibration frame), each as close to the ruler's as the cubes allow.
Then: the cube edges, the ruler, and the evening's 19 points (turn and shift only, each from the others). No arm.

python ruler-camera.py weight_px_per_mm
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration"), str(Path(__file__).resolve().parent)]
from camera import cubes, edges, fit as fitmod  # noqa: E402
from camera.fit import CENTRE, CENTRE_SPREAD, EDGE_HUBER_PX, REACH, STEPS  # noqa: E402
from camera.model import least_squares, rescaled  # noqa: E402
from points import load  # noqa: E402

weight = float(sys.argv[1])
config = ROOT.parent / "work" / "config-old"
frame = cv2.imread(str(config / "camera-frame.png"))
points = {c: (v["x"], v["y"]) for c, v in json.loads((config / "camera-points.json").read_text())["cubes"].items()}
old_P = np.array(json.loads((config / "camera.json").read_text())["true_to_pixel"])
circles = np.load(Path(__file__).resolve().parent / "circles-true.npy", allow_pickle=True).item()["found"]
circle_px = {n: (lambda h: h[:2] / h[2])(old_P[:, [0, 1, 3]] @ (*c, 1.0)) for n, c in circles.items()}
RULER = {("A", "B"): 177, ("B", "C"): 164, ("C", "D"): 165, ("A", "C"): 294, ("B", "D"): 320.25}

blobs = cubes.find_blobs(frame)
starts = [(*fitmod._start(frame, naming, points), naming) for naming in cubes.namings(blobs)]
_, params, naming = max(starts, key=lambda s: s[0])
params = fitmod._fit_outlines(naming, points, params)
params, _ = fitmod._fit_edges(frame, naming, points, params)
desk = cubes.desk_chroma(frame)
crops = [edges.crop(frame, naming[c], desk) for c in fitmod.COLOURS]
centre = np.array([frame.shape[1] / 2, frame.shape[0] / 2])


def on_desk(P, pixel):
    q = np.linalg.solve(P[:, [0, 1, 3]], (*pixel, 1.0))
    return q[:2] / q[2]


def ruler_misses(trial):
    P = fitmod._pinhole(trial).matrix
    return np.array([np.linalg.norm(on_desk(P, circle_px[a]) - on_desk(P, circle_px[b])) - r
                     for (a, b), r in RULER.items()])


for reach in REACH + REACH:
    found = [edges.search(crop, corners, seen, reach) for crop, corners, seen in
             zip(crops, fitmod._corners(params, points), fitmod._edges_seen(params, points))]

    def residuals(trial):
        parts = [edges.signed_distances(c, *f) for c, f in zip(fitmod._corners(trial, points), found)]
        parts.append((trial[CENTRE] - centre) / (CENTRE_SPREAD * frame.shape[1]))
        parts.append(weight * ruler_misses(trial))
        return np.concatenate(parts)

    params, res, _ = least_squares(residuals, params, STEPS, huber=EDGE_HUBER_PX, iterations=30)
edge = np.abs(np.concatenate([edges.signed_distances(c, *f) for c, f in zip(fitmod._corners(params, points), found)]))
P = fitmod._pinhole(params).matrix
K = cv2.decomposeProjectionMatrix(P)[0]
K /= K[2, 2]
print(f"weight {weight}: focal {K[0, 0]:.0f}, centre ({K[0, 2]:.0f}, {K[1, 2]:.0f}); cube edges median "
      f"{np.median(edge):.2f} px, 90% {np.percentile(edge, 90):.2f}; ruler misses {np.round(ruler_misses(params), 1)}")
_, rows = load(config)
centres = fitmod._centres(params, points)
cam = [np.array(c) for c in centres] + [
    on_desk(P if tuple(r["sight"].frame_size) == (2592, 1944) else rescaled(P, (2592, 1944),
            tuple(r["sight"].frame_size)), r["sight"].pixel) for r in rows[3:]]
cam = np.array(cam)
arm = np.array([(r["x"], r["y"]) for r in rows])
keys = [(round(x, 1), round(y, 1)) for x, y in arm]


def rigid(src, dst):
    s = (src * [-1, 1]) @ [1, 1j]
    d = dst @ [1, 1j]
    a = np.vdot(s - s.mean(), d - d.mean())
    a /= abs(a)
    b = d.mean() - a * s.mean()
    return lambda p: (lambda z: np.column_stack((z.real, z.imag)))(a * ((p * [-1, 1]) @ [1, 1j]) + b)


loo = []
for key in set(keys):
    g = [k for k, kk in enumerate(keys) if kk == key]
    rest = [k for k in range(len(keys)) if k not in g]
    loo += list(np.linalg.norm(rigid(arm[rest], cam[rest])(arm[g]) - cam[g], axis=1))
own = np.linalg.norm(rigid(arm, cam)(arm) - cam, axis=1)
print(f"   the 19 points, turn and shift: fitted typ {np.median(own):.2f} worst {own.max():.2f}; each from the others "
      f"typ {np.median(loo):.2f} worst {max(loo):.2f}")

# gecko's circles at desk level through this camera, against the arm's points placed by the cubes' points
from maxarm import raw  # noqa: E402
MARKS = {"A": (-201, -1, 46), "B": (-160, -161, 45), "C": (0, -201, 44), "D": (160, -160, 44), "1": (-181, -80, 44),
         "2": (-80, -181, 44), "3": (79, -180, 44), "4": (181, -81, 44), "5": (201, 1, 44), "6": (-159, -1, 44),
         "7": (-260, -1, 44)}
place = rigid(arm, cam)
axis = place(np.zeros((1, 2)))[0]
print("   circles at desk level: out = further from the axis than the arm says; round = toward +bearing")
outs = []
for name, (x, y, z) in MARKS.items():
    lib = np.array(raw._true_xy(x, y, max(z - 44.0, 0.0)))
    predicted = place(lib[None])[0]
    seen = on_desk(P, circle_px[name])
    radial = (predicted - axis) / np.linalg.norm(predicted - axis)
    tangent = np.array([-radial[1], radial[0]])
    ahead = place(np.array([[lib[0] + 0.01 * -lib[1], lib[1] + 0.01 * lib[0]]]))[0]
    if (ahead - predicted) @ tangent < 0:
        tangent = -tangent
    miss = seen - predicted
    outs.append((np.hypot(*lib), miss @ radial))
    print(f"     {name}: reach {np.hypot(*lib):5.1f}  out {miss @ radial:+5.2f}  round {miss @ tangent:+5.2f}")
r, o = np.array(outs).T
slope, at207 = np.polyfit(r - 207, o, 1)
print(f"   out = {at207:+.2f} {slope:+.4f} x (reach - 207)")
