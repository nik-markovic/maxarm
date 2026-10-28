"""The camera fitted to the cubes' edges and the owner's red cube points together: the arm's x, y placed on the desk
by a turn, scale and shift (and the board's mirror), the camera free. Then each place predicted from a fit without
it. No arm.

python joint.py config-dir [weight_px_per_mm]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration"), str(Path(__file__).resolve().parent)]
from camera import cubes, edges, fit as fitmod  # noqa: E402
from camera.fit import CENTRE, CENTRE_SPREAD, EDGE_HUBER_PX, FOCAL, REACH, STEPS  # noqa: E402
from camera.model import least_squares, rescaled  # noqa: E402
from points import load  # noqa: E402

config = Path(sys.argv[1])
weight = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0      # pixels of edge per mm of point miss
frame = cv2.imread(str(config / "camera-frame.png"))
pts_json = json.loads((config / "camera-points.json").read_text())["cubes"]
points = {c: (v["x"], v["y"]) for c, v in pts_json.items()}
fit0, rows = load(config)
arm = np.array([(r["x"], r["y"]) for r in rows])
sights = [r["sight"] for r in rows]

blobs = cubes.find_blobs(frame)
starts = [(*fitmod._start(frame, naming, points), naming) for naming in cubes.namings(blobs)]
overlap, params0, naming = max(starts, key=lambda s: s[0])
params0 = fitmod._fit_outlines(naming, points, params0)
params0, dist0 = fitmod._fit_edges(frame, naming, points, params0)
desk = cubes.desk_chroma(frame)
crops = [edges.crop(frame, naming[c], desk) for c in fitmod.COLOURS]
centre = np.array([frame.shape[1] / 2, frame.shape[0] / 2])
found = [edges.search(crop, corners, seen, REACH[-1]) for crop, corners, seen in
         zip(crops, fitmod._corners(params0, points), fitmod._edges_seen(params0, points))]


def seen_true(P, s):
    M = P if tuple(s.frame_size) == (2592, 1944) else rescaled(P, (2592, 1944), tuple(s.frame_size))
    x, y, w = np.linalg.solve(M[:, [0, 1, 3]], (*s.pixel, 1.0))
    return np.array([x / w, y / w])


# extra parameters: the arm's placement, a similarity: [a, b, tx, ty] with true = (a + ib) * mirror(desk) + t
def unpack(params):
    return params[:15], params[15:]


def arm_to_true(sim, xy):
    a, b, tx, ty = sim
    z = (a + 1j * b) * ((np.atleast_2d(xy) * [-1, 1]) @ [1, 1j]) + (tx + 1j * ty)
    return np.column_stack((z.real, z.imag))


def residuals(params, use):
    cam, sim = unpack(params)
    # the cubes' own positions stay where the camera's stretch puts them (as fit.py); edges as fit.py
    parts = [edges.signed_distances(corners, *f) for corners, f in zip(fitmod._corners(cam, points), found)]
    parts.append((cam[CENTRE] - centre) / (CENTRE_SPREAD * frame.shape[1]))
    P = fitmod._pinhole(cam).matrix
    seen = np.array([seen_true(P, sights[k]) for k in use])
    miss = (arm_to_true(sim, arm[use]) - seen).ravel() * weight
    return np.concatenate(parts + [miss])


def solve(use, start):
    steps = np.concatenate((STEPS, [1e-5, 1e-5, 0.01, 0.01]))
    sol, res, _ = least_squares(lambda p: residuals(p, use), start, steps, huber=EDGE_HUBER_PX, iterations=40)
    return sol


# start: the arm placed by a similarity onto the cubes' camera
P0 = fitmod._pinhole(params0).matrix
seen0 = np.array([seen_true(P0, s) for s in sights])
s = (arm * [-1, 1]) @ [1, 1j]
d = seen0 @ [1, 1j]
(ab, t), *_ = np.linalg.lstsq(np.column_stack((s, np.ones(len(s)))), d, rcond=None)
start = np.concatenate((params0, [ab.real, ab.imag, t.real, t.imag]))
every = list(range(len(rows)))
sol = solve(every, start)
cam, sim = unpack(sol)
P = fitmod._pinhole(cam).matrix
seen = np.array([seen_true(P, s) for s in sights])
miss = np.linalg.norm(arm_to_true(sim, arm) - seen, axis=1)
edge = np.abs(np.concatenate([edges.signed_distances(c, *f) for c, f in zip(fitmod._corners(cam, points), found)]))
edge0 = np.abs(np.concatenate([edges.signed_distances(c, *f) for c, f in zip(fitmod._corners(params0, points), found)]))
print(f"cubes alone:   focal {params0[FOCAL]:.0f} px, edges median {np.median(edge0):.2f} px, 90% {np.percentile(edge0, 90):.2f}")
print(f"joint (w={weight}): focal {cam[FOCAL]:.0f} px, edges median {np.median(edge):.2f} px, 90% {np.percentile(edge, 90):.2f}; "
      f"points typ {np.median(miss):.2f} worst {miss.max():.2f} mm; arm scale {abs(complex(sim[0], sim[1])):.4f}")
places = {}
for k, p in enumerate(arm):
    places.setdefault((round(p[0], 1), round(p[1], 1)), []).append(k)
loo = np.zeros(len(rows))
for g in places.values():
    use = [k for k in every if k not in g]
    so = solve(use, sol)
    c_, s_ = unpack(so)
    P_ = fitmod._pinhole(c_).matrix
    for k in g:
        loo[k] = np.linalg.norm(arm_to_true(s_, arm[k])[0] - seen_true(P_, sights[k]))
print(f"each place from the others: typ {np.median(loo):.2f}, 90% {np.percentile(loo, 90):.2f}, worst {loo.max():.2f} mm")
