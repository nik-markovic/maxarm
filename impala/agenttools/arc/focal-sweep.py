"""The cubes' camera refitted on camera-frame.png with the focal length held, at several values; for each, how
well the cubes' edges fit and how far the owner's 17 points then sit from the arm's readings. No arm.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration"), str(Path(__file__).resolve().parent)]
import cv2  # noqa: E402
from camera import cubes, edges, fit as fitmod  # noqa: E402
from camera.fit import CENTRE, CENTRE_SPREAD, EDGE_HUBER_PX, FOCAL, REACH, STEPS  # noqa: E402
from camera.model import least_squares  # noqa: E402
from camera.refine import seen_true  # noqa: E402
from points import fit_affine, load  # noqa: E402

frame = cv2.imread(str(ROOT / "config/camera-frame.png"))
points_json = json.loads((ROOT / "config/camera-points.json").read_text())["cubes"]
points = {c: (v["x"], v["y"]) for c, v in points_json.items()}
_, rows = load()
arm = np.array([(r["x"], r["y"]) for r in rows])

blobs = cubes.find_blobs(frame)
starts = [(*fitmod._start(frame, naming, points), naming) for naming in cubes.namings(blobs)]
overlap, params0, naming = max(starts, key=lambda s: s[0])
params0 = fitmod._fit_outlines(naming, points, params0)
params_free, dist_free = fitmod._fit_edges(frame, naming, points, params0)
print(f"focal free: {params_free[FOCAL]:.1f} px")


def fit_edges_fixed(params, free):
    desk = cubes.desk_chroma(frame)
    crops = [edges.crop(frame, naming[c], desk) for c in fitmod.COLOURS]
    centre = np.array([frame.shape[1] / 2, frame.shape[0] / 2])
    for reach in REACH:
        found = [edges.search(crop, corners, seen, reach) for crop, corners, seen in
                 zip(crops, fitmod._corners(params, points), fitmod._edges_seen(params, points))]

        def residuals(trial):
            parts = [edges.signed_distances(corners, *f) for corners, f in zip(fitmod._corners(trial, points), found)]
            parts.append((trial[CENTRE] - centre) / (CENTRE_SPREAD * frame.shape[1]))
            return np.concatenate(parts)

        params, dist, _ = least_squares(residuals, params, STEPS, huber=EDGE_HUBER_PX, free=free, iterations=30)
    return params, dist[:-2]


def sighted_true(params, row):
    P = fitmod._pinhole(params).matrix
    fake = fitmod.CameraFit(P, fitmod._desk_to_true(params), {}, (2592, 1944), (0, 0), 0)
    return seen_true(fake, row["sight"])


for focal in (1840, 1880, 1916, 1960, 2000, 2040, 2080):
    p = params_free.copy()
    # keep the scale at the cubes: move the camera along its axis in proportion
    p[FOCAL] = focal
    p[fitmod.SHIFT] = params_free[fitmod.SHIFT] * focal / params_free[FOCAL]
    free = [k for k in range(len(p)) if k != FOCAL]
    p, dist = fit_edges_fixed(p, free)
    cam = np.array([sighted_true(p, r) for r in rows])
    res = {}
    for kind in ("rigid", "similarity", "affine"):
        mapped, prm = fit_affine(arm, cam, kind)
        e = np.linalg.norm(mapped - cam, axis=1)
        res[kind] = (np.median(e), e.max(), prm[1] if kind != "affine" else None)
    pos = fitmod._pinhole(p).position
    print(f"focal {focal}: edges median {np.median(np.abs(dist)):.3f} px, 90% {np.percentile(np.abs(dist), 90):.2f} | "
          f"camera {pos[2]:.0f} mm up | points rigid {res['rigid'][0]:.2f}/{res['rigid'][1]:.2f}  "
          f"similarity {res['similarity'][0]:.2f}/{res['similarity'][1]:.2f} (scale {res['similarity'][2]:.4f})  "
          f"affine {res['affine'][0]:.2f}/{res['affine'][1]:.2f}")
