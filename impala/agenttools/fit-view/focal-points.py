"""Is the 1% the camera's? The cubes' camera refitted on config/camera-frame.png with its focal length held at several
values (the cubes' edges free to follow, as fit.py fits them); through each, the arm placed on the three cubes and
the Fine Tune points by a turn and a shift, as SAVE places it, and each point predicted from the others. No arm;
config/ only read.

    ../.venv/bin/python agenttools/fit-view/focal-points.py
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]
from camera import cubes, edges, store  # noqa: E402
from camera import fit as fitmod  # noqa: E402
from camera.fit import CENTRE, CENTRE_SPREAD, EDGE_HUBER_PX, FOCAL, HANDEDNESS, REACH, STEPS  # noqa: E402
from camera.model import least_squares  # noqa: E402

config = ROOT / "config"
frame = cv2.imread(str(config / "camera-frame.png"))
points = {colour: (point.x, point.y) for colour, point in store.load_points(config).items()}
refine_points = store.load_refine_points(config)

blobs = cubes.find_blobs(frame)
starts = [(*fitmod._start(frame, naming, points), naming) for naming in cubes.namings(blobs)]
_, params0, naming = max(starts, key=lambda s: s[0])
params0 = fitmod._fit_outlines(naming, points, params0)
params_free, _, _ = fitmod._fit_edges(frame, naming, points, params0)
desk = cubes.desk_chroma(frame)
crops = [edges.crop(frame, naming[c], desk) for c in fitmod.COLOURS]
centre = np.array([frame.shape[1] / 2, frame.shape[0] / 2])


def place_arm(desk, true):
    """The turn and shift that lay the arm's desk x, y rows, through the board's mirror, best onto the `true` rows
    (fit.py's until 30 Oct, when the arm got a stretch as well): true = turn @ HANDEDNESS @ desk + shift."""
    mirrored = np.asarray(desk, float) @ HANDEDNESS
    true = np.asarray(true, float)
    desk_mean, true_mean = mirrored.mean(axis=0), true.mean(axis=0)
    left, _, right = np.linalg.svd((true - true_mean).T @ (mirrored - desk_mean))
    turn = left @ np.diag([1.0, np.linalg.det(left @ right)]) @ right
    return turn, true_mean - turn @ desk_mean


def fit_held(focal):
    params = params_free.copy()
    params[fitmod.SHIFT] = params_free[fitmod.SHIFT] * focal / params_free[FOCAL]
    params[FOCAL] = focal
    free = [k for k in range(len(params)) if k != FOCAL]
    for reach in REACH:
        found = [edges.search(crop, corners, seen, reach) for crop, corners, seen in
                 zip(crops, fitmod._corners(params, points), fitmod._edges_seen(params, points))]

        def residuals(trial):
            parts = [edges.signed_distances(c, *f) for c, f in zip(fitmod._corners(trial, points), found)]
            parts.append((trial[CENTRE] - centre) / (CENTRE_SPREAD * frame.shape[1]))
            return np.concatenate(parts)

        params, distances, _ = least_squares(residuals, params, STEPS, huber=EDGE_HUBER_PX, free=free, iterations=30)
    return params, distances[:-2]


def placed_offsets(params, arm, seen):
    """Each point's distance after the best turn and shift, all together; and each predicted from the others."""
    turn, shift = place_arm(arm, seen)
    together = np.linalg.norm(seen - ((arm * [-1, 1]) @ turn.T + shift), axis=1)
    alone = []
    for k in range(len(arm)):
        rest = [i for i in range(len(arm)) if i != k]
        turn, shift = place_arm(arm[rest], seen[rest])
        alone.append(float(np.linalg.norm(seen[k] - (turn @ (arm[k] * [-1, 1]) + shift))))
    return together, np.array(alone)


print(f"focal free: {params_free[FOCAL]:.0f} px")
for focal in (1340, 1370, 1400, 1415, 1429, 1445, 1460, 1490, 1520):
    params, distances = fit_held(focal)
    fitted = fitmod._result(params, points, frame, distances)
    true_to_pixel = fitted.true_to_pixel
    arm, seen = [], []
    for colour, cube in fitted.cubes.items():
        arm.append((cube["x"], cube["y"]))
        seen.append(fitted.seen_centre(colour))
    for point in refine_points:
        x, y, w = np.linalg.solve(true_to_pixel[:, [0, 1, 3]], (*point.pixel, 1.0))
        arm.append((point.x, point.y))
        seen.append((x / w, y / w))
    together, alone = placed_offsets(params, np.array(arm), np.array(seen))
    height = fitmod._pinhole(params).position[2]
    print(f"focal {focal}: camera {height:.0f} mm up; cubes' edges {np.median(np.abs(distances)):.2f} px "
          f"(90% {np.percentile(np.abs(distances), 90):.2f}) | 18 points placed: rms {math.sqrt(np.mean(together**2)):.2f}, "
          f"worst {together.max():.2f}, cubes {' '.join(f'{v:.2f}' for v in together[:3])} | each from the others: "
          f"median {np.median(alone):.2f}, rms {math.sqrt(np.mean(alone**2)):.2f}, worst {alone.max():.2f}")
