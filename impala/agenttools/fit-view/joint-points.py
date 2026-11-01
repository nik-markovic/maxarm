"""The camera fitted to the cubes' edges and every point together (config/: the three cubes, the Fine Tune points),
the arm placed by a turn and a shift only, as SAVE places it; the camera's focal length, principal point and pose
free. Then each point predicted from a fit without it. WEIGHT: pixels of cube edge per mm of point miss. No arm;
config/ only read.

    ../.venv/bin/python agenttools/fit-view/joint-points.py [WEIGHT ...]
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
colours = list(fitmod.COLOURS)
labels = colours + [str(p.id) for p in refine_points]
arm = np.array([points[c] for c in colours] + [(p.x, p.y) for p in refine_points])
pixels = [p.pixel for p in refine_points]

blobs = cubes.find_blobs(frame)
starts = [(*fitmod._start(frame, naming, points), naming) for naming in cubes.namings(blobs)]
_, params0, naming = max(starts, key=lambda s: s[0])
params0 = fitmod._fit_outlines(naming, points, params0)
params0, _, _ = fitmod._fit_edges(frame, naming, points, params0)
desk = cubes.desk_chroma(frame)
crops = [edges.crop(frame, naming[c], desk) for c in colours]
centre = np.array([frame.shape[1] / 2, frame.shape[0] / 2])
found = [edges.search(crop, corners, seen, REACH[-1]) for crop, corners, seen in
         zip(crops, fitmod._corners(params0, points), fitmod._edges_seen(params0, points))]


def place_arm(desk, true):
    """The turn and shift that lay the arm's desk x, y rows, through the board's mirror, best onto the `true` rows
    (fit.py's until 30 Oct, when the arm got a stretch as well): true = turn @ HANDEDNESS @ desk + shift."""
    mirrored = np.asarray(desk, float) @ HANDEDNESS
    true = np.asarray(true, float)
    desk_mean, true_mean = mirrored.mean(axis=0), true.mean(axis=0)
    left, _, right = np.linalg.svd((true - true_mean).T @ (mirrored - desk_mean))
    turn = left @ np.diag([1.0, np.linalg.det(left @ right)]) @ right
    return turn, true_mean - turn @ desk_mean


def seen(cam):
    """Where the camera sees each point, true mm: the cubes where its fit stands them, the points through their pixel."""
    matrix = fitmod._pinhole(cam).matrix
    at = list(fitmod._centres(cam, points))
    for pixel in pixels:
        x, y, w = np.linalg.solve(matrix[:, [0, 1, 3]], (*pixel, 1.0))
        at.append(np.array([x / w, y / w]))
    return np.array(at)


def placed(rigid, xy):
    c, s = math.cos(rigid[0]), math.sin(rigid[0])
    return (np.atleast_2d(xy) * [-1, 1]) @ np.array([[c, -s], [s, c]]).T + rigid[1:3]


def edge_distances(cam):
    return np.concatenate([edges.signed_distances(c, *f) for c, f in zip(fitmod._corners(cam, points), found)])


def solve(use, start, weight):
    def residuals(params):
        cam, rigid = params[:15], params[15:]
        miss = (placed(rigid, arm[use]) - seen(cam)[use]).ravel() * weight
        return np.concatenate([edge_distances(cam), (cam[CENTRE] - centre) / (CENTRE_SPREAD * frame.shape[1]), miss])

    steps = np.concatenate((STEPS, [1e-5, 0.01, 0.01]))
    return least_squares(residuals, start, steps, huber=EDGE_HUBER_PX, iterations=40)[0]


def rigid_start(cam, use):
    turn, shift = place_arm(arm[use], seen(cam)[use])
    return np.array([math.atan2(turn[1, 0], turn[0, 0]), *shift])


every = list(range(len(arm)))
edge0 = np.abs(edge_distances(params0))
print(f"the cubes alone: focal {params0[FOCAL]:.0f} px, edges {np.median(edge0):.2f} px (90% "
      f"{np.percentile(edge0, 90):.2f})")
for weight in [float(value) for value in sys.argv[1:]] or [1.0, 2.0, 4.0]:
    sol = solve(every, np.concatenate((params0, rigid_start(params0, every))), weight)
    cam, rigid = sol[:15], sol[15:]
    miss = np.linalg.norm(placed(rigid, arm) - seen(cam), axis=1)
    edge = np.abs(edge_distances(cam))
    alone = []
    for k in every:
        use = [i for i in every if i != k]
        other = solve(use, sol, weight)
        alone.append(float(np.linalg.norm(placed(other[15:], arm[k])[0] - seen(other[:15])[k])))
    alone = np.array(alone)
    print(f"weight {weight}: focal {cam[FOCAL]:.0f} px, camera {fitmod._pinhole(cam).position[2]:.0f} mm up; edges "
          f"{np.median(edge):.2f} px (90% {np.percentile(edge, 90):.2f}) | 18 points: rms "
          f"{math.sqrt(np.mean(miss ** 2)):.2f}, worst {miss.max():.2f}, cubes {' '.join(f'{v:.2f}' for v in miss[:3])} "
          f"| each from the others: median {np.median(alone):.2f}, rms {math.sqrt(np.mean(alone ** 2)):.2f}, worst "
          f"{alone.max():.2f}")
    print("   each from the others: " + " ".join(f"{label}:{value:.2f}" for label, value in zip(labels, alone)))
