"""The camera fitted to the Fine Tune points alone: each point pairs the arm's x, y (the cup over the cube's centre)
with the pixel under the cube's centre on the desk. Everything about the camera free (focal length, principal point
held loosely to the frame's middle as fit.py holds it, where it stands, which way it looks), in the arm's own
x, y; the three cubes not used, only checked. Each point predicted from a fit without it. Also the desk alone as a
plane-to-picture map (8 numbers), for comparison. No arm; config/ only read.

    ../.venv/bin/python agenttools/fit-view/points-camera.py
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]
from camera import refine, store  # noqa: E402
from camera.fit import CENTRE_SPREAD, HANDEDNESS  # noqa: E402
from camera.model import Pinhole, least_squares  # noqa: E402

config = ROOT / "config"
saved = store.load_camera(config)
points = store.load_refine_points(config)
cubes = refine.cube_sightings(saved)
width, height = saved.frame_size
centre = np.array([width / 2, height / 2])


def table(sightings):
    return (np.array([(s.x, s.y) for s in sightings]), np.array([s.pixel for s in sightings], float))


arm, pixels = table([p.sighting for p in points])
cube_arm, cube_pixels = table(cubes)
labels = [str(p.id) for p in points]

# The start: the saved camera, in the arm's x, y through the board's mirror.
intrinsic, rotation, position = cv2.decomposeProjectionMatrix(saved.true_to_pixel)[:3]
intrinsic = intrinsic / intrinsic[2, 2]
position = position[:3, 0] / position[3, 0]
start = np.concatenate(([intrinsic[0, 0]], intrinsic[:2, 2], cv2.Rodrigues(rotation)[0].ravel(),
                        -rotation @ position))
STEPS = np.array([0.5, 0.25, 0.25] + [1e-5] * 3 + [0.01] * 3)


def matrix(params):
    """3x4: desk (x, y, z) to pixels, the arm's x, y taken as they are."""
    lift = np.eye(4)
    lift[:2, :2] = HANDEDNESS
    return Pinhole(params[0], tuple(params[1:3]), params[3:6], params[6:9]).matrix @ lift


def on_desk(m, pixel):
    x, y, w = np.linalg.solve(m[:, [0, 1, 3]], (*pixel, 1.0))
    return np.array([x / w, y / w])


def misses(params, xy, px):
    m = matrix(params)
    return np.array([on_desk(m, p) for p in px]) - xy


def solve(xy, px, first=start):
    def residuals(params):
        return np.concatenate([misses(params, xy, px).ravel(), (params[1:3] - centre) / (CENTRE_SPREAD * width)])
    return least_squares(residuals, first, STEPS, huber=1e9, iterations=200)[0]


def homography_misses(xy_fit, px_fit, xy, px):
    to_desk, _ = cv2.findHomography(px_fit, xy_fit, 0)
    mapped = cv2.perspectiveTransform(px.reshape(-1, 1, 2), to_desk).reshape(-1, 2)
    return mapped - xy


def report(name, together, cubes_off, alone):
    print(f"{name}: the 15 together rms {math.sqrt(np.mean(together ** 2)):.2f}, worst {together.max():.2f} | each "
          f"from the others median {np.median(alone):.2f}, rms {math.sqrt(np.mean(alone ** 2)):.2f}, worst "
          f"{alone.max():.2f} | the cubes, not used: {' '.join(f'{v:.2f}' for v in cubes_off)}")
    print("   from the others: " + " ".join(f"{l}:{v:.2f}" for l, v in zip(labels, alone)))


# Today's: SAVE's turn and shift, the camera as the cubes' shapes made it.
now = np.linalg.norm([refine.seen_desk(saved, p.sighting) - (p.x, p.y) for p in points], axis=1)
now_cubes = np.linalg.norm([refine.seen_desk(saved, c) - (c.x, c.y) for c in cubes], axis=1)
print(f"SAVE now (the 3 cubes in it): the 15 rms {math.sqrt(np.mean(now ** 2)):.2f}, worst {now.max():.2f}; cubes "
      f"{' '.join(f'{v:.2f}' for v in now_cubes)}")

params = solve(arm, pixels)
together = np.linalg.norm(misses(params, arm, pixels), axis=1)
cubes_off = np.linalg.norm(misses(params, cube_arm, cube_pixels), axis=1)
alone = np.array([np.linalg.norm(misses(solve(np.delete(arm, k, 0), np.delete(pixels, k, 0), params),
                                        arm[k:k + 1], pixels[k:k + 1])) for k in range(len(arm))])
pinhole = Pinhole(params[0], tuple(params[1:3]), params[3:6], params[6:9])
print(f"the camera from the points: focal {params[0]:.0f} px (the cubes' {start[0]:.0f}), principal point "
      f"({params[1]:.0f}, {params[2]:.0f}), stands {pinhole.position[2]:.0f} mm up")
report("camera from the 15 points", together, cubes_off, alone)

together = np.linalg.norm(homography_misses(arm, pixels, arm, pixels), axis=1)
cubes_off = np.linalg.norm(homography_misses(arm, pixels, cube_arm, cube_pixels), axis=1)
alone = np.array([np.linalg.norm(homography_misses(np.delete(arm, k, 0), np.delete(pixels, k, 0), arm[k:k + 1],
                                                   pixels[k:k + 1])) for k in range(len(arm))])
report("desk plane map (8 numbers)", together, cubes_off, alone)
