#!/usr/bin/env python3
"""One camera fitted to every point's cube outline at once (40 mm the only length, no arm numbers), against
CALIBRATE's three-cube camera; each judged by how the arm's formula readings lay on the cubes it sees.

    python agenttools/calcheck/joint-camera.py [config dir]      # default: config/

Prints each camera's focal length, principal point and place, and for each the arm laid on its cubes by a turn,
shift and one scale (and by a turn and shift alone): the scale, and the cubes' misses, mm. Also both with the
principal point held at the frame's center. STATUS-kiwi §8.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from applib.camera import store                                      # noqa: E402
from applib.camera.fit import EDGE_HUBER_PX                          # noqa: E402
from applib.camera.model import CUBE_MM, Pinhole, least_squares      # noqa: E402
from applib.camera.sighting import misses, on_desk, place, start  # noqa: F401     # noqa: E402
from maxarm import raw                                                # noqa: E402

CAMERA_STEPS = [0.05, 0.05, 0.05, 1e-5, 1e-5, 1e-5, 0.01, 0.01, 0.01]
CUBE_STEPS = [0.01, 0.01, 1e-5]


def to_pinhole(matrix: np.ndarray) -> np.ndarray:
    """A 3x4 camera as the joint fit's numbers: focal, principal point, Rodrigues turn, shift."""
    intrinsic = cv2.RQDecomp3x3(matrix[:, :3])[1]
    scale = intrinsic[2, 2]
    intrinsic, rest = intrinsic / scale, matrix / scale
    rotation = np.linalg.solve(intrinsic, rest[:, :3])
    shift = np.linalg.solve(intrinsic, rest[:, 3])
    if np.linalg.det(rotation) < 0:
        rotation, shift = -rotation, -shift
    focal = (intrinsic[0, 0] + intrinsic[1, 1]) / 2
    return np.array([focal, intrinsic[0, 2], intrinsic[1, 2], *cv2.Rodrigues(rotation)[0].ravel(), *shift])


def camera_of(numbers: np.ndarray) -> np.ndarray:
    return Pinhole(numbers[0], (numbers[1], numbers[2]), numbers[3:6], numbers[6:9]).matrix


def joint_fit(matrix: np.ndarray, sightings: list, center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """One camera and every cube on all the sightings' edges, from `matrix`; the principal point held at `center`
    if given."""
    cubes = [place(matrix, s, start(matrix, s)[:3]) for s in sightings]
    numbers = np.concatenate([to_pinhole(matrix), *cubes])
    held = []
    if center is not None:
        numbers[1:3], held = center, [1, 2]

    def residuals(trial: np.ndarray) -> np.ndarray:
        camera = camera_of(trial[:9])
        return np.concatenate([misses(camera, s, np.append(trial[9 + 3 * i:12 + 3 * i], CUBE_MM))
                               for i, s in enumerate(sightings)])

    steps = np.array(CAMERA_STEPS + CUBE_STEPS * len(sightings))
    free = [i for i in range(len(numbers)) if i not in held]
    solution, left, _ = least_squares(residuals, numbers, steps, huber=EDGE_HUBER_PX, free=free, iterations=60)
    print(f"  joint fit: edges {np.median(np.abs(left)):.2f} px median, {np.percentile(np.abs(left), 90):.2f} at 90%")
    return solution


def similarity(source: np.ndarray, target: np.ndarray, is_scaled: bool):
    """Turn (mirror allowed), shift and, if `is_scaled`, one scale laying `source` rows on `target` rows."""
    a, b = source - source.mean(0), target - target.mean(0)
    u, s, vt = np.linalg.svd(a.T @ b)
    rotation = (u @ vt).T
    scale = s.sum() / (a ** 2).sum() if is_scaled else 1.0
    laid = scale * a @ rotation.T + target.mean(0)
    return scale, np.linalg.norm(laid - target, axis=1)


def describe(name: str, matrix: np.ndarray, sightings: list, readings: np.ndarray) -> None:
    numbers = to_pinhole(matrix)
    position = -cv2.Rodrigues(numbers[3:6])[0].T @ numbers[6:9]
    seen = np.array([place(matrix, s, start(matrix, s)[:3])[:2] if s.edges is not None else on_desk(matrix, s)
                     for s in sightings])
    scale, scaled = similarity(readings, seen, True)
    _, rigid = similarity(readings, seen, False)
    print(f"{name}: f {numbers[0]:.1f} px, center ({numbers[1]:.1f}, {numbers[2]:.1f}), "
          f"{np.linalg.norm(position):.1f} mm away, {position[2]:.1f} mm up")
    print(f"  the arm laid by turn+shift: cubes {np.median(rigid):.2f} mm median, {rigid.max():.2f} worst")
    print(f"  by turn+shift+scale: scale {scale:.4f} (camera mm per arm mm), cubes {np.median(scaled):.2f} median, "
          f"{scaled.max():.2f} worst")


def main() -> None:
    config_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "config"
    fit = store.load_camera(config_dir)
    points = [point for point in store.load_refine_points(config_dir, fit) if point.edges and point.raw]
    sightings = [point.sighting for point in points]
    readings = np.array([raw.formula_true(tuple(point.raw))[:2] for point in points])
    alone = fit.cube_fit().true_to_pixel
    print(f"{config_dir}: {len(points)} points with edges and raw readings")
    describe("CALIBRATE's three-cube camera", alone, sightings, readings)
    joint = camera_of(joint_fit(alone, sightings)[:9])
    describe("one camera on every point's edges", joint, sightings, readings)
    width, height = fit.frame_size
    middle = ((width - 1) / 2, (height - 1) / 2)
    held = camera_of(joint_fit(alone, sightings, center=middle)[:9])
    describe("the same, its principal point held at the frame's center", held, sightings, readings)
    three = camera_of(joint_fit(alone, sightings[:3], center=middle)[:9])
    describe("CALIBRATE's three cubes, principal point at the center", three, sightings, readings)


if __name__ == "__main__":
    main()
