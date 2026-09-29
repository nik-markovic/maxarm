#!/usr/bin/env python3
"""Refine condor's camera with the tiles in a frame: a real camera, and a metric desk.

condor solves its camera from six cube points by DLT, which is free to give the
camera non-square pixels. With the arm placing the cubes to +-2.6 mm, it used
that freedom: 1994 px across, 1895 down, a 5% aspect no sensor has. The top-down
view inherited it as a shear -- tiles fitted as parallelograms lean up to 5
degrees, and an O comes out as an ellipse tilted 20 degrees.

This fits a camera that could exist -- one focal length, square pixels, no skew
-- to two kinds of evidence at once:

- **The tiles' shape.** Each is a rigid 17.5 x 20 x 4 mm box. Its trusted edges
  (`tiles/pose.py`) are scored where the camera projects them into the frame,
  with every tile's position and turn fitted alongside.
- **Where the arm put the cubes**, from the config. They carry the arm's error,
  so they are weighed at that: a cube 2.6 mm off costs what a few percent of
  tile edge agreement does. Flat tiles leave the camera's tilt and principal
  point trading against each other; the cubes, which have height, settle that.

Lens distortion was fitted and is negligible here (k1 = +0.007, under half a
pixel where tiles lie), so the camera stays a 3x4 and the config keeps its form.

    ./calibration/refine.py files/tiles-1.png            # rewrite config/calibration.json
    ./calibration/refine.py files/tiles-1.png --dry-run  # report only
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "tiles")]

import cv2                              # noqa: E402
import numpy as np                      # noqa: E402

import pose                             # noqa: E402
import read                             # noqa: E402
import scene                            # noqa: E402
from mapping import DeskMapping, _plane_of  # noqa: E402

CALIBRATION = ROOT / "config" / "calibration.json"
# The arm's placement error, per axis (bison/condor: calibrate-readback.py).
ARM_ERROR_MM = 2.6
# A cube one arm-error off costs this share of the tiles' edge agreement.
CUBE_WEIGHT = 0.02
# Coordinate-descent steps: f, cx, cy in pixels; rotation in radians; translation in mm.
CAMERA_STEPS = (8.0, 8.0, 8.0, 0.002, 0.002, 0.002, 1.0, 1.0, 1.0)
TILE_STEPS = (0.2, 0.2, np.radians(0.3))
ROUNDS = 8


class FrameEdges:
    """The frame's colour structure tensor: gradient across any direction, in closed form."""

    def __init__(self, frame: np.ndarray):
        lab = cv2.cvtColor(cv2.GaussianBlur(frame, (0, 0), 1.0), cv2.COLOR_BGR2LAB).astype(np.float32)
        self.tensor = np.zeros(frame.shape[:2] + (3,), np.float32)
        for channel in range(3):
            gx = cv2.Scharr(lab[..., channel], cv2.CV_32F, 1, 0) / 32
            gy = cv2.Scharr(lab[..., channel], cv2.CV_32F, 0, 1) / 32
            self.tensor[..., 0] += gx * gx
            self.tensor[..., 1] += gx * gy
            self.tensor[..., 2] += gy * gy

    def across(self, pixels: np.ndarray, normals: np.ndarray) -> np.ndarray:
        t = pose._bilinear(self.tensor, pixels[:, 0], pixels[:, 1])
        nx, ny = normals[:, 0], normals[:, 1]
        return np.sqrt(np.maximum(nx * nx * t[:, 0] + 2 * nx * ny * t[:, 1] + ny * ny * t[:, 2], 0))


class Camera:
    """K [R | t], one focal length. params: f, cx, cy, rotation vector (3), translation (3)."""

    def __init__(self, params):
        self.params = np.asarray(params, float)

    @property
    def matrix(self) -> np.ndarray:
        f, cx, cy = self.params[:3]
        rotation, _ = cv2.Rodrigues(self.params[3:6])
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]) @ np.column_stack((rotation, self.params[6:]))

    @property
    def centre(self) -> np.ndarray:
        rotation, _ = cv2.Rodrigues(self.params[3:6])
        return -rotation.T @ self.params[6:]

    def project(self, world: np.ndarray) -> np.ndarray:
        homogeneous = np.column_stack((world, np.ones(len(world)))) @ self.matrix.T
        return homogeneous[:, :2] / homogeneous[:, 2:]

    def to_plane(self, pixels: np.ndarray, height: float) -> np.ndarray:
        ground = np.linalg.inv(_plane_of(self.matrix, height)) @ np.column_stack(
            (pixels, np.ones(len(pixels)))).T
        return (ground[:2] / ground[2]).T

    @classmethod
    def nearest_to(cls, matrix: np.ndarray) -> "Camera":
        """The square-pixel camera closest to a 3x4: its focal lengths averaged."""
        k, rotation, centre = cv2.decomposeProjectionMatrix(matrix / np.linalg.norm(matrix[2, :3]))[:3]
        k = k / k[2, 2]
        rvec, _ = cv2.Rodrigues(rotation)
        tvec = -rotation @ (centre[:3] / centre[3]).ravel()
        return cls([(k[0, 0] + k[1, 1]) / 2, k[0, 2], k[1, 2], *rvec.ravel(), *tvec])


def edge_score(edges: FrameEdges, camera: Camera, tile) -> float:
    """How well a tile at (x, y, baseline angle) meets the frame's edges, trusted edges only."""
    x, y, angle = tile
    corners = pose.tile_corners(np.array((x, y)), angle)
    points, normals, weights = [], [], []
    along = np.linspace(0.1, 0.9, 16)
    for index in range(4):
        start, end = corners[index], corners[(index + 1) % 4]
        direction = (end - start) / np.linalg.norm(end - start)
        normal = np.array([direction[1], -direction[0]])
        if np.dot(normal, np.array((x, y)) - start) > 0:
            normal = -normal
        # Facing away from the camera: the top face's own edge. Facing it: the
        # bottom edge, where the side the camera sees meets the desk.
        is_far = np.dot(normal, camera.centre[:2] - (start + end) / 2) <= 0
        edge = start + np.outer(along, end - start)
        points.append(np.column_stack((edge, np.full(len(along), pose.TILE_THICKNESS_MM if is_far else 0.0))))
        normals.append(np.repeat(np.append(normal, 0.0)[None], len(along), axis=0))
        weights.append(np.full(len(along), 1.0 if is_far else pose.NEAR_EDGE_WEIGHT))
    points, normals, weights = np.vstack(points), np.vstack(normals), np.concatenate(weights)
    pixels = camera.project(points)
    direction = camera.project(points + 0.2 * normals) - pixels
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return float((edges.across(pixels, direction) * weights).sum() / weights.sum())


def carried(old: Camera, new: Camera, tile) -> np.ndarray:
    """The same tile under another camera, kept where it is in the frame.

    Without this, every trial camera moves every tile a millimetre and scores
    worse for it, and no change to the camera is ever accepted. Carried, a trial
    changes only the tiles' shapes, which is what they can judge.
    """
    x, y, angle = tile
    ahead = np.array((x, y)) + 5.0 * np.array((np.cos(angle), np.sin(angle)))
    pixels = old.project(np.array([[x, y, pose.TILE_THICKNESS_MM], [*ahead, pose.TILE_THICKNESS_MM]]))
    ground = new.to_plane(pixels, pose.TILE_THICKNESS_MM)
    return np.array((*ground[0], np.arctan2(*(ground[1] - ground[0])[::-1])))


def cube_misses_mm(camera: Camera, mapping: DeskMapping) -> np.ndarray:
    """How far each cube feature lands, through this camera, from where the arm put it."""
    return np.array([np.hypot(*(camera.to_plane(np.array([obs.pixel]), obs.height_mm)[0] - obs.ground))
                     for obs in mapping.observations])


def descend(value_of, start, steps, rounds):
    best, value, steps = np.array(start, float), value_of(start), np.array(steps, float)
    for _ in range(rounds):
        improved = False
        for axis in range(len(best)):
            for sign in (1, -1):
                trial = best.copy()
                trial[axis] += sign * steps[axis]
                trial_value = value_of(trial)
                if trial_value > value:
                    best, value, improved = trial, trial_value, True
        if not improved:
            steps /= 2
    return best, value


def refine(frame: np.ndarray, mapping: DeskMapping, tiles, cube_weight: float = CUBE_WEIGHT):
    edges = FrameEdges(frame)
    camera = Camera.nearest_to(mapping.camera())
    tiles = [carried(Camera.nearest_to(mapping.camera()), camera, t) for t in tiles]
    tiles = [descend(lambda p: edge_score(edges, camera, p), t, TILE_STEPS, 10)[0] for t in tiles]
    reference = np.mean([edge_score(edges, camera, t) for t in tiles])

    def value(params):
        trial = Camera(params)
        agreement = np.mean([edge_score(edges, trial, carried(camera, trial, t)) for t in tiles]) / reference
        misses = cube_misses_mm(trial, mapping) / ARM_ERROR_MM
        return agreement - cube_weight * float(np.mean(misses ** 2))

    for _ in range(ROUNDS):
        tiles = [descend(lambda p: edge_score(edges, camera, p), t, TILE_STEPS, 10)[0] for t in tiles]
        params, _ = descend(value, camera.params, CAMERA_STEPS, 10)
        tiles = [carried(camera, Camera(params), t) for t in tiles]
        camera = Camera(params)
    agreement = np.mean([edge_score(edges, camera, t) for t in tiles]) / reference
    return camera, tiles, agreement


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frame", type=Path, help="a frame with tiles in it, camera untouched since")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    frame = cv2.imread(str(args.frame))
    mapping = DeskMapping.load(CALIBRATION)
    found = scene.find_tiles(frame, mapping, read.Reader())
    start = [(*t.centre_mm, np.radians(t.baseline_deg)) for t in found]
    before = Camera.nearest_to(mapping.camera())
    print(f"{len(found)} tiles; condor's camera misses its cubes by 0 mm by construction")
    camera, _, agreement = refine(frame, mapping, start)
    f, cx, cy = camera.params[:3]
    height, width = frame.shape[:2]
    fov = np.degrees(2 * np.arctan(np.array((width, height, np.hypot(width, height))) / 2 / f))
    print(f"camera: f {f:.0f} px (was {before.params[0]:.0f} averaged), principal point ({cx:.0f}, {cy:.0f}), "
          f"field of view {fov[0]:.1f} x {fov[1]:.1f}, {fov[2]:.1f} diagonal")
    print(f"        at {np.round(camera.centre, 1)} mm AACS; tile edge agreement x{agreement:.3f}")
    print(f"cubes now miss by {np.round(cube_misses_mm(camera, mapping), 1)} mm (arm error {ARM_ERROR_MM})")
    if args.dry_run:
        return 0

    refined = json.loads(CALIBRATION.read_text())
    refined["model"] = "camera, refined with tiles"
    refined["pixel_to_arm"] = np.linalg.inv(_plane_of(camera.matrix, 0.0)).tolist()
    refined["pixel_to_arm_lifted"] = np.linalg.inv(_plane_of(camera.matrix, refined["lift_mm"])).tolist()
    refined["camera"] = {"focal_px": f, "principal_point": [cx, cy],
                         "rotation_vector": camera.params[3:6].tolist(),
                         "translation_mm": camera.params[6:].tolist(), "refined_with": args.frame.name}
    CALIBRATION.write_text(json.dumps(refined, indent=2) + "\n")
    print(f"wrote {CALIBRATION.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
