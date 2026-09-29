#!/usr/bin/env python3
"""Fit a physical camera to the tiles, then register it to the arm with the cubes.

condor's camera is a 6-point DLT, free to have non-square pixels, and it took
them: 1994 px across, 1895 down. This fits one focal length, square pixels and
no skew, plus the principal point and the pose, to the trusted edges of every
tile (`tiles/pose.py`) in the original frame -- each tile a rigid 17.5 x 20 x 4
box with its own position and turn. Tiles fix the camera's shape and field of
view; they cannot say where the arm's axes are. The cubes do that, as a turn and
shift in the desk plane fitted through where the arm put them.

    ./agenttools/fit-camera.py files/tiles-1.png     # -> config/camera-fit.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "calibration"), str(ROOT / "tiles")]

import cv2                              # noqa: E402
import numpy as np                      # noqa: E402

import pose                             # noqa: E402
import read                             # noqa: E402
import scene                            # noqa: E402
from mapping import DeskMapping         # noqa: E402


class ImageEdges:
    """The structure tensor of the camera frame itself, as pose.EdgeField keeps for the view."""

    def __init__(self, frame):
        lab = cv2.cvtColor(cv2.GaussianBlur(frame, (0, 0), 1.0), cv2.COLOR_BGR2LAB).astype(np.float32)
        self.tensor = np.zeros(frame.shape[:2] + (3,), np.float32)
        for channel in range(3):
            gx = cv2.Scharr(lab[..., channel], cv2.CV_32F, 1, 0) / 32
            gy = cv2.Scharr(lab[..., channel], cv2.CV_32F, 0, 1) / 32
            self.tensor[..., 0] += gx * gx
            self.tensor[..., 1] += gx * gy
            self.tensor[..., 2] += gy * gy

    def across(self, pixels, normals):
        t = pose._bilinear(self.tensor, pixels[:, 0], pixels[:, 1])
        nx, ny = normals[:, 0], normals[:, 1]
        return np.sqrt(np.maximum(nx * nx * t[:, 0] + 2 * nx * ny * t[:, 1] + ny * ny * t[:, 2], 0))


class Camera:
    """K [R | t] with one focal length and radial distortion.

    params: f, cx, cy, rx, ry, rz, tx, ty, tz, k1 -- k1 is OpenCV's first radial term.
    """

    def __init__(self, params):
        params = np.asarray(params, float)
        self.params = params if len(params) == 10 else np.append(params, 0.0)

    def intrinsics(self):
        f, cx, cy = self.params[:3]
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]), np.array([self.params[9], 0, 0, 0, 0])

    def matrix(self):
        """The pinhole part, for plane homographies and the camera centre."""
        rotation, _ = cv2.Rodrigues(self.params[3:6])
        return self.intrinsics()[0] @ np.column_stack((rotation, self.params[6:9]))

    def project(self, world):
        k, distortion = self.intrinsics()
        pixels, _ = cv2.projectPoints(np.asarray(world, np.float64).reshape(-1, 1, 3),
                                      self.params[3:6], self.params[6:9], k, distortion)
        return pixels.reshape(-1, 2)

    def to_plane(self, pixels, height):
        """Where each pixel's ray meets the plane `height` up."""
        k, distortion = self.intrinsics()
        normalised = cv2.undistortPoints(np.asarray(pixels, np.float64).reshape(-1, 1, 2), k, distortion)
        rays = np.column_stack((normalised.reshape(-1, 2), np.ones(len(pixels))))
        rotation, _ = cv2.Rodrigues(self.params[3:6])
        directions = rays @ rotation          # R^T applied to each row
        centre = -rotation.T @ self.params[6:9]
        reach = (height - centre[2]) / directions[:, 2]
        return centre[:2] + directions[:, :2] * reach[:, None]

    def centre(self):
        _, _, vectors = np.linalg.svd(self.matrix())
        return vectors[-1][:3] / vectors[-1][3]

    @classmethod
    def from_matrix(cls, matrix, frame_size):
        """The nearest square-pixel camera to a 3x4, as a starting point."""
        k, rotation, centre = cv2.decomposeProjectionMatrix(matrix / np.linalg.norm(matrix[2, :3]))[:3]
        k = k / k[2, 2]
        centre = (centre[:3] / centre[3]).ravel()
        rvec, _ = cv2.Rodrigues(rotation)
        tvec = -rotation @ centre
        return cls([(k[0, 0] + k[1, 1]) / 2, k[0, 2], k[1, 2], *rvec.ravel(), *tvec])


def tile_points(camera_xyz, tile, samples=16):
    """Trusted-edge sample points of one tile, in world mm, with their outward normals."""
    x, y, angle = tile
    corners = pose.tile_corners(np.array((x, y)), angle)
    points, normals, weights = [], [], []
    along = np.linspace(0.1, 0.9, samples)
    for index in range(4):
        start, end = corners[index], corners[(index + 1) % 4]
        direction = (end - start) / np.linalg.norm(end - start)
        normal = np.array([direction[1], -direction[0]])
        if np.dot(normal, np.array((x, y)) - start) > 0:
            normal = -normal
        is_far = np.dot(normal, camera_xyz[:2] - (start + end) / 2) <= 0
        height = pose.TILE_THICKNESS_MM if is_far else 0.0
        edge = start + np.outer(along, end - start)
        points.append(np.column_stack((edge, np.full(samples, height))))
        normals.append(np.repeat(np.append(normal, 0.0)[None], samples, axis=0))
        weights.append(np.full(samples, 1.0 if is_far else pose.NEAR_EDGE_WEIGHT))
    return np.vstack(points), np.vstack(normals), np.concatenate(weights)


def score(edges, camera, tile):
    points, normals, weights = tile_points(camera.centre(), tile)
    pixels = camera.project(points)
    ahead = camera.project(points + 0.2 * normals)
    direction = ahead - pixels
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return float((edges.across(pixels, direction) * weights).sum() / weights.sum())


def descend(value_of, start, steps, rounds=30):
    best, value, steps = np.array(start, float), value_of(start), np.array(steps, float)
    for _ in range(rounds):
        improved = False
        for axis in range(len(best)):
            if not steps[axis]:
                continue
            for sign in (1, -1):
                trial = best.copy()
                trial[axis] += sign * steps[axis]
                trial_value = value_of(trial)
                if trial_value > value:
                    best, value, improved = trial, trial_value, True
        if not improved:
            steps /= 2
    return best, value


def carried(old, new, tile):
    """The same tile under another camera: its centre and baseline kept where they are in the image.

    Without this every trial camera moves all the tiles a millimetre and scores
    worse for it, so no change to the camera is ever accepted. Carried, a trial
    camera changes only the tiles' shapes -- which is what they can judge.
    """
    x, y, angle = tile
    ahead = np.array((x, y)) + 5.0 * np.array((np.cos(angle), np.sin(angle)))
    pixels = old.project(np.array([[x, y, pose.TILE_THICKNESS_MM], [*ahead, pose.TILE_THICKNESS_MM]]))
    ground = new.to_plane(pixels, pose.TILE_THICKNESS_MM)
    return np.array((*ground[0], np.arctan2(*(ground[1] - ground[0])[::-1])))


def fit(edges, camera, tiles, fixed_centre, is_distorted=False, rounds=8):
    tiles = [np.array(t, float) for t in tiles]
    camera_steps = np.array([8.0, 0 if fixed_centre else 8.0, 0 if fixed_centre else 8.0,
                             0.002, 0.002, 0.002, 1.0, 1.0, 1.0, 0.01 if is_distorted else 0.0])
    for _ in range(rounds):
        tiles = [descend(lambda p: score(edges, camera, p), t, (0.2, 0.2, np.radians(0.3)), 10)[0]
                 for t in tiles]

        def value(params):
            trial = Camera(params)
            return sum(score(edges, trial, carried(camera, trial, t)) for t in tiles)

        params, _ = descend(value, camera.params, camera_steps, 10)
        tiles = [carried(camera, Camera(params), t) for t in tiles]
        camera = Camera(params)
    return camera, tiles


def register(camera, mapping):
    """Turn and shift in the desk plane taking the camera's world to the arm's, through the cubes."""
    seen, arm = [], []
    for obs in mapping.observations:
        # Where the camera's ray through this pixel meets the plane the feature is on.
        seen.append(camera.to_plane(np.array([obs.pixel]), obs.height_mm)[0])
        arm.append(obs.ground)
    seen, arm = np.array(seen), np.array(arm)
    transform, _ = cv2.estimateAffinePartial2D(seen, arm, method=cv2.LMEDS)
    scale = np.hypot(transform[0, 0], transform[1, 0])
    rigid = transform.copy()
    rigid[:, :2] /= scale                      # shape comes from the tiles; keep their millimetres
    centre_seen, centre_arm = seen.mean(axis=0), arm.mean(axis=0)
    rigid[:, 2] = centre_arm - rigid[:, :2] @ centre_seen
    residuals = np.linalg.norm(seen @ rigid[:, :2].T + rigid[:, 2] - arm, axis=1)
    return rigid, scale, residuals


def main() -> int:
    frame = cv2.imread(sys.argv[1])
    mapping = DeskMapping.load(ROOT / "config" / "calibration.json")
    found = scene.find_tiles(frame, mapping, read.Reader())
    tiles = [(*t.centre_mm, np.radians(t.baseline_deg)) for t in found]
    edges = ImageEdges(frame)
    start = Camera.from_matrix(mapping.camera(), frame.shape[1::-1])
    height, width = frame.shape[:2]
    k = cv2.decomposeProjectionMatrix(mapping.camera())[0]
    print(f"condor's DLT: fx {k[0, 0] / k[2, 2]:.0f}, fy {k[1, 1] / k[2, 2]:.0f}; "
          f"square-pixel start f {start.params[0]:.0f}")

    results = {}
    runs = (("pinhole, centred principal point", True, False),
            ("pinhole", False, False),
            ("pinhole + radial k1", False, True))
    for label, fixed, is_distorted in runs:
        params = start.params.copy()
        if fixed:
            params[1:3] = (width / 2, height / 2)
        camera = Camera(params)
        # Tiles stay where they are in the image when the start camera changes.
        starts = [carried(start, camera, t) for t in tiles]
        camera, fitted = fit(edges, camera, starts, fixed, is_distorted)
        rigid, scale, residuals = register(camera, mapping)
        f, cx, cy = camera.params[:3]
        fov = np.degrees(2 * np.arctan(np.array((width / 2, height / 2, np.hypot(width, height) / 2)) / f))
        total = sum(score(edges, camera, t) for t in fitted) / len(fitted)
        print(f"{label}: f {f:.0f} px, centre ({cx:.0f}, {cy:.0f}), k1 {camera.params[9]:+.3f}, "
              f"field of view {fov[0]:.1f} x {fov[1]:.1f} deg, {fov[2]:.1f} diagonal; mean edge {total:.2f}")
        print(f"   camera at {np.round(camera.centre(), 1)} (tile frame); cubes need scale {scale:.3f} "
              f"to meet the arm; after turn+shift they miss by {np.round(residuals, 1)} mm")
        results[label] = {"params": camera.params.tolist(), "rigid": rigid.tolist(),
                          "tiles": [list(t) for t in fitted]}
    (ROOT / "config" / "camera-fit.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
