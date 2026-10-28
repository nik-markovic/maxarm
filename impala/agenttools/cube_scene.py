#!/usr/bin/env python3
"""Three 40 mm cubes on a wooden desk, drawn through a camera that is known exactly: the fit's ground truth.

    scene = render()                       # the owner's layout and a camera like the owner's
    scene.frame                            # BGR, 1296 x 972
    scene.points                           # desk x, y under each cube, as the arm would record them
    scene.desk_to_pixel                    # 3x4: what the fit should find

The arm's x and y can be given a stretch from true millimetres, as a real arm's
are, and a green, non-convex shape stands near the arm's base, as the arm's own
green body does. Faces are flat-shaded, each cube casts a soft shadow, and the
whole is blurred and noisy a little. Shapes are drawn at eight times the size
and averaged down: OpenCV fills a polygon half a pixel wider than asked, which
would make every cube look a little closer than it is. It is a drawing: it says
whether the fit finds a camera that is there to find, nothing about a real camera.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera.fit import HANDEDNESS                           # noqa: E402
from camera.model import CUBE_MM, cube_corners              # noqa: E402

SIZE = (1296, 972)
SPOTS = {"blue": (-140.0, -220.0), "green": (140.0, -220.0), "red": (140.0, 0.0)}
TURNS_DEG = {"blue": 30.0, "green": 8.0, "red": 50.0}
COLOURS_BGR = {"blue": (235, 200, 60), "green": (110, 175, 20), "red": (60, 70, 230)}
LIGHT = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
SUPERSAMPLE = 8


@dataclass
class Scene:
    frame: np.ndarray
    points: Dict[str, Tuple[float, float]]
    desk_to_pixel: np.ndarray
    true_to_pixel: np.ndarray


def render(spots: Optional[Dict[str, Tuple[float, float]]] = None, stretch=((1.0, 0.0), (0.0, 1.0)),
           camera_at=(-250.0, -390.0, 250.0), looking_at=(-53.0, -147.0, 0.0), focal=0.74,
           decoy: bool = True, seed: int = 7) -> Scene:
    """The cubes at `spots` (desk x, y), the arm's x, y mapped to true mm by `stretch` and the board's mirror.

    `camera_at` and `looking_at` are true mm; `focal` is a share of the frame's width.
    """
    spots = spots or SPOTS
    width, height = SIZE
    desk_to_true = np.array(stretch, float) @ HANDEDNESS
    true_to_pixel = _camera(np.array(camera_at, float), np.array(looking_at, float), focal * width)
    rng = np.random.default_rng(seed)
    frame = _desk(rng)
    if decoy:
        _decoy(frame, true_to_pixel)
    cubes = sorted(spots.items(), key=lambda item: -np.linalg.norm(desk_to_true @ item[1] - camera_at[:2]))
    for colour, spot in cubes:
        centre = desk_to_true @ spot
        _shadow(frame, true_to_pixel, centre, math.radians(TURNS_DEG[colour]))
        _cube(frame, true_to_pixel, centre, math.radians(TURNS_DEG[colour]), np.array(COLOURS_BGR[colour]),
              np.array(camera_at, float))
    frame = cv2.GaussianBlur(frame, (0, 0), 1.0)
    frame = np.clip(frame + rng.normal(0, 2.0, frame.shape), 0, 255).astype(np.uint8)
    lift = np.eye(4)
    lift[:2, :2] = desk_to_true
    return Scene(frame, dict(spots), true_to_pixel @ lift, true_to_pixel)


def _camera(position: np.ndarray, target: np.ndarray, focal: float) -> np.ndarray:
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.array([right, down, forward])
    intrinsic = np.array([[focal, 0, SIZE[0] / 2 + 9], [0, focal, SIZE[1] / 2 - 6], [0, 0, 1.0]])
    return intrinsic @ np.column_stack((rotation, -rotation @ position))


def _project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def _desk(rng: np.random.Generator) -> np.ndarray:
    """Brown wood: a base colour, grain running across in streaks, and noise."""
    width, height = SIZE
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float32)
    grain = np.sin((rows * 0.9 + columns * 0.25) / 7.0 + 3 * np.sin(columns / 90.0)) * 9.0
    streaks = cv2.GaussianBlur(rng.normal(0, 18, (height, width)).astype(np.float32), (0, 0), sigmaX=25, sigmaY=2)
    shade = (grain + streaks)[..., None]
    return np.clip(np.array([92, 108, 122], np.float32) + shade + rng.normal(0, 3, (height, width, 3)), 0, 255)


def _decoy(frame: np.ndarray, true_to_pixel: np.ndarray) -> None:
    """A green L on the desk round the arm's base: as green as a cube, but no cube's shape."""
    shape = np.array([(-60, -60), (60, -60), (60, -20), (-20, -20), (-20, 60), (-60, 60)], float)
    pixels = _project(true_to_pixel, np.column_stack((shape, np.zeros(len(shape)))))
    _fill(frame, pixels, (60, 140, 30))


def _shadow(frame: np.ndarray, true_to_pixel: np.ndarray, centre: np.ndarray, turn: float) -> None:
    corners = cube_corners(centre, turn)
    away = -LIGHT[:2] / LIGHT[2] * CUBE_MM
    footprint = np.vstack((corners[:4, :2], corners[:4, :2] + away))
    hull = cv2.convexHull(_project(true_to_pixel, np.column_stack((footprint, np.zeros(8)))).astype(np.float32))
    mask = np.zeros(frame.shape[:2], np.float32)
    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 6.0)
    frame *= (1.0 - 0.45 * mask)[..., None]


def _cube(frame: np.ndarray, true_to_pixel: np.ndarray, centre: np.ndarray, turn: float, colour: np.ndarray,
          camera: np.ndarray) -> None:
    corners = cube_corners(centre, turn)
    pixels = _project(true_to_pixel, corners)
    c, s = math.cos(turn), math.sin(turn)
    faces = [((4, 5, 6, 7), (0, 0, 1)), ((0, 1, 5, 4), (s, -c, 0)), ((1, 2, 6, 5), (c, s, 0)),
             ((2, 3, 7, 6), (-s, c, 0)), ((3, 0, 4, 7), (-c, -s, 0))]
    for indices, normal in faces:
        normal = np.array(normal, float)
        if (camera - corners[list(indices)].mean(axis=0)) @ normal <= 0:
            continue
        light = 0.35 + 0.65 * max(0.0, float(normal @ LIGHT))
        _fill(frame, pixels[list(indices)], np.clip(colour * light, 0, 255))


def _fill(frame: np.ndarray, polygon: np.ndarray, colour) -> None:
    """Paint `polygon` (pixel corners, pixel centres at whole numbers) by how much of each pixel it covers."""
    x0, y0 = np.floor(polygon.min(axis=0)).astype(int) - 2
    x1, y1 = np.ceil(polygon.max(axis=0)).astype(int) + 3
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, frame.shape[1]), min(y1, frame.shape[0])
    fine = np.zeros(((y1 - y0) * SUPERSAMPLE, (x1 - x0) * SUPERSAMPLE), np.float32)
    corners = ((polygon - (x0, y0) + 0.5) * SUPERSAMPLE - 0.5) * 16
    cv2.fillPoly(fine, [np.round(corners).astype(np.int32)], 1.0, cv2.LINE_8, shift=4)
    cover = cv2.resize(fine, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)[..., None]
    frame[y0:y1, x0:x1] = frame[y0:y1, x0:x1] * (1 - cover) + np.array(colour, np.float32) * cover
