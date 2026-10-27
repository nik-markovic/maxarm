#!/usr/bin/env python3
"""25 mm tiles on a desk, drawn through a camera that is known exactly: the tile finder's ground truth.

    scene = render([(-150.0, -150.0, 10.0), (-100.0, -200.0, -30.0)])   # desk x, y, turn of each tile
    scene.frame                     # BGR, 2592 x 1944, as the snapshot is
    scene.desk_to_pixel             # 3x4: what camera.json would hold
    scene.fit                       # the same as the camera page's CameraFit, to write as camera.json
    scene.tiles                     # desk x, y, turn, as asked

Each tile is a 25 x 25 x 2 mm box with rounded corners: a cream top face with a letter's strokes in dark ink, its
rounded rim a millimetre wide and a little darker, brown sides, a faint shadow. The desk is wood, white paper or
plain grey. The camera stands where the owner's does, in front of the arm and to its right, and the arm's x and y
can be given a stretch from true millimetres, as a real arm's are.
Shapes are drawn at eight times the size and averaged down. It is a drawing: it says whether the finder finds tiles
that are there to find, nothing about a real camera.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "calibration"), str(HERE)]

from camera.fit import HANDEDNESS, CameraFit                # noqa: E402
from cube_scene import _fill as _fill_in_frame, _project         # noqa: E402

SIZE = (2592, 1944)
TILE_MM = 25.0
HEIGHT_MM = 2.0
CORNER_MM = 2.0
RIM_MM = 1.0
TOP_BGR = (214, 232, 240)
RIM_BGR = (170, 190, 205)
SIDE_BGR = (70, 105, 150)
INK_BGR = (40, 38, 36)
# About where camera.json has the owner's camera: in front of the arm and to its right, 210 mm up.
CAMERA_AT = (-266.0, -287.0, 206.0)            # true mm: the board's mirror makes desk x = -true x
LOOKING_AT = (110.0, -150.0, 0.0)
FOCAL_SHARE = 0.78
# Strokes of a few capitals, in a tile's own mm: x along its bottom edge, y up it, centre (0, 0).
LETTERS = {
    "E": [[(5, 7), (-5, 7), (-5, -7), (5, -7)], [(-5, 0), (3, 0)]],
    "T": [[(-6, 7), (6, 7)], [(0, 7), (0, -7)]],
    "N": [[(-5, -7), (-5, 7), (5, -7), (5, 7)]],
    "L": [[(-5, 7), (-5, -7), (5, -7)]],
    "V": [[(-6, 7), (0, -7), (6, 7)]],
    "Z": [[(-5, 7), (5, 7), (-5, -7), (5, -7)]],
}


@dataclass
class Scene:
    frame: np.ndarray
    desk_to_pixel: np.ndarray
    fit: CameraFit
    tiles: List[Tuple[float, float, float]]


def render(tiles: Sequence[Tuple[float, float, float]], desk: str = "wood", stretch=((1.0, 0.0), (0.0, 1.0)),
           camera_at=CAMERA_AT, looking_at=LOOKING_AT, letters: Optional[str] = "ETNLVZ", seed: int = 3) -> Scene:
    """`tiles` as (desk x, y, turn degrees), each drawn with the next of `letters` on it, or none."""
    width, height = SIZE
    desk_to_true = np.array(stretch, float) @ HANDEDNESS
    true_to_pixel = _camera(np.array(camera_at, float), np.array(looking_at, float), FOCAL_SHARE * width)
    rng = np.random.default_rng(seed)
    frame = _desk(desk, rng)
    camera = np.array(camera_at, float)
    order = sorted(range(len(tiles)), key=lambda index: -np.linalg.norm(
        desk_to_true @ tiles[index][:2] - camera[:2]))
    for index in order:
        x, y, turn = tiles[index]
        letter = letters[index % len(letters)] if letters else None
        _tile(frame, true_to_pixel, desk_to_true, (x, y), math.radians(turn), letter)
    frame = cv2.GaussianBlur(frame, (0, 0), 1.2)
    frame = np.clip(frame + rng.normal(0, 2.5, frame.shape), 0, 255).astype(np.uint8)
    lift = np.eye(4)
    lift[:2, :2] = desk_to_true
    fit = CameraFit(true_to_pixel, desk_to_true, {}, SIZE, (0.0, 0.0), 0)
    return Scene(frame, true_to_pixel @ lift, fit, [tuple(map(float, tile)) for tile in tiles])


def _fill(frame: np.ndarray, polygon: np.ndarray, colour) -> None:
    """cube_scene's fill, for a polygon that may lie partly or wholly outside the frame."""
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    if high[0] < 0 or high[1] < 0 or low[0] >= frame.shape[1] or low[1] >= frame.shape[0]:
        return
    _fill_in_frame(frame, polygon, colour)


def _camera(position: np.ndarray, target: np.ndarray, focal: float) -> np.ndarray:
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.array([right, down, forward])
    intrinsic = np.array([[focal, 0, SIZE[0] / 2 + 21], [0, focal, SIZE[1] / 2 + 17], [0, 0, 1.0]])
    return intrinsic @ np.column_stack((rotation, -rotation @ position))


def rounded_square(centre, turn: float, size: float = TILE_MM, radius: float = CORNER_MM, per_corner: int = 6):
    """The outline of a square with rounded corners, desk mm, counter-clockwise on the desk."""
    half = size / 2 - radius
    points = []
    for quarter, (sx, sy) in enumerate(((1, -1), (1, 1), (-1, 1), (-1, -1))):
        start = -math.pi / 2 + quarter * math.pi / 2
        for step in range(per_corner + 1):
            angle = start + (math.pi / 2) * step / per_corner
            points.append((sx * half + radius * math.cos(angle), sy * half + radius * math.sin(angle)))
    local = np.array(points)
    c, s = math.cos(turn), math.sin(turn)
    return np.asarray(centre) + local @ np.array([[c, s], [-s, c]])


def _tile(frame, true_to_pixel, desk_to_true, centre, turn: float, letter: Optional[str]) -> None:
    outline = rounded_square(centre, turn) @ desk_to_true.T
    bottom = _project(true_to_pixel, np.column_stack((outline, np.zeros(len(outline)))))
    top = _project(true_to_pixel, np.column_stack((outline, np.full(len(outline), HEIGHT_MM))))
    _shadow(frame, true_to_pixel, outline)
    _fill(frame, cv2.convexHull(np.vstack((bottom, top)).astype(np.float32)).reshape(-1, 2), SIDE_BGR)
    _fill(frame, top, RIM_BGR)
    flat = rounded_square(centre, turn, TILE_MM - 2 * RIM_MM, CORNER_MM - RIM_MM) @ desk_to_true.T
    _fill(frame, _project(true_to_pixel, np.column_stack((flat, np.full(len(flat), HEIGHT_MM)))), TOP_BGR)
    if letter:
        c, s = math.cos(turn), math.sin(turn)
        for stroke in LETTERS[letter]:
            for start, end in zip(stroke[:-1], stroke[1:]):
                _stroke(frame, true_to_pixel, desk_to_true, centre, np.array([[c, s], [-s, c]]), start, end)


def _shadow(frame, true_to_pixel, outline: np.ndarray) -> None:
    """A soft shadow on the desk, a little away from the light."""
    pixels = _project(true_to_pixel, np.column_stack((outline + (1.5, 1.0), np.zeros(len(outline)))))
    x0, y0 = np.maximum(np.floor(pixels.min(axis=0)).astype(int) - 20, 0)
    x1, y1 = np.minimum(np.ceil(pixels.max(axis=0)).astype(int) + 20, SIZE)
    mask = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.fillPoly(mask, [np.round(pixels - (x0, y0)).astype(np.int32)], 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), 3.0)
    frame[y0:y1, x0:x1] *= (1.0 - 0.35 * mask)[..., None]


def _stroke(frame, true_to_pixel, desk_to_true, centre, turn, start, end, width_mm: float = 1.0) -> None:
    """One straight stroke of a letter, `width_mm` wide, on the top face."""
    start, end = np.array(start, float), np.array(end, float)
    along = (end - start) / np.linalg.norm(end - start)
    side = np.array([-along[1], along[0]]) * width_mm / 2
    quad = np.array([start - side - along * 0.4, end - side + along * 0.4, end + side + along * 0.4,
                     start + side - along * 0.4])
    desk = np.asarray(centre) + quad @ turn
    pixels = _project(true_to_pixel, np.column_stack((desk @ desk_to_true.T, np.full(4, HEIGHT_MM))))
    _fill(frame, pixels, INK_BGR)


def _desk(kind: str, rng: np.random.Generator) -> np.ndarray:
    width, height = SIZE
    if kind == "grey":
        return np.clip(np.full((height, width, 3), 128.0) + rng.normal(0, 3, (height, width, 3)), 0, 255)
    if kind == "paper":
        shade = cv2.GaussianBlur(rng.normal(0, 6, (height, width)).astype(np.float32), (0, 0), 30)[..., None]
        return np.clip(np.array([236, 240, 242], np.float32) + shade + rng.normal(0, 2, (height, width, 3)), 0, 255)
    rows, columns = np.mgrid[0:height, 0:width].astype(np.float32)
    grain = np.sin((rows * 0.9 + columns * 0.25) / 9.0 + 3 * np.sin(columns / 120.0)) * 10.0
    streaks = cv2.GaussianBlur(rng.normal(0, 22, (height, width)).astype(np.float32), (0, 0), sigmaX=40, sigmaY=3)
    shade = (grain + streaks)[..., None]
    return np.clip(np.array([88, 104, 120], np.float32) + shade + rng.normal(0, 3, (height, width, 3)), 0, 255)


if __name__ == "__main__":
    scene = render([(-150.0, -150.0, 10.0), (-100.0, -200.0, -30.0), (-120.0, -120.0, 40.0)])
    cv2.imwrite("/tmp/tile-scene.jpg", scene.frame)
    print("wrote /tmp/tile-scene.jpg")
