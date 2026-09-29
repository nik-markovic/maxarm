"""Where a tile is and how it is turned, fitted to the edges that can be trusted.

The tile is a known box -- 17.5 mm along the letter's baseline, 20 mm up it,
4 mm thick -- so only its position and turn are unknown. The long side is up
the letter on all 13 single tiles of the first scene, measured.

In the view at tile height its top face is exact, but the camera also
sees the side faces that point its way, as a band beside the top face. The
boundary between top face and side face is pale against pale and is not used.
What is used:

- **Top-face edges facing away from the camera.** Nothing of the tile lies
  beyond them, so they are the top face's own outline against the desk.
- **The bottom edges on the camera's side**, where the side face meets the
  desk. The calibrated camera says exactly where they fall for a 4 mm tile, so
  they are predicted, never searched for, and they count for half: the band's
  width is as good as the thickness is.

An edge is scored as the colour gradient across it, in either direction, so
the desk may be lighter or darker than the tile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from topdown import TopDown

TILE_MM = (17.5, 20.0)          # along the letter's baseline, then up the letter
TILE_THICKNESS_MM = 4.0
NEAR_EDGE_WEIGHT = 0.5
# Corners are rounded and chipped; sample the straight part of each edge.
CORNER_MARGIN_MM = 1.5
SAMPLES_PER_EDGE = 24
COARSE_SAMPLES_PER_EDGE = 10
# The glyph is within a few millimetres of the tile's centre on every letter.
SEARCH_MM = 3.0
SEARCH_STEP_MM = 0.5
ANGLE_STEP_DEG = 2.0
COARSE_STEP_MM = 1.0
COARSE_ANGLE_STEP_DEG = 6.0
COARSE_KEEP = 3
# How far the baseline may be from where the letter says it is.
LETTER_ANGLE_SPREAD_DEG = 10.0


@dataclass(frozen=True)
class TilePose:
    centre_mm: Tuple[float, float]      # AACS x, y of the top face's centre
    angle_deg: float                    # the baseline's direction, in AACS, 0-180
    edge_score: float                   # mean gradient across the outline
    contrast: float                     # that, over the patch's median gradient

    def corners_mm(self) -> np.ndarray:
        return tile_corners(np.array(self.centre_mm), np.radians(self.angle_deg))


class EdgeField:
    """Where the view's colour changes, and how sharply in each direction.

    Stored as the structure tensor summed over L, a and b: for any direction n,
    the colour gradient across an edge with that normal is sqrt(n' J n), so one
    lookup of three numbers scores a sample point whichever way its edge runs.
    """

    def __init__(self, view: TopDown, camera_xyz: np.ndarray):
        self.view = view
        self.camera_xyz = camera_xyz
        lab = cv2.cvtColor(cv2.GaussianBlur(view.image, (0, 0), view.px_per_mm * 0.25),
                           cv2.COLOR_BGR2LAB).astype(np.float32)
        tensor = np.zeros(view.image.shape[:2] + (3,), np.float32)
        for channel in range(3):
            gx = cv2.Scharr(lab[..., channel], cv2.CV_32F, 1, 0) / 32
            gy = cv2.Scharr(lab[..., channel], cv2.CV_32F, 0, 1) / 32
            tensor[..., 0] += gx * gx
            tensor[..., 1] += gx * gy
            tensor[..., 2] += gy * gy
        self.tensor = tensor
        # view pixel = ground_to_view @ (x, y, 1); the view is ground turned and scaled
        self.ground_to_view = np.linalg.inv(view.view_to_ground)

    def outline(self, centre: np.ndarray, angle: float, samples: int = SAMPLES_PER_EDGE
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample points in ground mm, their outward normals, and their weights."""
        top = tile_corners(centre, angle)
        to_camera = self.camera_xyz[:2] - centre
        shift = (TILE_THICKNESS_MM / self.camera_xyz[2]) * (self.camera_xyz[:2] - top)
        points, normals, weights = [], [], []
        along = np.linspace(0, 1, samples)
        for index in range(4):
            start, end = top[index], top[(index + 1) % 4]
            length = np.linalg.norm(end - start)
            direction = (end - start) / length
            normal = np.array([direction[1], -direction[0]])
            if np.dot(normal, centre - start) > 0:
                normal = -normal
            margin = CORNER_MARGIN_MM / length
            t = margin + along * (1 - 2 * margin)
            if np.dot(normal, to_camera) <= 0:
                edge = start + np.outer(t, end - start)
                weight = 1.0
            else:
                bottom_start, bottom_end = start + shift[index], end + shift[(index + 1) % 4]
                edge = bottom_start + np.outer(t, bottom_end - bottom_start)
                weight = NEAR_EDGE_WEIGHT
            points.append(edge)
            normals.append(np.repeat(normal[None], len(edge), axis=0))
            weights.append(np.full(len(edge), weight))
        return np.vstack(points), np.vstack(normals), np.concatenate(weights)

    def strength(self, points_mm: np.ndarray, normals_mm: np.ndarray, is_exact: bool = True
                 ) -> np.ndarray:
        """Colour gradient across the edge at each point, in either direction."""
        homogeneous = np.column_stack((points_mm, np.ones(len(points_mm))))
        view = homogeneous @ self.ground_to_view.T
        u, v = view[:, 0] / view[:, 2], view[:, 1] / view[:, 2]
        tensor = _bilinear(self.tensor, u, v) if is_exact else _nearest(self.tensor, u, v)
        # ground normal (nx, ny) -> view (-ny, nx): the view is ground turned a quarter
        nu, nv = -normals_mm[:, 1], normals_mm[:, 0]
        across = nu * nu * tensor[:, 0] + 2 * nu * nv * tensor[:, 1] + nv * nv * tensor[:, 2]
        return np.sqrt(np.maximum(across, 0.0))

    def score(self, centre: np.ndarray, angle: float) -> float:
        points, normals, weights = self.outline(centre, angle)
        return float((self.strength(points, normals) * weights).sum() / weights.sum())

    def background(self, centre: np.ndarray) -> float:
        """The median gradient around a tile: the desk's own texture there."""
        u, v = (int(round(c)) for c in self.view.to_view(tuple(centre)))
        half = round(25 * self.view.px_per_mm)
        patch = self.tensor[max(0, v - half):v + half, max(0, u - half):u + half]
        return float(np.median(np.sqrt(patch[..., 0] + patch[..., 2]))) + 1e-6


def fit_tile(field: EdgeField, near_mm: Tuple[float, float],
             about_deg: Optional[float] = None) -> TilePose:
    """The best tile outline within a few millimetres of `near_mm`.

    Any turn, unless `about_deg` says which way the baseline runs -- which the
    letter does once it is read, and which settles the one thing the outline
    cannot: the 17.5 mm side and the 20 mm side look alike to within the band.
    """
    near = np.array(near_mm)
    if about_deg is None:
        angles = np.arange(0, 180, COARSE_ANGLE_STEP_DEG)
    else:
        angles = about_deg + np.arange(-LETTER_ANGLE_SPREAD_DEG, LETTER_ANGLE_SPREAD_DEG + 1e-9,
                                       COARSE_ANGLE_STEP_DEG)
    # Coarse over everything, then fine around the few best: the outline score
    # is smooth over a few degrees and a millimetre, so the right pose is never
    # far from a coarse winner.
    coarse = _grid_search(field, [(near, angle) for angle in np.radians(angles)], SEARCH_MM,
                          COARSE_STEP_MM, is_coarse=True)
    fine = [(centre, angle + delta) for _, centre, angle in coarse[:COARSE_KEEP]
            for delta in np.radians(np.arange(-COARSE_ANGLE_STEP_DEG, COARSE_ANGLE_STEP_DEG + 1e-9,
                                              ANGLE_STEP_DEG))]
    best = _grid_search(field, fine, COARSE_STEP_MM, SEARCH_STEP_MM, is_coarse=False)[0]
    centre, angle = _refine(field, best[1], best[2])
    score = field.score(centre, angle)
    return TilePose((float(centre[0]), float(centre[1])), float(np.degrees(angle)) % 180.0,
                    score, score / field.background(centre))


def _grid_search(field: EdgeField, starts, reach_mm: float, step_mm: float, is_coarse: bool):
    """Each (centre, angle) start, shifted over a square grid; best first, one per start."""
    offsets = np.arange(-reach_mm, reach_mm + 1e-9, step_mm)
    grid = np.array([(dx, dy) for dx in offsets for dy in offsets])
    results = []
    for centre, angle in starts:
        points, normals, weights = field.outline(
            np.asarray(centre), angle, COARSE_SAMPLES_PER_EDGE if is_coarse else SAMPLES_PER_EDGE)
        # Every offset shares the outline's shape; shift it rather than rebuild it.
        shifted = (points[None, :, :] + grid[:, None, :]).reshape(-1, 2)
        strength = field.strength(shifted, np.tile(normals, (len(grid), 1)),
                                  is_exact=not is_coarse).reshape(len(grid), -1)
        scores = (strength * weights).sum(axis=1) / weights.sum()
        index = int(np.argmax(scores))
        results.append((float(scores[index]), np.asarray(centre) + grid[index], angle))
    return sorted(results, key=lambda result: -result[0])


def _refine(field: EdgeField, centre: np.ndarray, angle: float) -> Tuple[np.ndarray, float]:
    params = np.array([centre[0], centre[1], angle])
    steps = np.array([SEARCH_STEP_MM / 2, SEARCH_STEP_MM / 2, np.radians(ANGLE_STEP_DEG / 2)])
    value = field.score(params[:2], params[2])
    for _ in range(40):
        improved = False
        for axis in range(3):
            for sign in (1.0, -1.0):
                trial = params.copy()
                trial[axis] += sign * steps[axis]
                trial_value = field.score(trial[:2], trial[2])
                if trial_value > value:
                    params, value, improved = trial, trial_value, True
        if not improved:
            steps /= 2
            if steps[0] < 0.02:
                break
    return params[:2], float(params[2])


def tile_corners(centre: np.ndarray, angle: float) -> np.ndarray:
    """Top-face corners in ground mm, counter-clockwise from baseline-right, letter-up."""
    baseline = np.array([np.cos(angle), np.sin(angle)])
    up = np.array([-baseline[1], baseline[0]])
    half_w, half_h = TILE_MM[0] / 2, TILE_MM[1] / 2
    return np.array([centre + sx * half_w * baseline + sy * half_h * up
                     for sx, sy in ((1, -1), (1, 1), (-1, 1), (-1, -1))])


def _nearest(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    rows = np.clip(np.round(v).astype(int), 0, image.shape[0] - 1)
    cols = np.clip(np.round(u).astype(int), 0, image.shape[1] - 1)
    return image[rows, cols]


def _bilinear(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    u = np.clip(u, 0, width - 1.001)
    v = np.clip(v, 0, height - 1.001)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = (u - u0)[:, None], (v - v0)[:, None]
    return ((1 - fu) * (1 - fv) * image[v0, u0] + fu * (1 - fv) * image[v0, u0 + 1]
            + (1 - fu) * fv * image[v0 + 1, u0] + fu * fv * image[v0 + 1, u0 + 1])
