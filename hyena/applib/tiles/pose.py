"""Where a tile's top face is and how it is turned, fitted to the edges round it, and whether it is a tile at all.

The top face is a known square, so only its centre and turn are unknown. In the
view at the top faces' height it is exact, and each of its four sides is an edge:
against the desk on the sides that face away from the camera, against the tile's
own darker side face on the sides that face it. The camera sees those side faces
as a band beyond the top face, and where the band meets the desk is a second
edge, as far out as the camera's height and distance put it: a pose is scored on
those too, so that a square slid onto the band's outer edge -- on a pale desk
the strongest edge a tile has -- explains half of what it should.

A side counts by how strong its edge is along four fifths of its length, at the
least: a tile's edge runs the whole side, while a bar of its letter, a crack in
the desk or a neighbour's edge runs part of it. A pose counts by its sides'
geometric mean, so that a pose with a side missing loses to one with all four.
Both matter because a letter's bars are straight and strong: three of a tile's
sides and a bar make a square too, half a tile off, and so do two tiles in line
with a bar on each.

A fit is a tile when each of its sides stands out from what lies just beyond it,
measured the same way: the gradient across lines parallel to the side, a few
millimetres out. A side of a tile is where one material meets another; the desk's
grain has as much beside it as on it, and a side a fit made up has nothing there
at all. Weak sides are fine as long as they stand out: a pale face on white paper
shows its far sides faintly, but on smooth paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .edges import EdgeField

# Corners are rounded: sides are sampled on their straight part only, this far in from each corner, mm.
CORNER_MARGIN_MM = 2.5
# A side's strength: what its edge reaches along this share of its length, at the least.
SIDE_PERCENTILE = 20.0
SAMPLE_MM = 0.5
COARSE_SAMPLE_MM = 1.5
# The fit searches this far round where the votes put the centre, mm, and every turn: a letter's strokes
# can pull a tile's votes most of half a tile off its centre.
SEARCH_MM = 9.0
COARSE_STEP_MM = 1.5
COARSE_TURN_DEG = 5.0
# The close fit searches round the rough one by what that can be off: half its step, and half its turn's.
FINE_REACH_MM = 1.0
FINE_STEP_MM = 0.25
FINE_TURN_REACH_DEG = 3.0
FINE_TURN_DEG = 1.0
# The side faces' bottom edges count for this much of a top face's side: next to another tile there is no band.
BAND_WEIGHT = 0.5
# What lies beyond a side is measured on lines this far outside it, mm: past the tile's side face, which the
# camera sees beyond the sides facing it.
BEYOND_MM = (6.0, 7.5, 9.0)


@dataclass(frozen=True)
class Fit:
    centre: Tuple[float, float]     # desk mm
    turn_deg: float                 # -45 to 45: the sides' direction from desk +x
    sides: Tuple[float, ...]        # each side's median gradient across it, colour units a mm
    beyond: Tuple[float, ...]       # the same a few millimetres outside each side
    is_seen: bool                   # every side is in the frame

    @property
    def contrast(self) -> float:
        """How far the weakest side stands out from what lies beyond it."""
        return min(side / max(beyond, 1e-3) for side, beyond in zip(self.sides, self.beyond))


def square_corners(centre, turn_rad: float, size_mm: float) -> np.ndarray:
    """A square's corners, desk mm, counter-clockwise on the desk as the view shows it."""
    along = np.array([np.cos(turn_rad), np.sin(turn_rad)])
    across = np.array([-along[1], along[0]])
    half = size_mm / 2
    return np.array([np.asarray(centre) + half * (sx * along + sy * across)
                     for sx, sy in ((1, -1), (1, 1), (-1, 1), (-1, -1))])


def outline(centre, turn_rad: float, size_mm: float, step_mm: float):
    """Points on the straight part of each side, desk mm, and their outward normals: (4, n, 2) each."""
    along = np.arange(-size_mm / 2 + CORNER_MARGIN_MM, size_mm / 2 - CORNER_MARGIN_MM + 1e-9, step_mm)
    points, normals = [], []
    for quarter in range(4):
        angle = turn_rad + quarter * np.pi / 2
        out = np.array([np.cos(angle), np.sin(angle)])
        side = np.array([-out[1], out[0]])
        points.append(np.asarray(centre) + out * size_mm / 2 + along[:, None] * side)
        normals.append(np.repeat(out[None], len(along), axis=0))
    return np.array(points), np.array(normals)


def _edges(field: EdgeField, centre, turn_rad: float, size_mm: float, step_mm: float):
    """What a pose is scored on, round (0, 0): the top face's four sides, then the bottom edges of the side faces,
    seen beyond the sides that face the camera. Points and normals (8, n, 2), and each edge's weight (8,)."""
    points, normals = outline((0.0, 0.0), turn_rad, size_mm, step_mm)
    camera = field.camera_xyz
    # Where the desk under the top face's edge is seen in the view, from that edge: towards the camera's foot.
    band = (camera[:2] - np.asarray(centre)) * field.view.height_mm / camera[2]
    facing = np.clip(normals[:, 0] @ band / max(float(np.linalg.norm(band)), 1e-9), 0.0, 1.0)
    return (np.concatenate((points, points + band)), np.concatenate((normals, normals)),
            np.concatenate((np.ones(4), BAND_WEIGHT * facing)))


def rough_fit(field: EdgeField, near: Tuple[float, float], size_mm: float) -> Fit:
    """The square `size_mm` across whose sides lie best on the view's edges within SEARCH_MM of `near`, at any turn:
    to a millimetre and a few degrees, where `close_fit()` starts. Not a measure of whether it is a tile: a side a
    millimetre off its edge stands out little."""
    turns = np.radians(np.arange(0.0, 90.0, COARSE_TURN_DEG))
    centre, turn = _search(field, np.asarray(near, float), turns, SEARCH_MM, COARSE_STEP_MM, COARSE_SAMPLE_MM,
                           size_mm, is_exact=False)
    return measure(field, centre, turn, size_mm)


def close_fit(field: EdgeField, rough: Fit, size_mm: float) -> Fit:
    """`rough` fitted closely: to a few hundredths of a millimetre and a tenth of a degree."""
    centre, turn = np.array(rough.centre), np.radians(rough.turn_deg)
    turns = turn + np.radians(np.arange(-FINE_TURN_REACH_DEG, FINE_TURN_REACH_DEG + 1e-9, FINE_TURN_DEG))
    centre, turn = _search(field, centre, turns, FINE_REACH_MM, FINE_STEP_MM, SAMPLE_MM, size_mm, is_exact=True)
    centre, turn = _polish(field, centre, turn, size_mm)
    return measure(field, centre, turn, size_mm)


def measure(field: EdgeField, centre, turn_rad: float, size_mm: float) -> Fit:
    """How a square there lies on the view: each side, and the same just beyond it."""
    points, normals = outline(centre, turn_rad, size_mm, SAMPLE_MM)
    sides = _side(field.across(points, normals))
    out = np.concatenate([points + normals * distance for distance in BEYOND_MM], axis=1)
    beyond = np.median(field.across(out, np.concatenate([normals] * len(BEYOND_MM), axis=1)), axis=1)
    seen = field.is_seen(points.reshape(-1, 2))
    return Fit((float(centre[0]), float(centre[1])), _turn_deg(turn_rad), tuple(float(side) for side in sides),
               tuple(float(value) for value in beyond), bool(seen.all()))


def _search(field: EdgeField, centre: np.ndarray, turns: np.ndarray, reach_mm: float, step_mm: float,
            sample_mm: float, size_mm: float, is_exact: bool):
    """The best centre on a grid round `centre`, and the best of `turns`, all tried at once."""
    offsets = np.arange(-reach_mm, reach_mm + 1e-9, step_mm)
    grid = centre + np.array([(dx, dy) for dx in offsets for dy in offsets])
    shapes = [_edges(field, centre, float(turn), size_mm, sample_mm) for turn in turns]
    points = np.array([shape[0] for shape in shapes])            # turns, 8, n, 2
    normals = np.array([shape[1] for shape in shapes])
    weights = np.array([shape[2] for shape in shapes])           # turns, 8
    shifted = points[:, None] + grid[None, :, None, None, :]     # turns, grid, 8, n, 2
    strength = field.across(shifted, np.broadcast_to(normals[:, None], shifted.shape), is_exact)
    scores = _score(_side(strength), weights[:, None])           # turns, grid
    turn_index, grid_index = np.unravel_index(int(np.argmax(scores)), scores.shape)
    return grid[grid_index], float(turns[turn_index])


def _polish(field: EdgeField, centre: np.ndarray, turn: float, size_mm: float):
    """Coordinate steps, halved each time none helps, down to a fiftieth of a millimetre."""
    params = np.array([centre[0], centre[1], turn])
    steps = np.array([FINE_STEP_MM / 2, FINE_STEP_MM / 2, np.radians(FINE_TURN_DEG / 2)])

    def score(trial: np.ndarray) -> float:
        points, normals, weights = _edges(field, trial[:2], trial[2], size_mm, SAMPLE_MM)
        return float(_score(_side(field.across(points + trial[:2], normals)), weights))

    value = score(params)
    for _ in range(60):
        is_better = False
        for axis in range(3):
            for sign in (1.0, -1.0):
                trial = params.copy()
                trial[axis] += sign * steps[axis]
                trial_value = score(trial)
                if trial_value > value:
                    params, value, is_better = trial, trial_value, True
        if not is_better:
            steps /= 2
            if steps[0] < 0.02:
                break
    return params[:2], float(params[2])


def _side(strength: np.ndarray) -> np.ndarray:
    """Each side's strength from its samples, the last axis."""
    return np.percentile(strength, SIDE_PERCENTILE, axis=-1)


def _score(edges: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The weighted geometric mean of the edges' strengths, the last axis."""
    return np.exp((np.log(np.maximum(edges, 1e-3)) * weights).sum(axis=-1) / weights.sum(axis=-1))


def _turn_deg(turn_rad: float) -> float:
    return float((np.degrees(turn_rad) + 45.0) % 90.0 - 45.0)
