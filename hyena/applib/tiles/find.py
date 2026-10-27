"""The 25 mm tiles in a camera frame: where each one's top face is on the desk, and how it is turned.

    camera = Camera.load(Path("config/camera.json"))
    for tile in find_tiles(frame, camera):
        print(tile.x, tile.y, tile.turn)

The work is done in desk millimetres, on the frame resampled onto the plane of
the tiles' top faces, 2 mm up (`topdown`), where every tile is the same square:

1. `votes` -- every colour edge votes for the centre of a 25 mm square it could
   be a side of; the places most voted for are where tiles may be.
2. `pose`  -- a square fitted to the edges round each of those places.
3. Checks. Each side of a tile stands out from what lies just beyond it,
   measured the same way; its whole outline is in view; and tiles cannot
   overlap, so of two fits that do, the one that stands out more is kept.

Nothing assumes the tiles are lighter or darker than the desk, or any colour at
all. Tiles may touch: the seam between two counts for both. The letters on them
are not read here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from maxarm import Camera
from maxarm.geometry import BASE_FAN_DEG, bearing_degrees
from maxarm.path import BASE_LIMIT_HALF_MM, REACH_MAX_MM

from . import pose, topdown, votes
from .edges import EdgeField

TILE_MM = 25.0
TILE_HEIGHT_MM = 2.0
# Where to look: round the arm, a little beyond its reach and its turn, so that tiles just out of reach are found
# as well. Behind the arm there is its base, its cables and often the desk's edge.
SEARCH_MM = REACH_MAX_MM + 60.0
SEARCH_BEARING_DEG = BASE_FAN_DEG + 15.0
# A tile's weakest side stands out this many times from what lies just beyond it.
MIN_CONTRAST = 4.0
# A candidate this far inside a tile already found is that tile again, mm.
SAME_TILE_MM = 8.0
# Two fits overlapping by more than this share of a face are one tile found twice, or a fit gone astray.
MAX_OVERLAP = 0.1
# The most places fitted, best voted first.
MAX_CANDIDATES = 150


@dataclass(frozen=True)
class Tile:
    """A tile's top face, desk mm, 2 mm over the desk. `turn` is the sides' direction, degrees from desk +x."""

    x: float
    y: float
    turn: float                                 # -45 to 45: a square looks the same a quarter turn on
    corners: Tuple[Tuple[float, float], ...]    # the face as if its corners were sharp
    pixel: Tuple[float, float]                  # where the frame shows the face's centre
    contrast: float                             # how far its weakest side stands out from beyond it

    def to_json(self) -> dict:
        return {"x": round(self.x, 2), "y": round(self.y, 2), "turn": round(self.turn, 1),
                "corners": [[round(x, 2), round(y, 2)] for x, y in self.corners],
                "pixel": [round(self.pixel[0], 1), round(self.pixel[1], 1)], "contrast": round(self.contrast, 2)}


def find_tiles(frame: np.ndarray, camera: Camera) -> List[Tile]:
    """Every tile the frame shows round the arm, left to right as seen from behind it."""
    camera.check_frame(frame)
    view = topdown.render(frame, camera, TILE_HEIGHT_MM, (-SEARCH_MM, SEARCH_MM, -SEARCH_MM, SEARCH_MM))
    field = EdgeField(view, camera.position())
    fits: List[pose.Fit] = []
    for candidate in votes.candidates(field, TILE_MM / 2, MAX_CANDIDATES):
        place = (candidate.x, candidate.y)
        if not _is_searched(place) or _is_on(fits, place):
            continue
        rough = pose.rough_fit(field, place, TILE_MM)
        if not rough.is_seen:
            continue
        fit = pose.close_fit(field, rough, TILE_MM)
        if fit.is_seen and fit.contrast >= MIN_CONTRAST:
            fits.append(fit)
    tiles = [_tile(fit, camera) for fit in _without_overlaps(fits)]
    return sorted(tiles, key=lambda tile: math.atan2(tile.x, -tile.y))


def _is_searched(place: Tuple[float, float]) -> bool:
    """Round the arm, in front of it, and not where its base stands."""
    x, y = place
    return (math.hypot(x, y) <= SEARCH_MM and abs(bearing_degrees(x, y)) <= SEARCH_BEARING_DEG
            and max(abs(x), abs(y)) > BASE_LIMIT_HALF_MM)


def _is_on(fits: List[pose.Fit], place: Tuple[float, float]) -> bool:
    """Whether `place` is well inside the face of a tile already fitted."""
    for fit in fits:
        turn = math.radians(fit.turn_deg)
        dx, dy = place[0] - fit.centre[0], place[1] - fit.centre[1]
        along, across = dx * math.cos(turn) + dy * math.sin(turn), -dx * math.sin(turn) + dy * math.cos(turn)
        if max(abs(along), abs(across)) < TILE_MM / 2 - SAME_TILE_MM:
            return True
    return False


def _without_overlaps(fits: List[pose.Fit]) -> List[pose.Fit]:
    kept: List[pose.Fit] = []
    area = TILE_MM * TILE_MM
    for fit in sorted(fits, key=lambda fit: -fit.contrast):
        face = _face(fit)
        if all(cv2.intersectConvexConvex(face, _face(other))[0] <= MAX_OVERLAP * area for other in kept):
            kept.append(fit)
    return kept


def _face(fit: pose.Fit) -> np.ndarray:
    return pose.square_corners(fit.centre, math.radians(fit.turn_deg), TILE_MM).astype(np.float32)


def _tile(fit: pose.Fit, camera: Camera) -> Tile:
    px, py, _ = camera.from_desk((fit.centre[0], fit.centre[1], TILE_HEIGHT_MM))
    corners = pose.square_corners(fit.centre, math.radians(fit.turn_deg), TILE_MM)
    return Tile(fit.centre[0], fit.centre[1], fit.turn_deg, tuple((float(x), float(y)) for x, y in corners),
                (px, py), fit.contrast)
