"""The 25 mm tiles in a camera frame: where each one's top face is on the desk, and how it is turned.

    camera = Camera.load(Path("config/camera.json"))
    for tile in find_tiles(frame, camera, height_mm=2.0):
        print(tile.x, tile.y, tile.turn)

The work is done in desk millimeters, on the frame resampled onto the plane of
the tiles' top faces, `height_mm` up (`topdown`), where every tile is the same square:

1. `votes` -- every color edge votes for the center of a 25 mm square it could
   be a side of; the places most voted for are where tiles may be.
2. `pose`  -- a square fitted to the edges round each of those places, its size
   a little free (`FACE_MM`).
3. Checks. Each side of a tile stands out from what lies just beyond it,
   measured the same way; its whole outline is in view; and tiles cannot
   overlap, so of two fits that do, the one that stands out more is kept.

Nothing assumes the tiles are lighter or darker than the desk, or any color at
all. Tiles may touch: the seam between two counts for both. The letters on them
are not read here.

The fits are most of the time taken: a cluttered view has many places voted
for, a handful of them tiles. A fit owes nothing to the fit at
another place, so the places are dealt out, a few at a time, to worker
processes on the other cores and to this one (`applib/crew.py`); the workers take the
view's edges up from a folder in memory, where they are kept for them. The fits
are the same to the last digit whoever makes them, and step 3 takes them in the
votes' order.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from maxarm import Camera
from maxarm.geometry import BASE_FAN_DEG, bearing_degrees
from maxarm.path import BASE_LIMIT_HALF_MM, REACH_MAX_MM

from .. import crew
from . import pose, topdown, votes
from .edges import EdgeField

TILE_MM = 25.0
# How thick a tile is, unless the app's settings say otherwise (`settings.py`): where its top face is looked for, and
# what the heights the cup grips, lets go and visits at are counted from. A tile taken for thinner than it is is
# placed further from the camera than it lies.
TILE_HEIGHT_MM = 2.0
# The top face as the camera sees it: larger than the tile, the more so the closer to the camera.
# The outline given (`Tile.corners`) stays the tile's own 25 mm.
FACE_MM = (TILE_MM, TILE_MM + 2.5)
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
# The places a worker is handed at a time: few, so that whoever is free takes the next and none is left waiting.
SHARE_PLACES = 3
# Where the view's edges are kept for the workers: in memory, where there is such a place, else among the
# temporary files.
KEPT_IN = "/dev/shm" if os.path.isdir("/dev/shm") else None

_taken_up: Optional[Tuple[str, EdgeField]] = None      # a worker's: the edges it took up last, and from where


@dataclass(frozen=True)
class Tile:
    """A tile's top face, desk mm, `height` over the desk. `turn` is the sides' direction, degrees from desk +x."""

    x: float
    y: float
    turn: float                                 # -45 to 45: a square looks the same a quarter turn on
    corners: Tuple[Tuple[float, float], ...]    # the face as if its corners were sharp
    pixel: Tuple[float, float]                  # where the frame shows the face's center
    contrast: float                             # how far its weakest side stands out from beyond it
    height: float                               # the tile's thickness it was looked for at

    def to_json(self) -> dict:
        return {"x": round(self.x, 2), "y": round(self.y, 2), "turn": round(self.turn, 1),
                "corners": [[round(x, 2), round(y, 2)] for x, y in self.corners],
                "pixel": [round(self.pixel[0], 1), round(self.pixel[1], 1)], "contrast": round(self.contrast, 2),
                "height": self.height}


def find_tiles(frame: np.ndarray, camera: Camera, height_mm: float = TILE_HEIGHT_MM) -> List[Tile]:
    """Every tile `height_mm` thick the frame shows round the arm, left to right as seen from behind it."""
    camera.check_frame(frame)
    view = topdown.render(frame, camera, height_mm, (-SEARCH_MM, SEARCH_MM, -SEARCH_MM, SEARCH_MM))
    field = EdgeField(view, camera.position())
    places = [(candidate.x, candidate.y) for candidate in votes.candidates(field, TILE_MM / 2, MAX_CANDIDATES)]
    places = [place for place in places if _is_searched(place)]
    fits: List[pose.Fit] = []
    for place, fit in zip(places, _fit_each(field, places)):
        if fit is not None and not _is_on(fits, place) and fit.is_seen and fit.contrast >= MIN_CONTRAST:
            fits.append(fit)
    tiles = [_tile(fit, camera, height_mm) for fit in _without_overlaps(fits)]
    return sorted(tiles, key=lambda tile: math.atan2(tile.x, -tile.y))


def _fit_each(field: EdgeField, places: Sequence[Tuple[float, float]]) -> List[Optional[pose.Fit]]:
    """The square fitted at each of `places`, in their order: None where the frame does not show all of one."""
    if not crew.WORKERS:
        return [_fit_at(field, place) for place in places]
    shares = [places[first:first + SHARE_PLACES] for first in range(0, len(places), SHARE_PLACES)]
    with tempfile.TemporaryDirectory(prefix="tiles-", dir=KEPT_IN) as folder:
        field.keep(Path(folder))
        fitted = crew.run(_fit_share, [(folder, share) for share in shares],
                          here=lambda _, share: [_fit_at(field, place) for place in share])
    return [fit for share in fitted for fit in share]


def _fit_share(folder: str, places: Sequence[Tuple[float, float]]) -> List[Optional[pose.Fit]]:
    """A worker's part of `_fit_each()`, with the view's edges as they were kept in `folder`."""
    global _taken_up
    if _taken_up is None or _taken_up[0] != folder:
        _taken_up = (folder, EdgeField.kept(Path(folder)))
    return [_fit_at(_taken_up[1], place) for place in places]


def _fit_at(field: EdgeField, place: Tuple[float, float]) -> Optional[pose.Fit]:
    rough = pose.rough_fit(field, place, TILE_MM)
    return pose.close_fit(field, rough, FACE_MM) if rough.is_seen else None


def _is_searched(place: Tuple[float, float]) -> bool:
    """Round the arm, in front of it, and not where its base stands."""
    x, y = place
    return (math.hypot(x, y) <= SEARCH_MM and abs(bearing_degrees(x, y)) <= SEARCH_BEARING_DEG
            and max(abs(x), abs(y)) > BASE_LIMIT_HALF_MM)


def _is_on(fits: List[pose.Fit], place: Tuple[float, float]) -> bool:
    """Whether `place` is well inside the face of a tile already fitted."""
    for fit in fits:
        turn = math.radians(fit.turn_deg)
        dx, dy = place[0] - fit.center[0], place[1] - fit.center[1]
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
    return pose.square_corners(fit.center, math.radians(fit.turn_deg), TILE_MM).astype(np.float32)


def _tile(fit: pose.Fit, camera: Camera, height_mm: float) -> Tile:
    px, py, _ = camera.from_desk((fit.center[0], fit.center[1], height_mm))
    corners = pose.square_corners(fit.center, math.radians(fit.turn_deg), TILE_MM)
    return Tile(fit.center[0], fit.center[1], fit.turn_deg, tuple((float(x), float(y)) for x, y in corners),
                (px, py), fit.contrast, height_mm)
