"""Where the arm picks tiles up and where it sets them down, and which tiles it can take: the app's own layout.

    arm.use_coordinates(Space.DESK)
    for tile, place in zip(tiles, place_tiles(tiles, arm.can_reach)):
        if place.status is Status.READY: ...

Both areas are bands of the arm's reach, in desk mm round its base axis (the
owner's, PILOT-hyena, until the final app settles them):

- **pickup**: the outer two thirds of the reach, left of the arm's centre line
  (x < 0: the arm's left as it reaches out);
- **dropoff**: the inner third, on both sides.

A tile's centre is kept `REACH_MARGIN_MM` inside both of the arm's reach arcs,
so that the cup is never sent to the very edge of what the arm can do, and the
arm must be able to come down onto it: the library's `can_reach()`, at the tile's
top and over it. Each tile gets one status, the first of these that holds:

1. `OUT_OF_REACH` -- the arm cannot be sent to its centre;
2. `ELSEWHERE` -- in reach, but not in the pickup area;
3. `TOO_CLOSE` -- in the pickup area, but nearer than `MIN_GAP_MM` to another
   tile, which the cup coming down might push: every tile's place is the one
   the snapshot measured, so none may move before its turn;
4. `READY` -- the arm can pick it up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Sequence

import numpy as np

from maxarm.geometry import BASE_FAN_DEG, bearing_degrees
from maxarm.path import REACH_MAX_MM, REACH_MIN_MM

from .tiles import TILE_HEIGHT_MM, Tile

# A tile's centre stays this far inside both reach arcs: at the arcs themselves the arm is at its limit.
REACH_MARGIN_MM = 5.0
TILE_REACH_MM = (REACH_MIN_MM + REACH_MARGIN_MM, REACH_MAX_MM - REACH_MARGIN_MM)
# The dropoff area is this share of the reach, the inside of it; the pickup area the rest.
DROPOFF_SHARE = 1.0 / 3.0
DIVIDE_MM = TILE_REACH_MM[0] + (TILE_REACH_MM[1] - TILE_REACH_MM[0]) * DROPOFF_SHARE
# Which side of the arm's centre line the pickup area is: -1 for x < 0, the arm's left; 0 for both sides.
PICKUP_SIDE = -1
# Tiles in the pickup area are at least this far apart, edge to edge.
MIN_GAP_MM = 2.0
# The arm must reach the tile's top and this far over it, desk mm: it comes down onto the tile from there.
OVER_TILE_MM = 14.0


class Status(Enum):
    READY = "ready"
    TOO_CLOSE = "too close"
    ELSEWHERE = "elsewhere"
    OUT_OF_REACH = "out of reach"


@dataclass(frozen=True)
class Area:
    name: str
    inner_mm: float          # from the base axis, desk mm
    outer_mm: float
    side: int                # -1: x < 0 only; 1: x > 0 only; 0: both

    def contains(self, x: float, y: float) -> bool:
        reach = math.hypot(x, y)
        return (self.inner_mm <= reach <= self.outer_mm and abs(bearing_degrees(x, y)) <= BASE_FAN_DEG
                and (self.side == 0 or x * self.side > 0))

    def to_json(self) -> dict:
        return {"name": self.name, "inner": self.inner_mm, "outer": self.outer_mm, "side": self.side}


PICKUP = Area("pickup", DIVIDE_MM, TILE_REACH_MM[1], PICKUP_SIDE)
DROPOFF = Area("dropoff", TILE_REACH_MM[0], DIVIDE_MM, 0)


@dataclass(frozen=True)
class Place:
    """Where a tile is, for the arm."""

    status: Status
    area: str                # "pickup", "dropoff", or "" for neither
    reach_mm: float          # its centre from the arm's base axis
    detail: str              # why, in a few words, unless READY
    nearest: Optional[int]   # the closest other tile, its index, if any
    gap_mm: Optional[float]  # how far that one is, edge to edge; 0 if they touch or overlap

    def to_json(self) -> dict:
        return {"status": self.status.value, "area": self.area, "reach": round(self.reach_mm, 1),
                "detail": self.detail, "nearest": self.nearest,
                "gap": None if self.gap_mm is None else round(self.gap_mm, 1)}


def place_tiles(tiles: Sequence[Tile], can_reach: Callable[[float, float, float], bool]) -> List[Place]:
    """Each tile's status, in the order given. `can_reach(x, y, z)` takes desk mm: the library's, in DESK space."""
    places = []
    for index, tile in enumerate(tiles):
        nearest, gap = _nearest(tiles, index)
        area = next((area.name for area in (PICKUP, DROPOFF) if area.contains(tile.x, tile.y)), "")
        reach = math.hypot(tile.x, tile.y)
        problem = _reach_problem(tile, can_reach)
        if problem:
            status, detail = Status.OUT_OF_REACH, problem
        elif area != PICKUP.name:
            status, detail = Status.ELSEWHERE, ("in the dropoff area" if area else "outside the pickup area")
        elif gap is not None and gap < MIN_GAP_MM:
            status, detail = Status.TOO_CLOSE, f"{gap:.1f} mm from another tile, under {MIN_GAP_MM:.0f}"
        else:
            status, detail = Status.READY, ""
        places.append(Place(status, area, reach, detail, nearest, gap))
    return places


def _reach_problem(tile: Tile, can_reach: Callable[[float, float, float], bool]) -> str:
    reach = math.hypot(tile.x, tile.y)
    if reach < TILE_REACH_MM[0] or reach > TILE_REACH_MM[1]:
        return (f"{reach:.0f} mm from the base axis; tiles go {TILE_REACH_MM[0]:.0f} to {TILE_REACH_MM[1]:.0f}")
    if abs(bearing_degrees(tile.x, tile.y)) > BASE_FAN_DEG:
        return "behind the arm, where it cannot turn"
    if not all(can_reach(tile.x, tile.y, z) for z in (TILE_HEIGHT_MM, TILE_HEIGHT_MM + OVER_TILE_MM)):
        return "a pose the arm cannot hold"
    return ""


def _nearest(tiles: Sequence[Tile], index: int):
    """The tile nearest to `tiles[index]`, edge to edge, and how far; (None, None) if there is no other."""
    best = (None, None)
    for other_index, other in enumerate(tiles):
        if other_index != index:
            gap = gap_mm(tiles[index], other)
            if best[1] is None or gap < best[1]:
                best = (other_index, gap)
    return best


def gap_mm(a: Tile, b: Tile) -> float:
    """How far apart two tiles' faces are, edge to edge, their corners taken as sharp; 0 if they meet."""
    first, second = np.array(a.corners), np.array(b.corners)
    if _overlaps(first, second):
        return 0.0
    return min(min(_to_segment(point, start, end) for point in first for start, end in _sides(second)),
               min(_to_segment(point, start, end) for point in second for start, end in _sides(first)))


def _sides(corners: np.ndarray):
    return [(corners[index], corners[(index + 1) % len(corners)]) for index in range(len(corners))]


def _to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    along = end - start
    share = float(np.clip((point - start) @ along / (along @ along), 0.0, 1.0))
    return float(np.linalg.norm(point - (start + share * along)))


def _overlaps(first: np.ndarray, second: np.ndarray) -> bool:
    """Whether two convex polygons share any area: no side of either separates them."""
    for corners in (first, second):
        for start, end in _sides(corners):
            normal = np.array([end[1] - start[1], start[0] - end[0]])
            a, b = first @ normal, second @ normal
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True
