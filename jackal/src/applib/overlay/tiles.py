"""The tiles drawn over the camera's picture, as the Tiles page and the stream both show them.

    draw_tiles(canvas, view, scene)                 # scene: TileWork.scene()

- the work areas: pickup green, dropoff blue, on the desk; the line between them
  and the arm's centre line, dashed; the arc a word is set down along, dotted;
  the arm's reach, faint;
- each tile found: its top face outlined in its status's colour, thin and
  see-through so the tile's own edge shows beside it; its letter on a badge at
  the face's corner at the letter's bottom right, so the badge says which way up
  the letter lies. A letter not read is a '?' over the tile, ringed in white, the
  outline and the badge red; a blank tile a '-', ringed;
- where a tile a word took was: dotted, grey, its letter faint at the same corner
  and a chevron to where the letter's top pointed -- to put it back by;
- where each letter of the word goes: the tile's face, dashed until the tile is
  there, then filled, its letter at the letter's bottom right corner.

Everything is drawn at the tiles' top faces' height: what the camera sees of them.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from maxarm.geometry import BASE_FAN_DEG
from maxarm.path import REACH_MAX_MM, REACH_MIN_MM

from applib import areas
from applib.tiles import TILE_HEIGHT_MM

from . import inks
from .canvas import Canvas, Point, View, ink

BADGE_RADIUS = 9.0
BADGE_FONT = 12.0
# A '?' or '-' badge stands this far over the face's top corner in the picture, design px.
BADGE_OVER = 12.0


def draw_tiles(canvas: Canvas, view: View, scene: dict, show_areas: bool = True) -> None:
    if show_areas:
        draw_areas(canvas, view)
        draw_reach(canvas, view, 0.35)
    if scene.get("word"):
        _draw_word(canvas, view, scene["word"])
    found = scene.get("found")
    if not found:
        return
    moved = set(scene.get("moved") or ())
    for tile in found.get("carried") or []:
        _draw_carried(canvas, view, tile)
    for tile in found["tiles"]:
        if tile["n"] in moved:
            _draw_carried(canvas, view, tile)
        else:
            _draw_tile(canvas, view, tile)


def draw_areas(canvas: Canvas, view: View) -> None:
    for area in (areas.PICKUP, areas.DROPOFF):
        start, end = _bearings(area.side)
        outline = view.arc(area.outer_mm, start, end) + view.arc(area.inner_mm, end, start)
        colour = ink(inks.AREA[area.name])
        canvas.fill(outline, colour, 0.13)
        canvas.lines(outline, colour, 1.2, 0.55, closed=True)
    white = inks.BGR["white"]
    canvas.lines(view.arc(areas.PICKUP.inner_mm, -BASE_FAN_DEG, BASE_FAN_DEG), white, 1.5, 0.7, dash=(7, 5))
    canvas.lines([view.at(0.0, -areas.DROPOFF.inner_mm), view.at(0.0, -areas.PICKUP.outer_mm)], white, 1.5, 0.7,
                 dash=(7, 5))
    canvas.lines(view.arc(areas.WORD_ARC_MM, -BASE_FAN_DEG, BASE_FAN_DEG), ink(inks.AREA["dropoff"]), 1.5, 0.8,
                 dash=(2, 6))


def draw_reach(canvas: Canvas, view: View, opacity: float = 0.75) -> None:
    """The arm's reach on the desk: 95 and 270 mm, orange and red."""
    for radius, colour in ((REACH_MAX_MM, inks.BGR["reach"]), (REACH_MIN_MM, inks.BGR["reach_min"])):
        canvas.lines(view.arc(radius, -BASE_FAN_DEG, BASE_FAN_DEG), colour, 2, opacity)


def letter_badge(tile: dict) -> Tuple[str, Optional[Tuple[float, float]], bool]:
    """What a tile's badge shows, the face's corner it sits at, and whether it is ringed: the letter at the corner
    at its bottom right -- 135 degrees on from where its top points -- or '?' or '-' over the tile, ringed."""
    if "letter" not in tile:
        return str(tile["n"]), None, False
    if not tile["letter"]:
        return "-", None, True
    if not tile["is_read"]:
        return "?", None, True
    toward = math.radians(tile["upright"] + 135.0)
    corner = max(tile["corners"], key=lambda each: (each[0] - tile["x"]) * math.cos(toward)
                 + (each[1] - tile["y"]) * math.sin(toward))
    return tile["letter"], (corner[0], corner[1]), False


def chevron_up(tile: dict) -> List[Tuple[float, float]]:
    """A chevron on a tile's face, its tip towards where the letter's top points, desk mm; none for a letter not read
    or a blank tile."""
    if not tile.get("letter") or not tile.get("is_read"):
        return []
    angle = math.radians(tile["upright"])
    ux, uy = math.cos(angle), math.sin(angle)
    return [(tile["x"] + along * ux - across * uy, tile["y"] + along * uy + across * ux)
            for along, across in ((1.0, -6.0), (8.0, 0.0), (1.0, 6.0))]


def is_unread(tile: dict) -> bool:
    return bool(tile.get("letter")) and not tile.get("is_read")


def _draw_tile(canvas: Canvas, view: View, tile: dict) -> None:
    face = view.path(tile["corners"], TILE_HEIGHT_MM)
    if any(corner is None for corner in face):
        return
    colour = ink(inks.UNREAD if is_unread(tile) else inks.STATUS[tile["status"]])
    canvas.lines(face, colour, 1.0, 0.6, closed=True)
    text, corner, is_ringed = letter_badge(tile)
    if corner is not None:
        at = view.at(corner[0], corner[1], TILE_HEIGHT_MM)
    else:
        top = min(face, key=lambda point: point[1])
        at = (top[0], top[1] - BADGE_OVER * canvas.scale)
    _badge(canvas, text, at, colour, is_ringed)


def _draw_carried(canvas: Canvas, view: View, tile: dict) -> None:
    """Where a tile a word took was, and how its letter lay."""
    face = view.path(tile["corners"], TILE_HEIGHT_MM)
    if any(corner is None for corner in face):
        return
    grey = inks.BGR["carried"]
    canvas.lines(face, grey, 1.2, 0.75, dash=(2, 3), closed=True)
    canvas.lines(view.path(chevron_up(tile), TILE_HEIGHT_MM), grey, 1.5, 0.75)
    text, corner, is_ringed = letter_badge(tile)
    x, y = corner if corner is not None else (tile["x"], tile["y"])
    _badge(canvas, text, view.at(x, y, TILE_HEIGHT_MM), grey, is_ringed, is_faint=True)


def _draw_word(canvas: Canvas, view: View, word: dict) -> None:
    for step in word["steps"]:
        face = view.path(step["corners"], TILE_HEIGHT_MM)
        if any(corner is None for corner in face):
            continue
        colour = ink(inks.word_ink(step["status"]))
        if step["status"] == "placed":
            canvas.fill(face, colour, 0.3)
        canvas.lines(face, colour, 1.5, 0.9, closed=True, dash=None if step["status"] == "placed" else (5, 4))
        _, corner, _ = letter_badge({**step, "is_read": True, "upright": step["facing"]})
        _badge(canvas, step["letter"], view.at(corner[0], corner[1], TILE_HEIGHT_MM), colour, False)


def _badge(canvas: Canvas, text: str, at: Point, colour, is_ringed: bool, is_faint: bool = False) -> None:
    """A tile's letter on a small disc of its colour; ringed in white when there is no letter to show."""
    if at is None:
        return
    canvas.disc(at, BADGE_RADIUS, colour, 0.45 if is_faint else 0.92)
    if is_ringed:
        canvas.circle(at, BADGE_RADIUS, inks.BGR["white"], 2.0, 1.0)
    canvas.text(text, at, inks.BGR["badge_text"], BADGE_FONT, 0.75 if is_faint else 1.0, bold=True)


def _bearings(side: int) -> Tuple[float, float]:
    """The bearings an area spans, degrees from straight out: x < 0 is the arm's left, at negative bearings."""
    return (-BASE_FAN_DEG, 0.0) if side < 0 else (0.0, BASE_FAN_DEG) if side > 0 else (-BASE_FAN_DEG, BASE_FAN_DEG)

