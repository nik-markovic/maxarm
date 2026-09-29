"""Every tile in a frame: its letter, where it is in arm millimetres, and which way it faces.

    mapping = DeskMapping.load(Path("config/calibration.json"))
    for tile in find_tiles(frame, mapping, Reader()):
        print(tile.letter, tile.centre_mm, tile.baseline_deg)

The steps, each in its own module:

1. `topdown`  -- the frame resampled onto the tiles' top plane, 4 mm up.
2. `glyphs`   -- printed letters: ink on a quiet face. One per tile.
3. `pose`     -- the tile's outline around each letter, from the edges that face
                 away from the camera, with the side the camera sees predicted.
4. `crop`     -- the top face, square-on, straight from the camera frame.
5. `read`     -- the letter, and which way up it is.
6. `pose` again, if the letter says the tile's long side runs the other way.

Then two checks, neither tuned on a frame. A tile is only a tile if some turn
of it reads as an upright letter more likely than not -- the reader was taught
tile edges, shadows, bare desk and blank faces as "not an upright letter". And
tiles cannot overlap: best reading first, and one that overlaps an accepted
tile by more than a quarter of its face is the same tile found twice, usually
from its own shadow line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

import crop
import glyphs
import pose
import read
import topdown
from mapping import DeskMapping

TILE_HEIGHT_MM = pose.TILE_THICKNESS_MM
MIN_LETTER_PROBABILITY = 0.5
MAX_OVERLAP = 0.25


@dataclass(frozen=True)
class Tile:
    letter: str
    confidence: float
    centre_mm: Tuple[float, float]      # AACS x, y of the top face's centre
    baseline_deg: float                 # AACS direction the letter's baseline runs, -180..180
    corners_mm: np.ndarray              # top face, AACS x, y, at TILE_HEIGHT_MM
    edge_contrast: float                # outline strength over the desk's own texture


def find_tiles(frame: np.ndarray, mapping: DeskMapping, reader: read.Reader) -> List[Tile]:
    view = topdown.render(frame, mapping, TILE_HEIGHT_MM)
    field = pose.EdgeField(view, mapping.camera_position())
    tiles = []
    for glyph in glyphs.find_glyphs(view):
        fit = pose.fit_tile(field, view.to_ground(glyph.centre))
        reading = reader.read(_faces(frame, mapping, fit))
        if reading.confidence < MIN_LETTER_PROBABILITY:
            continue
        baseline = fit.angle_deg + 90.0 * reading.turn
        if reading.turn % 2:
            fit = pose.fit_tile(field, fit.centre_mm, about_deg=baseline % 180.0)
            # Keep the letter's sense of direction on the refitted axis.
            baseline = fit.angle_deg + 180.0 * round(((baseline - fit.angle_deg) % 360.0) / 180.0)
        baseline = (baseline + 180.0) % 360.0 - 180.0
        tiles.append(Tile(reading.letter, reading.confidence, fit.centre_mm, baseline,
                          pose.tile_corners(np.array(fit.centre_mm), np.radians(baseline)), fit.contrast))
    return _without_overlaps(tiles)


def _without_overlaps(tiles: List[Tile]) -> List[Tile]:
    kept: List[Tile] = []
    area = pose.TILE_MM[0] * pose.TILE_MM[1]
    for tile in sorted(tiles, key=lambda t: (-t.confidence, -t.edge_contrast)):
        face = tile.corners_mm.astype(np.float32)
        if all(cv2.intersectConvexConvex(face, other.corners_mm.astype(np.float32))[0]
               <= MAX_OVERLAP * area for other in kept):
            kept.append(tile)
    return kept


def _faces(frame: np.ndarray, mapping: DeskMapping, fit: pose.TilePose) -> List[np.ndarray]:
    faces = []
    for turn in range(4):
        face = crop.tile_face(frame, mapping, fit.centre_mm, fit.angle_deg, TILE_HEIGHT_MM, turn,
                              window_mm=read.WINDOW_MM, px_per_mm=read.INPUT_SIZE / read.WINDOW_MM)
        faces.append(face.mean(axis=2).astype(np.uint8))
    return faces
