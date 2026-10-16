"""Every tile in a frame: its letter, where it is, and which way it faces.

    camera = Camera.load(Path("config/camera.json"))
    for tile in find_tiles(frame, camera, Reader()):
        arm.grab(tile.px, tile.py, TILE_HEIGHT_MM, angle=tile.angle)

A tile comes back in camera space -- the pixel its top face is seen at, and the
way its letter runs in the frame -- which an arm in `Space.CAMERA` takes as is.
The work is done in desk millimetres, on a top-down view; the library's camera
is what makes that view, and what turns the answer back into pixels.

The steps, each in its own module:

1. `topdown`  -- the frame resampled onto the tiles' top plane, 4 mm up.
2. `glyphs`   -- printed letters: ink on a quiet face. One per tile.
3. `pose`     -- the tile's outline around each letter, from the edges that face
                 away from the camera, with the side the camera sees predicted.
4. `crop`     -- the top face, square-on, straight from the camera frame.
5. `read`     -- the letter, and which way up it is.
6. `pose` again, if the letter says the tile's long side runs the other way.

Then three checks, none tuned on a frame. A tile is only a tile if some turn
of it reads as an upright letter more likely than not -- the reader was taught
tile edges, shadows, bare desk and blank faces as "not an upright letter". Its
letter is printed at the face's centre, so an outline whose centre is further
from the letter than the fit was allowed to search is one that slid onto a
neighbour's edges, and is dropped: better missed than sent to the arm in the
wrong place. And tiles cannot overlap: best reading first, and one that
overlaps an accepted tile by more than a quarter of its face is the same tile
found twice, usually from its own shadow line.

**Tiles may touch at their corners, not along their edges.** Where two sit
snug, the seam between two pale faces barely shows and the outline is a
millimetre out at best; in a clump it is lost. The arm could not lift one
cleanly from a clump either. So the scene is checked rather than trusted:
every letter the camera can read must end up on a tile, and if one does not,
`find_tiles` raises `UnplacedLetters` -- what it did place, and where the
letters it could not are -- for the user to spread the tiles and try again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from maxarm import Camera
from maxarm.spaces import angle_from_desk

from . import crop, glyphs, pose, read, topdown

TILE_HEIGHT_MM = pose.TILE_THICKNESS_MM
MIN_LETTER_PROBABILITY = 0.5
MAX_OVERLAP = 0.25
# A letter no tile was placed on is read from a crop on the letter itself, turned
# through a quarter turn in these steps: the reader was trained to +-5 degrees.
LETTER_SEARCH_STEP_DEG = 10.0
# Only outside every placed tile by this much. A real letter's own tile cannot
# overlap a placed one, so the letter -- within 1.7 mm of its tile's centre, at
# least 8.75 mm from its edge -- is 7 mm or more outside it. Nearer is the placed
# tile's own rim or shadow: on white paper those read as a Z at 1-3 mm out.
LETTER_CLEARANCE_MM = 5.0


@dataclass(frozen=True)
class Tile:
    """One tile. `px`, `py` and `angle` are camera space; the desk values are for drawing.

    `angle` is the way the letter's baseline runs in the frame, degrees from +px
    towards +py: 0 reads upright to the camera.
    """

    letter: str
    confidence: float
    px: float                           # where the top face's centre is seen,
    py: float                           # at TILE_HEIGHT_MM
    angle: float
    desk_centre: Tuple[float, float]    # the same, in desk millimetres
    desk_angle: float                   # -180..180
    desk_corners: np.ndarray            # top face, at TILE_HEIGHT_MM
    edge_contrast: float                # outline strength over the desk's own texture


@dataclass(frozen=True)
class Letter:
    """A letter the camera reads but no tile was placed on."""
    letter: str
    desk_centre: Tuple[float, float]


class UnplacedLetters(Exception):
    """Letters with no tile under them: tiles touching along an edge. Spread them and look again."""

    def __init__(self, tiles: List[Tile], letters: List[Letter]):
        self.tiles, self.letters = tiles, letters
        where = ", ".join(f"{letter.letter} at ({letter.desk_centre[0]:.0f}, {letter.desk_centre[1]:.0f})"
                          for letter in letters)
        super().__init__(f"{len(letters)} letter(s) could not be placed on a tile: {where} desk mm. "
                         f"Tiles may touch at their corners but not along their edges.")


def find_tiles(frame: np.ndarray, camera: Camera, reader: read.Reader) -> List[Tile]:
    view = topdown.render(frame, camera, TILE_HEIGHT_MM)
    field = pose.EdgeField(view, camera.position())
    found = glyphs.find_glyphs(view)
    tiles = []
    for glyph in found:
        fit = pose.fit_tile(field, view.to_ground(glyph.centre))
        reading = reader.read(_faces(frame, camera, fit))
        if reading.confidence < MIN_LETTER_PROBABILITY:
            continue
        baseline = fit.angle_deg + 90.0 * reading.turn
        if reading.turn % 2:
            fit = pose.fit_tile(field, fit.centre_mm, about_deg=baseline % 180.0)
            # Keep the letter's sense of direction on the refitted axis.
            baseline = fit.angle_deg + 180.0 * round(((baseline - fit.angle_deg) % 360.0) / 180.0)
        if _letter_offset_mm(glyph, view, fit) > pose.SEARCH_MM:
            continue
        baseline = (baseline + 180.0) % 360.0 - 180.0
        top = (*fit.centre_mm, TILE_HEIGHT_MM)
        px, py, _ = camera.from_desk(top)
        tiles.append(Tile(reading.letter, reading.confidence, px, py,
                          angle_from_desk(camera, top, baseline), fit.centre_mm, baseline,
                          pose.tile_corners(np.array(fit.centre_mm), np.radians(baseline)), fit.contrast))
    tiles = _without_overlaps(tiles)
    unplaced = _unplaced_letters(frame, camera, reader, view, found, tiles)
    if unplaced:
        raise UnplacedLetters(tiles, unplaced)
    return tiles


def _unplaced_letters(frame: np.ndarray, camera: Camera, reader: read.Reader, view: topdown.TopDown,
                      found: List[glyphs.Glyph], tiles: List[Tile]) -> List[Letter]:
    """Glyphs clear of every placed tile that read as a letter on a crop centred on themselves."""
    faces = [tile.desk_corners.astype(np.float32) for tile in tiles]
    letters = []
    for glyph in found:
        centre = view.to_ground(glyph.centre)
        if any(cv2.pointPolygonTest(face, tuple(float(c) for c in centre), True) > -LETTER_CLEARANCE_MM
               for face in faces):
            continue
        best = max((reader.read(_faces_at(frame, camera, centre, angle))
                    for angle in np.arange(0.0, 90.0, LETTER_SEARCH_STEP_DEG)),
                   key=lambda reading: reading.confidence)
        if best.confidence >= MIN_LETTER_PROBABILITY:
            letters.append(Letter(best.letter, (float(centre[0]), float(centre[1]))))
    return letters


def _letter_offset_mm(glyph: glyphs.Glyph, view: topdown.TopDown, fit: pose.TilePose) -> float:
    """How far the letter's box, squared to the tile, lies from the tile's centre.

    The box rather than the ink's centroid: an L's centroid sits 2.4 mm off,
    its box 0.6. On the first two scenes' 27 tiles the box is at most 1.7 mm off.
    """
    outline = np.array([view.to_ground(tuple(point)) for point in glyph.contour.reshape(-1, 2).astype(float)])
    angle = np.radians(fit.angle_deg)
    axes = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    along = (outline - np.array(fit.centre_mm)) @ axes.T
    return float(np.hypot(*((along.max(axis=0) + along.min(axis=0)) / 2)))


def _without_overlaps(tiles: List[Tile]) -> List[Tile]:
    kept: List[Tile] = []
    area = pose.TILE_MM[0] * pose.TILE_MM[1]
    for tile in sorted(tiles, key=lambda t: (-t.confidence, -t.edge_contrast)):
        face = tile.desk_corners.astype(np.float32)
        if all(cv2.intersectConvexConvex(face, other.desk_corners.astype(np.float32))[0]
               <= MAX_OVERLAP * area for other in kept):
            kept.append(tile)
    return kept


def _faces(frame: np.ndarray, camera: Camera, fit: pose.TilePose) -> List[np.ndarray]:
    return _faces_at(frame, camera, fit.centre_mm, fit.angle_deg)


def _faces_at(frame: np.ndarray, camera: Camera, centre_mm, angle_deg: float) -> List[np.ndarray]:
    faces = []
    for turn in range(4):
        face = crop.tile_face(frame, camera, centre_mm, angle_deg, TILE_HEIGHT_MM, turn,
                              window_mm=read.WINDOW_MM, px_per_mm=read.INPUT_SIZE / read.WINDOW_MM)
        faces.append(face.mean(axis=2).astype(np.uint8))
    return faces
