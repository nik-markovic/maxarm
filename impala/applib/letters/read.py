"""Which capital each tile shows, and which way up it is on the desk.

    tiles = find_tiles(frame, camera)
    for tile, letter in zip(tiles, read_letters(frame, camera, tiles)):
        print(letter.letter if letter.is_read else "?", letter.upright, letter.confidence)

For each tile:

1. The ink (`ink.py`): none, and the tile is blank; a dot, and the letter's
   bottom right is that corner -- the owner's mark on the letters that are
   others turned, M and W.
2. The reader (`reader.py`) on the face at its four quarter turns: each
   capital's probability that turn is upright.
3. The letter: with a dot, the best capital at the quarter turn that puts the
   dot at the bottom right; without, the best capital at any quarter turn.
   It counts as read at `MIN_CONFIDENCE`; below, it is a guess, kept to show.

`upright` is where the letter's top points on the desk, degrees from desk +x,
like the tile's `turn`. A capital that reads the same turned -- an N upside down,
an O any way -- is upright at each of its `uprights`, so whatever sets it down
can take the nearest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from maxarm import Camera

from . import face, ink
from .reader import LETTERS, SAME_TURNED, SIZE, WINDOW_MM, Reader
from ..tiles import Tile

MIN_CONFIDENCE = 0.6


@dataclass(frozen=True)
class Letter:
    letter: str                         # "A" to "Z", the reader's best; "" if the tile is blank
    confidence: float                   # the reader's probability for it, at its quarter turn
    upright: Optional[float]            # where its top points, desk degrees; None if blank
    uprights: Tuple[float, ...]         # every way it reads upright: one, two (N, S, Z...) or four (O, X)
    has_dot: bool                       # a dot at its bottom right, which settled its way up
    runner_up: str                      # the best other capital at any quarter turn, "" if none
    runner_up_confidence: float
    ink_contrast: float                 # the strongest ink against the face round it

    @property
    def is_blank(self) -> bool:
        return not self.letter

    @property
    def is_read(self) -> bool:
        return bool(self.letter) and self.confidence >= MIN_CONFIDENCE

    def to_json(self) -> dict:
        return {"letter": self.letter, "is_read": self.is_read, "confidence": round(self.confidence, 3),
                "upright": None if self.upright is None else round(self.upright, 1),
                "uprights": [round(angle, 1) for angle in self.uprights], "has_dot": self.has_dot,
                "runner_up": self.runner_up, "runner_up_confidence": round(self.runner_up_confidence, 3),
                "ink": round(self.ink_contrast, 2)}


_READER: Optional[Reader] = None


def read_letters(frame: np.ndarray, camera: Camera, tiles: Sequence[Tile],
                 reader: Optional[Reader] = None) -> List[Letter]:
    """Each tile's letter, in the order given; `frame` the one the tiles were found in."""
    global _READER
    if reader is None:
        _READER = _READER or Reader()
        reader = _READER
    return [read_tile(frame, camera, tile, reader) for tile in tiles]


def read_tile(frame: np.ndarray, camera: Camera, tile: Tile, reader: Reader) -> Letter:
    found = ink.measure(face.window(frame, camera, tile, 0, ink.WINDOW_MM, ink.size_px()))
    if found.is_blank:
        return Letter("", 0.0, None, (), False, "", 0.0, found.contrast)
    scores = np.array([reader.probabilities(face.window(frame, camera, tile, quarter, WINDOW_MM, SIZE))
                       for quarter in range(4)])[:, :len(LETTERS)]
    if found.dot:
        quarter = face.quarter_with(*found.dot)
        index = int(np.argmax(scores[quarter]))
        quarters: Tuple[int, ...] = (quarter,)
    else:
        quarter, index = (int(value) for value in np.unravel_index(np.argmax(scores), scores.shape))
        quarters = tuple(sorted((quarter + turn) % 4 for turn in range(4)
                                if turn == 0 or LETTERS[index] in SAME_TURNED[turn]))
    others = np.delete(scores, index, axis=1)
    other_quarter, other_index = np.unravel_index(np.argmax(others), others.shape)
    runner_up = LETTERS[other_index + (other_index >= index)]
    return Letter(LETTERS[index], float(scores[quarter, index]), face.upright_deg(tile, quarter),
                  tuple(face.upright_deg(tile, each) for each in quarters), found.dot is not None,
                  runner_up, float(others[other_quarter, other_index]), found.contrast)
