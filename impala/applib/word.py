"""A word set down in the dropoff area: which tile for each letter, where it goes, and how it turns.

    arm.use_coordinates(Space.DESK)
    arm.set_clearance(CLEARANCE_MM)
    for step in plan_word("TEAM", tiles, letters, places, arm.can_reach, grip_mm=2.0, drop_mm=6.0):
        arm.carry(step.start, step.end, angle=step.upright, facing=step.facing)

The tiles on offer are those ready to pick up (`areas.Status.READY`) with their
letter read, each once. The word's letters go to `areas.word_slots()`, first to
last as the word reads, each turned the way the slots face; of several tiles
with a letter, the one that needs the least turning goes. Nothing found in the
snapshot may lie where a letter goes. Every tile's place is the snapshot's, so
a word is planned on one snapshot and set down before anything else moves.
The heights the cup grips and lets go at are the caller's.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from string import ascii_uppercase
from typing import Callable, List, Sequence

from maxarm.geometry import Position

from . import areas
from .letters import Letter
from .tiles import Tile

# Nothing travels sideways under this, desk mm, on the way: the tiles are 2 mm. The library keeps 5 mm over it,
# and over the height of what the cup carries.
CLEARANCE_MM = 5.0


class WordRefused(ValueError):
    """The word cannot be set down; the message says why."""


@dataclass(frozen=True)
class Step:
    """One letter of the word: the tile taken, the cup's points at the grip and the release, and the letter's turn."""

    index: int               # the tile's place in the finder's list
    letter: str
    start: Position          # the tile's centre, at the grip's height
    end: Position            # its slot, at the release's height
    upright: float           # where the letter's top points now, desk degrees: of its ways up, the nearest `facing`
    facing: float            # where it is to point once set down
    slot: areas.Slot


def letters_on_offer(letters: Sequence[Letter], places: Sequence[areas.Place]) -> List[int]:
    """The tiles a word may take: ready to pick up, their letter read. Their places in the finder's list."""
    return [index for index, (letter, place) in enumerate(zip(letters, places))
            if place.status is areas.Status.READY and letter.is_read]


def plan_word(text: str, tiles: Sequence[Tile], letters: Sequence[Letter], places: Sequence[areas.Place],
              can_reach: Callable[[float, float, float], bool], grip_mm: float, drop_mm: float) -> List[Step]:
    """How to set `text` down, a step per letter, first to last; raises `WordRefused`.

    `tiles`, `letters` and `places` are one snapshot's, in the finder's order;
    `can_reach(x, y, z)` takes desk mm, as `areas.place_tiles()` does. The cup
    grips each tile at desk height `grip_mm` and lets it go at `drop_mm`.
    """
    wanted = "".join(text.split()).upper()
    if not wanted:
        raise WordRefused("no letters given")
    strange = sorted({character for character in wanted if character not in ascii_uppercase})
    if strange:
        raise WordRefused(f"only the letters A to Z go on tiles, not {' '.join(strange)}")
    offer = letters_on_offer(letters, places)
    short = _short(wanted, [letters[index].letter for index in offer])
    if short:
        raise WordRefused(short)
    slots = areas.word_slots(len(wanted))
    if not slots:
        raise WordRefused(f"{len(wanted)} letters do not fit along the dropoff arc, "
                          f"{areas.WORD_MAX_LETTERS} at most")
    _check_slots(slots, tiles, can_reach)
    steps, free = [], list(offer)
    for character, slot in zip(wanted, slots):
        index = min((index for index in free if letters[index].letter == character),
                    key=lambda index: _turn(letters[index], slot.facing))
        free.remove(index)
        upright = min(letters[index].uprights, key=lambda way: abs(_wrapped(slot.facing - way)))
        tile = tiles[index]
        steps.append(Step(index, character, (tile.x, tile.y, grip_mm), (slot.x, slot.y, drop_mm), upright,
                          slot.facing, slot))
    return steps


def _short(wanted: str, offered: List[str]) -> str:
    """What the word needs that the tiles on offer do not hold, said; "" if nothing."""
    need, have = Counter(wanted), Counter(offered)
    missing = [letter if not have[letter] else f"{letter} {need[letter]} times ({have[letter]} on offer)"
               for letter in sorted(need) if need[letter] > have[letter]]
    if not missing:
        return ""
    held = " ".join(sorted(offered)) if offered else "none"
    return f"not on the tiles ready to pick up: {', '.join(missing)}; their letters are {held}"


def _check_slots(slots: Sequence[areas.Slot], tiles: Sequence[Tile],
                 can_reach: Callable[[float, float, float], bool]) -> None:
    for number, slot in enumerate(slots, 1):
        problem = areas.reach_problem(slot.x, slot.y, can_reach)
        if problem:
            raise WordRefused(f"letter {number} would go where the arm cannot set it down: {problem}")
    in_the_way = sorted({index + 1 for index, tile in enumerate(tiles) for slot in slots
                         if areas.gap_mm(slot, tile) < areas.MIN_GAP_MM})
    if in_the_way:
        raise WordRefused(f"the word's place along the dropoff arc is not clear: tile "
                          f"{', '.join(str(number) for number in in_the_way)} in the way")


def _turn(letter: Letter, facing: float) -> float:
    return min(abs(_wrapped(facing - way)) for way in letter.uprights)


def _wrapped(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0
