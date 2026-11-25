#!/usr/bin/env python3
"""Raw to true and back as one piece that can be swapped: the formula, this arm's corrections on it, or a model.

    from maxarm import converter
    converter.use(model)        # anything with to_true(raw) and to_raw(true): (x, y, z) mm, arm space
    converter.use(None)         # back to the default
    converter.current()         # what the library converts with now; converter.describe() says it in words

Every position the library sends or reads goes through `raw.raw_position()` and
`raw.true_position()`, and they ask `current()`. The default is the formula
(`raw.py`, the same for every MaxArm) with this arm's corrections on top, as the
camera calibration measured them: `config/arm-true.json`, written by CALIBRATE and
SAVE CALIBRATION, read again whenever it changes, so that a calibration applies at
once in every process using the library. Without the file, the formula alone.

A correction is anything with `at(raw, formula_true) -> (dx, dy, dz)`: how far, true
mm, the cup is from where the formula puts it, given the firmware's position -- the
servos' angles follow from it (`geometry.inverse()`) -- and the formula's true one.
The file names its kind; `register()` adds a kind, a model trained on the arm say,
with the loader that reads it. `Corrected` makes a converter of a correction: to
true it adds the correction; to raw it finds, by repeating, the raw position whose
true one is the target.
"""

import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Protocol, Tuple

from . import mesh, raw
from .geometry import Position

CORRECTION_FILE = Path(__file__).resolve().parents[2] / "config" / "arm-true.json"   # config/, beside src/
LOOK_EVERY_S = 1.0          # how often the file is looked at for a change; reload() looks at once
TO_RAW_ROUNDS = 8           # a correction changing 3 mm in 100 brings each round ~30 times closer


class Converter(Protocol):
    def to_true(self, raw_position: Position) -> Position: ...

    def to_raw(self, true_position: Position) -> Position: ...


class Correction(Protocol):
    def at(self, raw_position: Position, formula_true: Position) -> Tuple[float, float, float]: ...

    def describe(self) -> str: ...


class Formula:
    """The formula alone (`raw.py`)."""

    def to_true(self, raw_position: Position) -> Position:
        return raw.formula_true(raw_position)

    def to_raw(self, true_position: Position) -> Position:
        return raw.formula_raw(true_position)

    def describe(self) -> str:
        return "the formula alone"


class Corrected:
    """A converter on top of another, `base` (the formula by default), corrected by `correction`."""

    def __init__(self, correction: Correction, base: Optional[Converter] = None) -> None:
        self.correction = correction
        self.base = base or FORMULA

    def to_true(self, raw_position: Position) -> Position:
        true = self.base.to_true(raw_position)
        return _plus(true, self.correction.at(raw_position, true))

    def to_raw(self, true_position: Position) -> Position:
        estimate = true_position                                   # the base's true position, found by repeating
        for _ in range(TO_RAW_ROUNDS):
            off = self.correction.at(self.base.to_raw(estimate), estimate)
            estimate = _plus(true_position, tuple(-value for value in off))
        return self.base.to_raw(estimate)

    def describe(self) -> str:
        return f"the formula with {self.correction.describe()}"


FORMULA = Formula()
KINDS: Dict[str, Callable[[dict], Correction]] = {"mesh": mesh.Mesh.from_json}

_chosen: Optional[Converter] = None
_loaded: Tuple[Optional[int], Converter, Optional[str]] = (None, FORMULA, None)   # file stamp, converter, problem
_looked = -LOOK_EVERY_S


def register(kind: str, loader: Callable[[dict], Correction]) -> None:
    """A kind of correction the file may name, and what makes one of what the file holds."""
    KINDS[kind] = loader


def use(chosen: Optional[Converter]) -> None:
    """Convert with `chosen` from now on, process-wide; None: the default, the formula and the file."""
    global _chosen
    _chosen = chosen


def current() -> Converter:
    if _chosen is not None:
        return _chosen
    if time.monotonic() - _looked >= LOOK_EVERY_S:
        reload()
    return _loaded[1]


def reload() -> None:
    """Look at the file now: after writing it, so the next position read or sent goes through it."""
    global _loaded, _looked
    _looked = time.monotonic()
    try:
        stamp = os.stat(CORRECTION_FILE).st_mtime_ns
    except OSError:
        stamp = None
    if stamp == _loaded[0]:
        return
    if stamp is None:
        _loaded = (None, FORMULA, None)
        return
    try:
        _loaded = (stamp, load(CORRECTION_FILE), None)
    except (OSError, ValueError, KeyError, TypeError) as reason:     # the formula alone, and why
        _loaded = (stamp, FORMULA, f"{CORRECTION_FILE.name} not used: {reason}")


def load(path: Path) -> Converter:
    """The converter a correction file makes: the formula with the correction it holds."""
    stored = json.loads(Path(path).read_text())
    kind = stored.get("kind")
    if kind not in KINDS:
        raise ValueError(f"no correction of kind {kind!r}; known: {', '.join(KINDS)}")
    return Corrected(KINDS[kind](stored))


def page_mesh() -> Optional[dict]:
    """The mesh the default converter corrects by, as a page redraws it (`Mesh.to_page()`); None without one, or with
    another converter in use."""
    chosen = current()
    correction = getattr(chosen, "correction", None)
    return correction.to_page() if isinstance(correction, mesh.Mesh) else None


def describe() -> str:
    """What positions are converted with now, in words."""
    chosen = current()
    problem = None if _chosen is not None else _loaded[2]
    text = chosen.describe() if hasattr(chosen, "describe") else type(chosen).__name__
    return f"{text} ({problem})" if problem else text


def _plus(position: Position, off) -> Position:
    return (position[0] + off[0], position[1] + off[1], position[2] + off[2])
