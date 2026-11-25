"""The app's own settings, which the calibration app's pages and the /IOTCONNECT app share: `config/appcfg.json`.

    settings = Settings(CONFIG_DIR)
    settings.get("tile_offset_mm")          # 1.0 until someone sets it
    settings.set(reader="left")             # written at once
    settings.to_json()                      # every setting, its value and its default, for a page

Every default is here, in Python, and nowhere else: the pages are told them. The file holds only what differs from
the defaults -- a setting set back to its default is taken out of it -- so a later default reaches every rig that
never changed it. A value the file holds that is not one a setting takes is ignored, the default used, and said
on the console. Each app reads the file again whenever it has changed on disk.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

from . import areas
from .tiles import TILE_HEIGHT_MM

APPCFG_FILE = "appcfg.json"

# How thick the tiles are (`tiles/find.py`), in TILE_HEIGHT_STEP_MM: what the tiles page offers.
TILE_HEIGHT_RANGE_MM = (1.0, 8.0)
TILE_HEIGHT_STEP_MM = 0.5
# The tile offset: how far under a tile's top the cup grips it, its bellows pressed in that far.
# What the tiles page offers: 1 mm over the top to 3 mm in, in TILE_OFFSET_STEP_MM; the cup never goes under the desk
# (`tilework.py`).
TILE_OFFSET_MM = 1.0
TILE_OFFSET_RANGE_MM = (-1.0, 3.0)
TILE_OFFSET_STEP_MM = 0.5
READER = "back"


@dataclass(frozen=True)
class Setting:
    default: object
    accept: Callable[[object], object]       # the value as kept, or ValueError saying why not
    about: str


def _millimeters(what: str, low: float, high: float, step: float) -> Callable[[object], float]:
    def accept(value: object) -> float:
        try:
            mm = round(float(value), 1)
        except (TypeError, ValueError):
            raise ValueError(f"the {what} is a number of mm, not {value!r}") from None
        if not low <= mm <= high:
            raise ValueError(f"the {what} goes from {low:g} to {high:g} mm, not {mm:g}")
        if abs(mm / step - round(mm / step)) > 1e-6:
            raise ValueError(f"the {what} goes in steps of {step:g} mm, not {mm:g}")
        return mm
    return accept


def _reader(value: object) -> str:
    reader = str(value).strip().lower()
    if reader not in areas.READERS:
        raise ValueError(f"the reader stands {', '.join(areas.READERS)} of the arm, not {value!r}")
    return reader


SETTINGS: Dict[str, Setting] = {
    "tile_height_mm": Setting(TILE_HEIGHT_MM, _millimeters("tile height", *TILE_HEIGHT_RANGE_MM, TILE_HEIGHT_STEP_MM),
                              "how thick the tiles are, mm"),
    "tile_offset_mm": Setting(TILE_OFFSET_MM, _millimeters("tile offset", *TILE_OFFSET_RANGE_MM, TILE_OFFSET_STEP_MM),
                              "how far under a tile's top the cup grips it, mm"),
    "reader": Setting(READER, _reader, "where the person reading the word stands, round the arm"),
}


class Settings:
    def __init__(self, config_dir: Path) -> None:
        self.path = Path(config_dir) / APPCFG_FILE
        self.lock = threading.Lock()
        self._stamp: Optional[int] = None
        self._values: Dict[str, object] = {}

    def get(self, name: str) -> object:
        with self.lock:
            return self._current()[name]

    def set(self, **values: object) -> Dict[str, object]:
        """Each value checked and kept, the file written; ValueError, nothing kept, if any is not one its setting
        takes. Every setting's value afterwards."""
        unknown = sorted(set(values) - set(SETTINGS))
        if unknown:
            raise ValueError(f"there is no setting {', '.join(unknown)}")
        accepted = {name: SETTINGS[name].accept(value) for name, value in values.items()}
        with self.lock:
            current = {**self._current(), **accepted}
            changed = {name: value for name, value in current.items() if value != SETTINGS[name].default}
            if changed:
                partial = self.path.with_name(self.path.name + ".partial")
                partial.write_text(json.dumps(changed, indent=2) + "\n")
                partial.replace(self.path)
            elif self.path.exists():
                self.path.unlink()
            self._stamp, self._values = None, {}
            return dict(self._current())

    def to_json(self) -> dict:
        """Every setting: its value, its default and what it is, for a page."""
        with self.lock:
            current = self._current()
        return {name: {"value": current[name], "default": setting.default, "about": setting.about}
                for name, setting in SETTINGS.items()}

    def _current(self) -> Dict[str, object]:
        """The settings, read again only when the file has changed. Hold the lock."""
        stamp = self.path.stat().st_mtime_ns if self.path.exists() else None
        if stamp == self._stamp and self._values:
            return self._values
        stored = json.loads(self.path.read_text()) if stamp is not None else {}
        values = {}
        for name, setting in SETTINGS.items():
            values[name] = setting.default
            if name in stored:
                try:
                    values[name] = setting.accept(stored[name])
                except ValueError as reason:
                    print(f"{self.path.name}: {reason}; {name} is {setting.default!r}, its default", flush=True)
        self._stamp, self._values = stamp, values
        return values
