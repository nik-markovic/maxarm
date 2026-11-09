"""The app's own settings, which the calibration app's pages and the /IOTCONNECT app share: `config/appcfg.json`.

    settings = Settings(CONFIG_DIR)
    settings.get("grip_mm")                 # 2.0 until someone sets it
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

APPCFG_FILE = "appcfg.json"

# Where the cup grips a tile, desk mm: on the 2 mm tile's top, a grip that does not seal tried again deeper
# (`tilework.py`) (the owner's, 9 Nov). It goes down to the desk and up to a visit's height, 4 mm over the tile's top.
GRIP_MM = 2.0
# What the tiles page offers: 1 mm over a tile's top down to the desk, in GRIP_STEP_MM.
GRIP_RANGE_MM = (0.0, 3.0)
GRIP_STEP_MM = 0.5
READER = "back"


@dataclass(frozen=True)
class Setting:
    default: object
    accept: Callable[[object], object]       # the value as kept, or ValueError saying why not
    about: str


def _grip(value: object) -> float:
    try:
        grip = round(float(value), 1)
    except (TypeError, ValueError):
        raise ValueError(f"the grip height is a number of mm, not {value!r}") from None
    if not GRIP_RANGE_MM[0] <= grip <= GRIP_RANGE_MM[1]:
        raise ValueError(f"the grip height goes from {GRIP_RANGE_MM[0]:g} to {GRIP_RANGE_MM[1]:g} mm, not {grip:g}")
    if abs(grip / GRIP_STEP_MM - round(grip / GRIP_STEP_MM)) > 1e-6:
        raise ValueError(f"the grip height goes in steps of {GRIP_STEP_MM:g} mm, not {grip:g}")
    return grip


def _reader(value: object) -> str:
    reader = str(value).strip().lower()
    if reader not in areas.READERS:
        raise ValueError(f"the reader stands {', '.join(areas.READERS)} of the arm, not {value!r}")
    return reader


SETTINGS: Dict[str, Setting] = {
    "grip_mm": Setting(GRIP_MM, _grip, "the desk height the cup grips a tile at, mm"),
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
