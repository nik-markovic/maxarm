"""The desk calibration's files in `config/`.

| file                 | holds                                                         |
| -------------------- | ------------------------------------------------------------- |
| `desk.json`          | `desk_to_arm`, which the library reads, and the touch points  |
|                      | it was fitted through, which it does not                      |
| `desk-previous.json` | the `desk.json` that the last SAVE replaced                   |
| `desk-points.json`   | the points being recorded, kept on every change for a restart |
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

from maxarm.geometry import Position

from .plane import Plane

DESK_FILE = "desk.json"
PREVIOUS_FILE = "desk-previous.json"
POINTS_FILE = "desk-points.json"


@dataclass(frozen=True)
class Point:
    id: int
    x: float                           # arm space: where the cup was sent when it touched the desk
    y: float
    z: float
    measured: Optional[List[float]]    # where its servos said it was

    @property
    def target(self) -> Position:
        return (self.x, self.y, self.z)


def load_points(config_dir: Path) -> List[Point]:
    path = config_dir / POINTS_FILE
    if not path.exists():
        return []
    return [Point(**entry) for entry in json.loads(path.read_text())["points"]]


def save_points(config_dir: Path, points: List[Point]) -> None:
    _write(config_dir / POINTS_FILE, {"points": [asdict(point) for point in points]})


def save_desk(config_dir: Path, plane: Plane, points: List[Point]) -> None:
    """Write `desk.json`, keeping the one it replaces as `desk-previous.json`."""
    _write(config_dir / DESK_FILE, {
        "about": ("desk space to arm space: arm [x, y, z, 1] = desk_to_arm @ desk [x, y, z, 1]. Desk space "
                  "is millimetres on the desk, z the height above its surface (0: the cup snug on it). "
                  "A plane: arm x and y are desk x and y, and the surface is at arm z = "
                  f"{plane.tilt_x:.6f} x + {plane.tilt_y:.6f} y + {plane.offset:.3f}, fitted by "
                  f"calibration/desk through the {len(points)} touch points below (arm x, y, z)."),
        "desk_to_arm": plane.desk_to_arm(),
        "points": [list(point.target) for point in points],
    }, keep=config_dir / PREVIOUS_FILE)


def describe_desk(config_dir: Path) -> dict:
    """What the saved `desk.json` is, for the page."""
    path = config_dir / DESK_FILE
    if not path.exists():
        return {"exists": False}
    content = json.loads(path.read_text())
    saved = time.strftime("%d %b %H:%M", time.localtime(path.stat().st_mtime))
    if "points" not in content:
        return {"exists": True, "summary": f"an older calibration, not a plane fit ({saved})"}
    return {"exists": True,
            "summary": f"a plane through {len(content['points'])} touch points, saved {saved}"}


def _write(path: Path, content: dict, keep: Optional[Path] = None) -> None:
    """Whole or not at all, as a half-written file would stop the library loading. `keep` gets the old one."""
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(content, indent=2) + "\n")
    if keep is not None and path.exists():
        path.replace(keep)
    partial.replace(path)
