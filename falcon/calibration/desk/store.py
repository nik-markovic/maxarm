"""The desk calibration's files in `config/`.

| file                 | holds                                                         |
| -------------------- | ------------------------------------------------------------- |
| `desk.json`          | `desk_to_arm`, which the library reads, and the touch points  |
|                      | behind it, which it does not                                  |
| `desk-previous.json` | the `desk.json` that the last SAVE replaced                   |
| `desk-points.json`   | the points being recorded, kept on every change for a restart |

Points are arm space, true z. Files written before the library corrected the
firmware's heights hold raw z (no `"heights": "true"`), and are converted on loading.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from maxarm import raw
from maxarm.geometry import Position

from .plane import Plane

DESK_FILE = "desk.json"
PREVIOUS_FILE = "desk-previous.json"
POINTS_FILE = "desk-points.json"


@dataclass(frozen=True)
class Point:
    id: int
    x: float                           # arm space, true z: where the servos said the cup was, touching the desk
    y: float
    z: float
    sent: Optional[List[float]]        # where it had been sent, which the servos may not have reached

    @property
    def target(self) -> Position:
        return (self.x, self.y, self.z)


def load_points(config_dir: Path) -> Tuple[List[Point], int]:
    """The points, and the number the next one gets: numbers are not used twice until CLEAR."""
    path = config_dir / POINTS_FILE
    if not path.exists():
        return [], 1
    content = json.loads(path.read_text())
    points = [_point(entry) for entry in content["points"]]
    if content.get("heights") != "true":
        points = [_raw_to_true(point) for point in points]
    return points, content.get("next_id", max((point.id for point in points), default=0) + 1)


def save_points(config_dir: Path, points: List[Point], next_id: int) -> None:
    _write(config_dir / POINTS_FILE, {"heights": "true", "next_id": next_id,
                                      "points": [asdict(point) for point in points]})


def save_desk(config_dir: Path, plane: Plane, points: List[Point]) -> None:
    """Write `desk.json`, keeping the one it replaces as `desk-previous.json`."""
    _write(config_dir / DESK_FILE, {
        "about": ("desk space to arm space: arm [x, y, z, 1] = desk_to_arm @ desk [x, y, z, 1]. Arm z is true "
                  "z, the firmware's corrected by the library. Desk space is millimetres, z the height above "
                  "the desk (0: the cup snug on it). A plane: arm x and y are desk x and y, and the surface is "
                  f"at arm z = {plane.tilt_x:.6f} x + {plane.tilt_y:.6f} y + {plane.offset:.3f}. Fitted by "
                  f"calibration/desk through the {len(points)} touch points below, as the servos read them "
                  "(arm x, y, true z)."),
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
    if any(content.get("joint_offsets_deg", ())):
        return {"exists": False, "summary": f"fitted to raw heights ({saved}): not used; SAVE it again"}
    if "points" not in content:
        return {"exists": True, "summary": f"an older calibration, not a plane fit ({saved})"}
    return {"exists": True, "summary": f"a plane through {len(content['points'])} touch points, saved {saved}"}


def _point(entry: dict) -> Point:
    """A point as stored; the first files kept the target as x, y, z and the servos' reading beside it."""
    if "measured" in entry:
        sent = [entry["x"], entry["y"], entry["z"]]
        return Point(entry["id"], *(entry["measured"] or sent), sent=sent)
    return Point(**entry)


def _raw_to_true(point: Point) -> Point:
    x, y, z = (round(value, 2) for value in raw.true_position(point.target))
    sent = [round(value, 2) for value in raw.true_position(tuple(point.sent))] if point.sent else None
    return Point(point.id, x, y, z, sent)


def _write(path: Path, content: dict, keep: Optional[Path] = None) -> None:
    """Whole or not at all, as a half-written file would stop the library loading. `keep` gets the old one."""
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(content, indent=2) + "\n")
    if keep is not None and path.exists():
        path.replace(keep)
    partial.replace(path)
