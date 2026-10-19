"""The desk calibration's files in `config/`.

| file                 | holds                                                         |
| -------------------- | ------------------------------------------------------------- |
| `desk.json`          | `desk_to_arm` and `joint_offsets_deg`, which the library      |
|                      | reads, and the touch points behind them, which it does not    |
| `desk-previous.json` | the `desk.json` that the last SAVE replaced                   |
| `desk-points.json`   | the points being recorded, kept on every change for a restart |
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from maxarm.geometry import Position

from .plane import Desk

DESK_FILE = "desk.json"
PREVIOUS_FILE = "desk-previous.json"
POINTS_FILE = "desk-points.json"


@dataclass(frozen=True)
class Point:
    id: int
    x: float                           # arm space: where the servos said the cup was, touching the desk
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
    return points, content.get("next_id", max((point.id for point in points), default=0) + 1)


def save_points(config_dir: Path, points: List[Point], next_id: int) -> None:
    _write(config_dir / POINTS_FILE, {"next_id": next_id, "points": [asdict(point) for point in points]})


def save_desk(config_dir: Path, desk: Desk, points: List[Point]) -> None:
    """Write `desk.json`, keeping the one it replaces as `desk-previous.json`."""
    plane = desk.plane
    _write(config_dir / DESK_FILE, {
        "about": ("desk space to arm space. The board's x, y, z are first made the cup's true ones -- its "
                  "servos sit joint_offsets_deg off the zeros the firmware assumes -- and then true [x, y, z, 1] "
                  "= desk_to_arm @ desk [x, y, z, 1]. Desk space is millimetres, z the height above the desk "
                  "(0: the cup snug on it). A plane: true x and y are desk x and y, and the surface is at true "
                  f"z = {plane.tilt_x:.6f} x + {plane.tilt_y:.6f} y + {plane.offset:.3f}. Fitted by "
                  f"calibration/desk through the {len(points)} touch points below, as the servos read them "
                  "(board x, y, z)."),
        "desk_to_arm": plane.desk_to_arm(),
        "joint_offsets_deg": list(desk.joint_offsets),
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
    elbow = content.get("joint_offsets_deg", [0.0, 0.0, 0.0])[2]
    return {"exists": True, "summary": f"a plane through {len(content['points'])} touch points, elbow "
                                       f"{elbow:+.2f} deg, saved {saved}"}


def _point(entry: dict) -> Point:
    """A point as stored; the first files kept the target as x, y, z and the servos' reading beside it."""
    if "measured" in entry:
        sent = [entry["x"], entry["y"], entry["z"]]
        return Point(entry["id"], *(entry["measured"] or sent), sent=sent)
    return Point(**entry)


def _write(path: Path, content: dict, keep: Optional[Path] = None) -> None:
    """Whole or not at all, as a half-written file would stop the library loading. `keep` gets the old one."""
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(content, indent=2) + "\n")
    if keep is not None and path.exists():
        path.replace(keep)
    partial.replace(path)
