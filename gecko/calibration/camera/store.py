"""The camera calibration's files in `config/`.

| file                 | holds                                                                        |
| -------------------- | ---------------------------------------------------------------------------- |
| `camera-points.json` | where the cup was over each cube, as the servos read it, in desk space       |
| `camera-frame.png`   | the frame CALIBRATE took, the cubes in view and the arm parked: what is fitted |
| `camera-check.jpg`   | the latest check image, as the page shows it                                 |
| `camera.json`        | `desk_to_pixel`, 3x4, which the library reads: written by the fit, to come   |

Every file is written beside and renamed into place, so a crash never leaves half of one.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from maxarm.geometry import Position

POINTS_FILE = "camera-points.json"
FRAME_FILE = "camera-frame.png"
CHECK_FILE = "camera-check.jpg"
CAMERA_FILE = "camera.json"
COLOURS = ("blue", "green", "red")
CHECK_QUALITY = 90


@dataclass(frozen=True)
class CubePoint:
    x: float                 # desk space: where the servos said the cup was, over the cube's centre
    y: float
    z: float                 # the cup's height over the desk: 44 at the notch, 4 mm over the cube's top
    sent: List[float]        # where it had been sent, which the servos may not have reached

    @property
    def target(self) -> Position:
        return (self.x, self.y, self.z)


Points = Dict[str, Optional[CubePoint]]


def load_points(config_dir: Path) -> Points:
    """Each colour's point, None for a cube not recorded yet."""
    path = config_dir / POINTS_FILE
    stored = json.loads(path.read_text())["cubes"] if path.exists() else {}
    return {colour: CubePoint(**stored[colour]) if stored.get(colour) else None for colour in COLOURS}


def save_points(config_dir: Path, points: Points) -> None:
    content = {
        "about": ("Where the cup was over the centre of each cube's top, as the servos read it: desk space, "
                  "millimetres, z the height above the desk (desk.json, or the default desk). Each cube is "
                  "40 mm: under its x and y the base is at desk height 0 and the top at 40. `sent` is where "
                  "the cup had been sent. Written by calibration/, the camera page."),
        "cubes": {colour: asdict(point) if point else None for colour, point in points.items()},
    }
    _replace(config_dir / POINTS_FILE, lambda partial: partial.write_text(json.dumps(content, indent=2) + "\n"))


def save_frame(config_dir: Path, frame: np.ndarray) -> None:
    _replace(config_dir / FRAME_FILE, lambda partial: _write_image(partial, frame))


def save_check(config_dir: Path, image: np.ndarray) -> dict:
    """Write the check image, and say what it is, as `describe_check()` does."""
    _replace(config_dir / CHECK_FILE,
             lambda partial: _write_image(partial, image, [cv2.IMWRITE_JPEG_QUALITY, CHECK_QUALITY]))
    return _described(config_dir / CHECK_FILE, image.shape[1], image.shape[0])


def describe_check(config_dir: Path) -> Optional[dict]:
    """When the latest check image was taken, and its size; None if there is none."""
    path = config_dir / CHECK_FILE
    image = cv2.imread(str(path)) if path.exists() else None
    return None if image is None else _described(path, image.shape[1], image.shape[0])


def read_check(config_dir: Path) -> Optional[bytes]:
    path = config_dir / CHECK_FILE
    return path.read_bytes() if path.exists() else None


def describe_calibration(config_dir: Path) -> dict:
    """Whether there is a `camera.json`, for the page."""
    path = config_dir / CAMERA_FILE
    if not path.exists():
        return {"exists": False, "summary": "no camera.json yet"}
    return {"exists": True, "summary": f"camera.json, saved {_when(path)}"}


def _described(path: Path, width: int, height: int) -> dict:
    return {"taken": _when(path), "size": [width, height], "stamp": path.stat().st_mtime_ns}


def _when(path: Path) -> str:
    return time.strftime("%d %b %H:%M", time.localtime(path.stat().st_mtime))


def _write_image(path: Path, image: np.ndarray, options: Optional[list] = None) -> None:
    if not cv2.imwrite(str(path), image, options or []):
        raise RuntimeError(f"could not write {path}")


def _replace(path: Path, write: Callable[[Path], None]) -> None:
    """`write` it beside `path`, then rename it into place. The suffix stays: cv2 picks its encoder by it."""
    partial = path.with_name(f"{path.stem}.partial{path.suffix}")
    write(partial)
    partial.replace(path)
