"""The camera calibration's files in `config/`.

| file                        | holds                                                                  |
| --------------------------- | ---------------------------------------------------------------------- |
| `camera-points.json`        | where the cup was over each cube, as the servos read it, in desk space |
| `camera-frame.png`          | the frame CALIBRATE took, the cubes in view and the arm parked: fitted |
| `camera-check.jpg`          | the latest check image, as the page shows it                           |
| `camera.json`               | `desk_to_pixel`, 3x4, which the library reads, and the fit behind it   |
| `camera-refine-points.json` | the red cube's points, from the Fine Tune page, that refine it         |

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

from .cubes import COLOURS
from .fit import CameraFit
from .refine import Sighting

POINTS_FILE = "camera-points.json"
FRAME_FILE = "camera-frame.png"
CHECK_FILE = "camera-check.jpg"
CAMERA_FILE = "camera.json"
REFINE_FILE = "camera-refine-points.json"
REFINE_PREVIOUS_FILE = "camera-refine-points-previous.json"
CHECK_QUALITY = 90


@dataclass(frozen=True)
class CubePoint:
    x: float                 # desk space: where the servos said the cup was, over the cube's centre
    y: float
    z: float                 # the cup's height over the desk: 42 at the notch, 2 mm over the cube's top
    sent: List[float]        # where it had been sent, which the servos may not have reached

    @property
    def target(self) -> Position:
        return (self.x, self.y, self.z)


Points = Dict[str, Optional[CubePoint]]


@dataclass(frozen=True)
class RefinePoint:
    id: int
    x: float                 # desk space: where the servos said the cup was, over the red cube's centre
    y: float
    z: float
    sent: List[float]        # where it had been sent
    pixel: List[float]       # where the camera then saw the desk under the cube's centre, the arm parked
    frame_size: List[int]    # the frame it was seen in: the snapshot's, or an older live view's
    taken: str

    @property
    def sighting(self) -> Sighting:
        return Sighting(self.x, self.y, (self.pixel[0], self.pixel[1]), (self.frame_size[0], self.frame_size[1]))


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


def load_refine_points(config_dir: Path) -> List[RefinePoint]:
    path = config_dir / REFINE_FILE
    return [RefinePoint(**point) for point in json.loads(path.read_text())["points"]] if path.exists() else []


def save_refine_points(config_dir: Path, points: List[RefinePoint]) -> None:
    content = {
        "about": ("The red cube's points, which refine camera.json: for each, where the servos said the cup was "
                  "when the cube stood centred under it, in desk space as camera-points.json has it, and where the "
                  "camera then saw the desk under the cube's centre, the arm parked: a pixel of a frame of "
                  "frame_size. They belong to the camera as CALIBRATE fitted it, and are set aside when it "
                  "fits again. Written by calibration/, the Fine Tune page."),
        "points": [asdict(point) for point in points],
    }
    _replace(config_dir / REFINE_FILE, lambda partial: partial.write_text(json.dumps(content, indent=2) + "\n"))


def set_aside_refine_points(config_dir: Path) -> int:
    """The red cube's points, moved aside for a new calibration: how many there were."""
    count = len(load_refine_points(config_dir))
    if count:
        (config_dir / REFINE_FILE).replace(config_dir / REFINE_PREVIOUS_FILE)
    return count


def save_camera(config_dir: Path, fit: CameraFit) -> None:
    content = {
        "about": ("Desk space to camera pixels: pixel ~ desk_to_pixel @ [x, y, z, 1], desk mm, z the height "
                  "above the desk. It holds for frames of frame_size only, from a camera that has not moved. "
                  "Fitted by calibration/, the camera page, to the three 40 mm cubes in camera-frame.png, "
                  "standing where camera-points.json says: true_to_pixel is the camera alone, in true mm; "
                  "desk_to_true is how the arm's x and y lie in true mm -- a turn, the board's mirror and the "
                  "arm's x/y error; desk_to_pixel is the two together. `refined`, if there, says the arm's error "
                  "was fitted again to the cubes and the red cube's points in camera-refine-points.json, and "
                  "`cubes_alone` keeps the two matrices as the cubes alone gave them."),
        "desk_to_pixel": fit.desk_to_pixel.tolist(),
        **fit.to_json(),
    }
    _replace(config_dir / CAMERA_FILE, lambda partial: partial.write_text(json.dumps(content, indent=2) + "\n"))


def load_camera(config_dir: Path) -> Optional[CameraFit]:
    """The saved calibration; None if there is none, or it was not written by the fit."""
    path = config_dir / CAMERA_FILE
    stored = json.loads(path.read_text()) if path.exists() else {}
    return CameraFit.from_json(stored) if "true_to_pixel" in stored else None


def save_frame(config_dir: Path, frame: np.ndarray) -> None:
    _replace(config_dir / FRAME_FILE, lambda partial: _write_image(partial, frame))


def load_frame(config_dir: Path) -> Optional[np.ndarray]:
    """The frame CALIBRATE fitted the camera to; None if there is none."""
    path = config_dir / FRAME_FILE
    return cv2.imread(str(path)) if path.exists() else None


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
    """Whether there is a `camera.json`, and what it says, for the page."""
    path = config_dir / CAMERA_FILE
    if not path.exists():
        return {"exists": False, "summary": "no camera.json yet"}
    fit = load_camera(config_dir)
    saved = f"camera.json, saved {_when(path)}"
    return {"exists": True, "saved": saved, "summary": f"{saved}: {describe_fit(fit)}" if fit else saved}


def describe_fit(fit: CameraFit) -> str:
    """The calibration in a line: how well the cubes fit, and the arm's x/y error it takes in."""
    refined = (f"; refined by {fit.refined['points']} red cube points, all within {fit.refined['off_mm'][1]:.1f} mm"
               if fit.refined else "")
    return (f"{fit.frame_size[0]} x {fit.frame_size[1]}, cube edges within {fit.edge_px[0]:.1f} px; "
            f"{describe_arm(fit)}{refined}")


def describe_arm(fit: CameraFit) -> str:
    """The arm's x/y error, as the calibration takes it in."""
    x_scale, y_scale, skew = fit.arm_scale()
    return f"the arm's x goes {_percent(x_scale)}, y {_percent(y_scale)}, axes {abs(skew):.1f} deg off square"


def _percent(scale: float) -> str:
    return f"{abs(scale - 1) * 100:.1f}% {'short' if scale < 1 else 'long'}"


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
