"""The camera calibration's files in `config/`.

| file                        | holds                                                                   |
| --------------------------- | ----------------------------------------------------------------------- |
| `camera-points.json`        | where the cup was over each cube, as the servos read it, in desk space |
| `camera-frame.png`          | the frame CALIBRATE took, the cubes in view and the arm parked: fitted |
| `camera-check.jpg`          | the latest check image, as the page shows it                           |
| `camera.json`               | `desk_to_pixel`, 3x4, which the library reads, and the fit behind it   |
| `camera-refine-points.json` | the points that refine it: CALIBRATE's three cubes, then the red cube's |
|                             | from the Fine Tune page                                                |
| `camera-view.json`          | QUALITY or FPS: which of the camera's modes the live view is           |

Every file is written beside and renamed into place, so a crash never leaves half of one.
"""

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from maxarm.geometry import Position

from .cubes import COLOURS
from .devices import QUALITY
from .fit import CameraFit, stretch_of
from .model import CUBE_MM
from .sighting import Edges, Sighting

POINTS_FILE = "camera-points.json"
FRAME_FILE = "camera-frame.png"
CHECK_FILE = "camera-check.jpg"
CAMERA_FILE = "camera.json"
REFINE_FILE = "camera-refine-points.json"
REFINE_PREVIOUS_FILE = "camera-refine-points-previous.json"
VIEW_FILE = "camera-view.json"
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
    cube: Optional[str] = None   # the camera page's cube it is, seen in CALIBRATE's frame; None: the Fine Tune page's
    turn_deg: Optional[float] = None     # the cube's turn, true mm, as its edges' corners are numbered
    edges: Optional[List[dict]] = None   # the edges found round the cube: [{"corners": [a, b], "found": [x, y, ...]}]

    @property
    def sighting(self) -> Sighting:
        return Sighting(self.x, self.y, (self.pixel[0], self.pixel[1]), (self.frame_size[0], self.frame_size[1]),
                        self.turn_deg, edges_from_json(self.edges) if self.edges else None)


def edges_to_json(edges: Edges) -> List[dict]:
    """The edges found round a cube, an entry per edge: the corners it joins and the points found on it."""
    starts, ends, found = edges
    pairs = sorted({(int(a), int(b)) for a, b in zip(starts, ends)})
    return [{"corners": list(pair),
             "found": [round(float(value), 2) for point in found[(starts == pair[0]) & (ends == pair[1])]
                       for value in point]} for pair in pairs]


def edges_from_json(stored: List[dict]) -> Edges:
    starts, ends, found = [], [], []
    for edge in stored:
        points = np.array(edge["found"], float).reshape(-1, 2)
        starts += [edge["corners"][0]] * len(points)
        ends += [edge["corners"][1]] * len(points)
        found.append(points)
    return np.array(starts, int), np.array(ends, int), np.concatenate(found) if found else np.zeros((0, 2))


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


def load_refine_points(config_dir: Path, fit: Optional[CameraFit] = None) -> List[RefinePoint]:
    """The points on file. With `fit`, a file from before the cubes were points -- or none -- gets `fit`'s three
    cubes as points 1, 2 and 3, the points it had numbered on after them, and is written so."""
    path = config_dir / REFINE_FILE
    stored = json.loads(path.read_text())["points"] if path.exists() else None
    if fit is None or (stored is not None and all("cube" in point for point in stored)):
        return [RefinePoint(**point) for point in stored or []]
    cubes = cube_refine_points(fit, load_points(config_dir), _when(config_dir / FRAME_FILE))
    points = cubes + [RefinePoint(**{**point, "id": len(cubes) + index + 1})
                      for index, point in enumerate(stored or [])]
    save_refine_points(config_dir, points)
    return points


def cube_refine_points(fit: CameraFit, recorded: Points, taken: str) -> List[RefinePoint]:
    """The three cubes CALIBRATE fitted the camera to, as points 1, 2 and 3: where the arm said the cup was over
    each, the pixel under its centre in CALIBRATE's frame, and the edges found round it there, if `fit` is
    CALIBRATE's own."""
    points = []
    for index, (colour, cube) in enumerate(fit.cube_fit().cubes.items()):
        point = recorded.get(colour)
        z = point.z if point else CUBE_MM                # its point forgotten on the camera page since: its top
        sent = point.sent if point else [cube["x"], cube["y"], z]
        turn_deg, edges = (fit.cube_edges or {}).get(colour, (None, None))
        points.append(RefinePoint(index + 1, cube["x"], cube["y"], z, sent, pixel=list(cube["pixel"]),
                                  frame_size=list(fit.frame_size), taken=taken, cube=colour, turn_deg=turn_deg,
                                  edges=edges_to_json(edges) if edges is not None else None))
    return points


def save_refine_points(config_dir: Path, points: List[RefinePoint]) -> None:
    content = {
        "about": ("The points which refine camera.json: for each, where the servos said the cup was over the "
                  "cube's centre, in desk space as camera-points.json has it, and how the camera saw the cube, the "
                  "arm parked, in a frame of frame_size: the pixel under its centre, and the edges found round it, "
                  "each the corners it joins, numbered as the cube is turned by turn_deg (true mm), and the pixels "
                  "found on it, x, y after x, y. `cube` names the camera page's cube a point is, recorded there and "
                  "seen in CALIBRATE's frame, camera-frame.png (points 1, 2 and 3 as CALIBRATE writes them); null "
                  "for the red cube's points from the Fine Tune page. A point forgotten on the Fine Tune page is "
                  "gone, the cubes' too. They belong to the camera as CALIBRATE fitted it, and are set aside when "
                  "it fits again. Written by calibration/."),
        "points": [asdict(point) for point in points],
    }
    _replace(config_dir / REFINE_FILE, lambda partial: partial.write_text(_json_text(content)))


def set_aside_refine_points(config_dir: Path) -> int:
    """The points, moved aside for a new calibration: how many there were."""
    count = len(load_refine_points(config_dir))
    if count:
        (config_dir / REFINE_FILE).replace(config_dir / REFINE_PREVIOUS_FILE)
    return count


def load_view_choice(config_dir: Path) -> str:
    """QUALITY or FPS, as the camera page last chose; QUALITY if it has not."""
    path = config_dir / VIEW_FILE
    return json.loads(path.read_text())["choice"] if path.exists() else QUALITY


def save_view_choice(config_dir: Path, choice: str) -> None:
    content = {
        "about": ("Which of the camera's modes the live view is: quality, its largest YUYV, uncompressed, or fps, "
                  "its largest MJPG, fast. A calibration holds for the mode it was taken in only. Written by "
                  "calibration/, the camera page."),
        "choice": choice,
    }
    _replace(config_dir / VIEW_FILE, lambda partial: partial.write_text(json.dumps(content, indent=2) + "\n"))


def set_aside_calibration(config_dir: Path) -> List[str]:
    """The camera calibration and everything it was made of -- camera.json, the cubes' points, the refine points, the
    frame, the check image -- renamed `*-previous`: what was set aside."""
    moved = []
    for name in (CAMERA_FILE, POINTS_FILE, REFINE_FILE, FRAME_FILE, CHECK_FILE):
        path = config_dir / name
        if path.exists():
            path.replace(path.with_name(f"{path.stem}-previous{path.suffix}"))
            moved.append(name)
    return moved


def save_camera(config_dir: Path, fit: CameraFit, lens: Optional[dict] = None) -> None:
    """camera.json; `lens`, the camera and the focus the live view holds it at, kept from the file if not given."""
    lens = lens or load_lens(config_dir)
    content = {
        "about": ("Desk space to camera pixels: pixel ~ desk_to_pixel @ [x, y, z, 1], desk mm, z the height "
                  "above the desk. It holds for frames of frame_size only, from a camera that has not moved. "
                  "Fitted by calibration/, the camera page, to the three 40 mm cubes in camera-frame.png, "
                  "standing where camera-points.json says: true_to_pixel is the camera alone, in true mm (the "
                  "cubes' 40 mm its ruler), turned and shifted along the arm's x and y; desk_to_true takes the "
                  "arm's x, y to true mm: the board's mirror and a stretch, the arm's measure against the "
                  "camera's. Together they put the arm's points on the cubes the camera saw, exactly; "
                  "desk_to_pixel is the two together. `refined`, if there, says the camera was fitted again, all "
                  "of it, to the pictures of the points in camera-refine-points.json, and the arm laid on them "
                  "all, least squares; `cubes_alone` keeps the two matrices as the cubes alone gave them. `lens` "
                  "is the camera it was calibrated with, as it calls itself, and the focus it "
                  "was held at (focus_absolute): the live view holds that camera at it, since focusing moves the "
                  "picture."),
        "desk_to_pixel": fit.desk_to_pixel.tolist(),
        **({"lens": lens} if lens else {}),
        **fit.to_json(),
    }
    _replace(config_dir / CAMERA_FILE, lambda partial: partial.write_text(json.dumps(content, indent=2) + "\n"))


def load_lens(config_dir: Path) -> Optional[dict]:
    """{"camera": ..., "focus": ...}: what camera.json was calibrated with; None if it does not say."""
    path = config_dir / CAMERA_FILE
    return json.loads(path.read_text()).get("lens") if path.exists() else None


def load_camera(config_dir: Path) -> Optional[CameraFit]:
    """The saved calibration; None if there is none, or it was not written by the fit."""
    path = config_dir / CAMERA_FILE
    stored = json.loads(path.read_text()) if path.exists() else {}
    return CameraFit.from_json(stored) if "true_to_pixel" in stored else None


class CameraFile:
    """`camera.json`, read again only when the file changes: the calibration, or why there is none."""

    def __init__(self, config_dir: Path) -> None:
        self.path = Path(config_dir) / CAMERA_FILE
        self._loaded: tuple = (None, None)

    def get(self) -> Tuple[Optional[CameraFit], Optional[str]]:
        if not self.path.exists():
            return None, "there is no camera calibration yet: CALIBRATE on the camera page"
        stamp = self.path.stat().st_mtime_ns
        if stamp != self._loaded[0]:
            self._loaded = (stamp, load_camera(self.path.parent))
        fit = self._loaded[1]
        return (fit, None) if fit else (None, "camera.json was not written by the camera page: CALIBRATE again")


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
    """The calibration in a line: how well the cubes fit the frame, how the arm measures against the camera, and how
    far the points it was refined by are off."""
    height = (f", their cubes' tops within {fit.refined['height_mm'][1]:.1f} mm of 40"
              if fit.refined and fit.refined.get("height_mm") else "")
    refined = (f"; refined by {fit.refined['points']} points, the arm's readings {fit.refined['off_mm'][0]:.1f} mm "
               f"from where they all put them typically, {fit.refined['off_mm'][1]:.1f} at worst{height}"
               if fit.refined else "")
    return (f"{fit.frame_size[0]} x {fit.frame_size[1]}, cube edges within {fit.edge_px[0]:.1f} px; "
            f"{describe_stretch(fit.desk_to_true)}{refined}")


def describe_stretch(desk_to_true: np.ndarray) -> str:
    """How the arm measures against the camera, in words: the camera's distances against the arm's readings."""
    low, high, skew = stretch_of(desk_to_true)
    if low >= 1:
        apart = f"{100 * (low - 1):.1f} to {100 * (high - 1):.1f}% longer"
    elif high <= 1:
        apart = f"{100 * (1 - high):.1f} to {100 * (1 - low):.1f}% shorter"
    else:
        apart = f"{100 * (1 - low):.1f}% shorter to {100 * (high - 1):.1f}% longer"
    return f"the camera measures the arm's distances {apart}, its x and y {skew:.1f} deg off square"


def _json_text(content: dict) -> str:
    """Indented JSON with each list of numbers on one line: an edge's found points are hundreds of them."""
    text = json.dumps(content, indent=2)
    return re.sub(r"\[\s+([-0-9.eE,\s]*?)\s+\]", lambda numbers: f"[{' '.join(numbers.group(1).split())}]",
                  text) + "\n"


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
