"""The camera calibration's files in `config/`.

| file                        | holds                                                                   |
| --------------------------- | ----------------------------------------------------------------------- |
| `camera-points.json`        | where the cup was over each cube, as the servos read it, in desk space |
| `camera-frame.png`          | the frame CALIBRATE took, the cubes in view and the arm parked: fitted |
| `camera-check.jpg`          | the latest check image, as the page shows it                           |
| `camera.json`               | `desk_to_pixel`, 3x4, which the library reads, and the fit behind it   |
| `camera-refine-points.json` | the points that refine it: CALIBRATE's three cubes, then the red cube's |
|                             | from the Fine Tune page                                                |
| `arm-true.json`             | the arm's raw-to-true correction the points make (`maxarm/mesh.py`),   |
|                             | which the library reads on the fly (`maxarm/converter.py`)             |
| `camera-view.json`          | QUALITY or FPS: which of the camera's modes the live view is           |

Every file is written beside and renamed into place, so a crash never leaves half of one.
"""

import dataclasses
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from maxarm import Space, Spaces, converter, raw
from maxarm.geometry import Position
from maxarm.mesh import Mesh

from .cubes import COLORS
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
CORRECTION_FILE = converter.CORRECTION_FILE.name
VIEW_FILE = "camera-view.json"
# Per point recorded on Fine Tune: its snapshot, lossless, and the cube as found, drawn -- for an agent or a person to
# review a calibration with. Only the points on file have them: a point forgotten takes its pictures along, and a new
# calibration all of them.
DEBUG_DIR = "debug"
CHECK_QUALITY = 90


@dataclass(frozen=True)
class CubePoint:
    x: float                 # desk space: where the servos said the cup was, over the cube's center
    y: float
    z: float                 # the cup's height over the desk: at the notch, just over the cube's top
    sent: List[float]        # where it had been sent, which the servos may not have reached
    raw: Optional[List[float]] = None    # the firmware's position, as the servos read it: what calibrations are made of

    @property
    def target(self) -> Position:
        return (self.x, self.y, self.z)


Points = Dict[str, Optional[CubePoint]]


@dataclass(frozen=True)
class RefinePoint:
    id: int
    x: float                 # desk space: where the servos said the cup was, over the red cube's center
    y: float
    z: float
    sent: List[float]        # where it had been sent
    pixel: List[float]       # where the camera then saw the desk under the cube's center, the arm parked
    frame_size: List[int]    # the frame it was seen in: the snapshot's, or an older live view's
    taken: str
    cube: Optional[str] = None   # the camera page's cube it is, seen in CALIBRATE's frame; None: the Fine Tune page's
    turn_deg: Optional[float] = None     # the cube's turn, true mm, as its edges' corners are numbered
    edges: Optional[List[dict]] = None   # the edges found round the cube: [{"corners": [a, b], "found": [x, y, ...]}]
    raw: Optional[List[float]] = None    # the firmware's position, as the servos read it: what calibrations are made of

    @property
    def sighting(self) -> Sighting:
        """The point as a sighting, its x, y where the arm says the cup was as the library converts its raw reading
        now: through the latest calibration."""
        x, y = (self.x, self.y) if self.raw is None else raw.true_position(tuple(self.raw))[:2]
        return Sighting(x, y, (self.pixel[0], self.pixel[1]), (self.frame_size[0], self.frame_size[1]),
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
    """Each color's point, None for a cube not recorded yet; its raw reading as `with_raw()` has it."""
    path = config_dir / POINTS_FILE
    stored = json.loads(path.read_text())["cubes"] if path.exists() else {}
    return {color: with_raw(config_dir, CubePoint(**stored[color])) if stored.get(color) else None
            for color in COLORS}


def with_raw(config_dir: Path, point):
    """A point with its raw reading: its own, or for one kept before the raw reading was, the formula's raw position
    for its x, y, z -- the formula alone recorded it then."""
    if point.raw is not None:
        return point
    arm = Spaces(config_dir).convert((point.x, point.y, point.z), Space.DESK, Space.ARM)
    return dataclasses.replace(point, raw=[round(value, 3) for value in raw.formula_raw(arm)])


def raw_reading(position: Position) -> List[float]:
    """The firmware's position for a true arm-space one read just now: what a point keeps."""
    return [round(value, 3) for value in raw.raw_position(position)]


def save_points(config_dir: Path, points: Points) -> None:
    content = {
        "about": ("Where the cup was over the center of each cube's top, as the servos read it: desk space, "
                  "millimeters, z the height above the desk (desk.json, or the default desk). Each cube is "
                  "40 mm: under its x and y the base is at desk height 0 and the top at 40. `sent` is where "
                  "the cup had been sent; `raw` the firmware's position, what the calibration is made of. Written by "
                  "calibration/, the camera page."),
        "cubes": {color: asdict(point) if point else None for color, point in points.items()},
    }
    _replace(config_dir / POINTS_FILE, lambda partial: partial.write_text(json.dumps(content, indent=2) + "\n"))


def load_refine_points(config_dir: Path, fit: Optional[CameraFit] = None) -> List[RefinePoint]:
    """The points on file. With `fit`, a file from before the cubes were points -- or none -- gets `fit`'s three
    cubes as points 1, 2 and 3, the points it had numbered on after them, and is written so."""
    path = config_dir / REFINE_FILE
    stored = json.loads(path.read_text())["points"] if path.exists() else None
    if fit is None or (stored is not None and all("cube" in point for point in stored)):
        return [with_raw(config_dir, RefinePoint(**point)) for point in stored or []]
    cubes = [with_raw(config_dir, point) for point in
             cube_refine_points(fit, load_points(config_dir), _when(config_dir / FRAME_FILE))]
    points = cubes + [with_raw(config_dir, RefinePoint(**{**point, "id": len(cubes) + index + 1}))
                      for index, point in enumerate(stored or [])]
    save_refine_points(config_dir, points)
    return points


def cube_refine_points(fit: CameraFit, recorded: Points, taken: str) -> List[RefinePoint]:
    """The three cubes CALIBRATE fitted the camera to, as points 1, 2 and 3: where the arm said the cup was over
    each, the pixel under its center in CALIBRATE's frame, and the edges found round it there, if `fit` is
    CALIBRATE's own."""
    points = []
    for index, (color, cube) in enumerate(fit.cube_fit().cubes.items()):
        point = recorded.get(color)
        z = point.z if point else CUBE_MM                # its point forgotten on the camera page since: its top
        sent = point.sent if point else [cube["x"], cube["y"], z]
        turn_deg, edges = (fit.cube_edges or {}).get(color, (None, None))
        points.append(RefinePoint(index + 1, cube["x"], cube["y"], z, sent, pixel=list(cube["pixel"]),
                                  frame_size=list(fit.frame_size), taken=taken, cube=color, turn_deg=turn_deg,
                                  edges=edges_to_json(edges) if edges is not None else None,
                                  raw=point.raw if point else None))
    return points


def save_refine_points(config_dir: Path, points: List[RefinePoint]) -> None:
    content = {
        "about": ("The points which refine camera.json: for each, where the servos said the cup was over the "
                  "cube's center, in desk space as camera-points.json has it, and how the camera saw the cube, the "
                  "arm parked, in a frame of frame_size: the pixel under its center, and the edges found round it, "
                  "each the corners it joins, numbered as the cube is turned by turn_deg (true mm), and the pixels "
                  "found on it, x, y after x, y. `cube` names the camera page's cube a point is, recorded there and "
                  "seen in CALIBRATE's frame, camera-frame.png (points 1, 2 and 3 as CALIBRATE writes them); null "
                  "for the red cube's points from the Fine Tune page. `raw` is the firmware's position over the cube, "
                  "what the calibration is made of; x, y, z are it as the library converted it then. "
                  "A point forgotten on the Fine Tune page is "
                  "gone, the cubes' too. They belong to the camera as CALIBRATE fitted it, and are set aside when "
                  "it fits again. Written by calibration/."),
        "points": [asdict(point) for point in points],
    }
    _replace(config_dir / REFINE_FILE, lambda partial: partial.write_text(_json_text(content)))
    _keep_pictures_of(config_dir, points)


def set_aside_refine_points(config_dir: Path) -> int:
    """The points moved aside for a new calibration, their pictures deleted: how many there were."""
    count = len(load_refine_points(config_dir))
    if count:
        (config_dir / REFINE_FILE).replace(config_dir / REFINE_PREVIOUS_FILE)
    shutil.rmtree(config_dir / DEBUG_DIR, ignore_errors=True)
    return count


def save_point_pictures(config_dir: Path, point_id: int, taken: str, snapshot: Optional[np.ndarray],
                        found: bytes) -> None:
    """A recorded point's snapshot (None: unchanged, as after REDO) and the cube as found in it (PNG), in DEBUG_DIR,
    named by the point and when it was taken."""
    folder = config_dir / DEBUG_DIR
    folder.mkdir(exist_ok=True)
    name = _picture_name(point_id, taken)
    if snapshot is not None:
        _replace(folder / f"{name}.png", lambda partial: _write_image(partial, snapshot))
    _replace(folder / f"{name}-found.png", lambda partial: partial.write_bytes(found))


def _picture_name(point_id: int, taken: str) -> str:
    return f"point-{point_id}-{re.sub(r'[^0-9A-Za-z]+', '', taken)}"


def _keep_pictures_of(config_dir: Path, points: List[RefinePoint]) -> None:
    """DEBUG_DIR's pictures of points no longer on file deleted."""
    folder = config_dir / DEBUG_DIR
    if not folder.exists():
        return
    names = {_picture_name(point.id, point.taken) for point in points}
    for path in folder.iterdir():
        if path.name.split(".")[0].removesuffix("-found") not in names:
            path.unlink(missing_ok=True)


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
    for name in (CAMERA_FILE, POINTS_FILE, REFINE_FILE, FRAME_FILE, CHECK_FILE, CORRECTION_FILE):
        path = config_dir / name
        if path.exists():
            path.replace(path.with_name(f"{path.stem}-previous{path.suffix}"))
            moved.append(name)
    shutil.rmtree(config_dir / DEBUG_DIR, ignore_errors=True)
    _reload(config_dir / CORRECTION_FILE)
    return moved


def save_camera(config_dir: Path, fit: CameraFit, lens: Optional[dict] = None) -> None:
    """camera.json; `lens`, the camera and the focus the live view holds it at, kept from the file if not given."""
    lens = lens or load_lens(config_dir)
    content = {
        "about": ("Desk space to camera pixels: pixel ~ desk_to_pixel @ [x, y, z, 1], desk mm, z the height "
                  "above the desk. It holds for frames of frame_size only, from a camera that has not moved. "
                  "Written by the camera page's CALIBRATE and Fine Tune's SAVE; config/README.md says what each "
                  "key is."),
        "desk_to_pixel": fit.desk_to_pixel.tolist(),
        **({"lens": lens} if lens else {}),
        **fit.to_json(),
    }
    _replace(config_dir / CAMERA_FILE, lambda partial: partial.write_text(json.dumps(content, indent=2) + "\n"))


def save_correction(config_dir: Path, mesh: Optional[Mesh]) -> None:
    """arm-true.json: the arm's raw-to-true correction, which the library reads on the fly; read again at once here.
    None: no correction, the formula alone -- the file set aside, `-previous`."""
    path = config_dir / CORRECTION_FILE
    if mesh is None:
        if path.exists():
            path.replace(path.with_name(f"{path.stem}-previous{path.suffix}"))
        _reload(path)
        return
    content = {
        "about": ("The arm's raw-to-true correction, on top of the formula (maxarm/raw.py), as the camera "
                  "calibration measured it: a row per point, where the formula put the cup over the cube (its "
                  "bearing, degrees, 0 straight out, + toward +x, and its reach from the base axis, mm) and how much "
                  "further out (mm) and further round (degrees) the camera saw the cube, the arm's axis and turn "
                  "laid on the camera by its base and the points' bearings (camera.json's `anchor`). The rows make "
                  "the pattern they share over the whole desk and a mesh of what it leaves at each, fading out "
                  "beyond the points (maxarm/mesh.py), made again from the rows when read. Read "
                  "by maxarm/converter.py whenever it changes. Written by calibration/, CALIBRATE and SAVE."),
        **mesh.to_json(),
    }
    _replace(path, lambda partial: partial.write_text(json.dumps(content, indent=2) + "\n"))
    _reload(path)


def _reload(path: Path) -> None:
    """The library's converter reads the correction again now, if it is the one it reads."""
    if path.resolve() == converter.CORRECTION_FILE.resolve():
        converter.reload()


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
    misses = (f", each {fit.refined['off_mm'][0]:.1f} mm from where the others put it typically, "
              f"{fit.refined['off_mm'][1]:.1f} at worst" if fit.refined and fit.refined.get("off_mm") else "")
    refined = (f"; refined by {fit.refined['points']} points{misses}{height}"
               if fit.refined else
               f"; the arm's readings {', '.join(f'{fit.cube_off_mm(color):.1f}' for color in fit.cubes)} mm "
               f"from its cubes" if fit.anchor is None else "")
    return (f"{fit.frame_size[0]} x {fit.frame_size[1]}, cube edges within {fit.edge_px[0]:.1f} px; "
            f"{describe_anchor(fit.anchor) if fit.anchor else describe_stretch(fit.measured_stretch)}{refined}")


def describe_anchor(anchor: dict) -> str:
    """How the arm was laid on the camera, in words: the base, the turn, what fits worst."""
    report, base = anchor.get("report", {}), anchor.get("base")
    laid = (f"the arm's axis {report['base']['axis_off_mm']:.1f} mm from the base found by its edges, turned "
            f"{report['base']['turn_off_deg']:+.2f} deg from its sides by the points' bearings" if base and "base" in report
            else "the arm's axis and turn from the points' readings (no base found)")
    worst = report.get("worst")
    return (laid + (f"; fits worst: {worst['what']}, {worst['spreads']:.1f} spreads" if worst else "")
            + "".join(f"; point {point} does not fit, its reach {miss:.1f} mm off the others': left out"
                      for point, miss in report.get("does_not_fit_mm", {}).items()))


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
