#!/usr/bin/env python3
"""Coordinate spaces: the arm takes points in whichever one the caller works in.

Every space is defined by how its points map onto **desk space**, which is
where the library plans: millimetres, with z the height above the desk and 0
the cup snug on its surface.

| space    | a point is           | onto desk space by                        | file          |
| -------- | -------------------- | ----------------------------------------- | ------------- |
| `ARM`    | the arm's x, y, z    | `desk_to_arm`, a 4x4, inverted            | `desk.json`   |
| `DESK`   | x, y, height         | nothing to do                             |               |
| `CUSTOM` | your own x, y, z     | `custom_to_desk`, a 4x4                   | `custom.json` |
| `CAMERA` | px, py, height       | `desk_to_pixel`, a 3x4, solved per height | `camera.json` |

Arm z is true z (`raw.py`): the library corrects the firmware's heights on
the way to and from the board, so a level desk is a level plane in arm space,
and under a MaxArm standing on it that plane is arm z = 44. That is the
**default desk**, used whenever there is no `desk.json`: desk space is then arm
space with 44 taken off z. `desk.json` is the desk calibration, for a desk that
is not quite level under the arm or not quite where the default puts it.

A `desk.json` that carries `joint_offsets_deg` was fitted to the firmware's
own heights, before the library corrected them: it is not used, the default
is, and `ignored_desk` says why. The desk page's SAVE rewrites it.

All four files live in `config/` and hold one matrix each.
"""

import json
import math
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np

from .geometry import Position
from .raw import DESK_Z_MM

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# The desk under an ideally placed arm, in true z: true z counts heights above the
# desk from it.
DEFAULT_DESK_Z_MM = DESK_Z_MM


def default_desk_to_arm() -> np.ndarray:
    """The default desk as `desk.json` would hold it: x and y as they are, 44 added to z."""
    matrix = np.eye(4)
    matrix[2, 3] = DEFAULT_DESK_Z_MM
    return matrix


class Space(Enum):
    ARM = "arm"
    DESK = "desk"
    CUSTOM = "custom"
    CAMERA = "camera"


class Affine:
    """A space whose points map onto desk space by one 4x4 matrix."""

    def __init__(self, to_desk: np.ndarray) -> None:
        self.matrix = np.asarray(to_desk, dtype=float)
        self._from_desk = np.linalg.inv(self.matrix)

    def to_desk(self, point: Position) -> Position:
        return _apply(self.matrix, point)

    def from_desk(self, point: Position) -> Position:
        return _apply(self._from_desk, point)


class Camera:
    """Camera space: a pixel, and the height above the desk of what is seen there.

    A pixel names a ray, not a point, so the height picks the point along it:
    the top face of a 4 mm tile is (px, py, 4.0), with the pixel where that top
    face is seen. One 3x4 projection covers every height exactly -- for each
    height it is a plane-to-image homography, inverted to go the other way.
    """

    def __init__(self, desk_to_pixel: np.ndarray) -> None:
        self.matrix = np.asarray(desk_to_pixel, dtype=float)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Camera":
        return cls(_read_matrix(Path(path), "desk_to_pixel", (3, 4)))

    def to_desk(self, point: Position) -> Position:
        px, py, height = point
        x, y, w = np.linalg.solve(self.plane_to_pixel(height), (px, py, 1.0))
        return (float(x / w), float(y / w), float(height))

    def from_desk(self, point: Position) -> Position:
        u, v, w = self.matrix @ (point[0], point[1], point[2], 1.0)
        return (float(u / w), float(v / w), float(point[2]))

    def plane_to_pixel(self, height: float) -> np.ndarray:
        """The 3x3 homography taking (x, y, 1) on the plane `height` up to a pixel."""
        m = self.matrix
        return np.column_stack((m[:, 0], m[:, 1], m[:, 3] + height * m[:, 2]))

    def to_pixels(self, points: np.ndarray) -> np.ndarray:
        """Desk (x, y, z) rows to pixel rows, for drawing over a frame."""
        homogeneous = np.column_stack((points, np.ones(len(points)))) @ self.matrix.T
        return homogeneous[:, :2] / homogeneous[:, 2:]

    def position(self) -> np.ndarray:
        """Where the camera stands, in desk space: the 3x4's null vector."""
        _, _, vectors = np.linalg.svd(self.matrix)
        return vectors[-1][:3] / vectors[-1][3]


Transform = Union[Affine, Camera]


def angle_to_desk(space: Transform, point: Position, angle: float) -> float:
    """A direction at `point`, degrees from the space's +x towards its +y, as desk space has it."""
    return _direction(space.to_desk, point, angle)


def angle_from_desk(space: Transform, desk_point: Position, desk_angle: float) -> float:
    """The way back: a direction in desk space, as `space` has it at that point."""
    return _direction(space.from_desk, desk_point, desk_angle)


class Spaces:
    """The spaces `config_dir` describes. A missing file matters only when its space is asked for;
    a missing `desk.json` not even then: the default desk stands in.

    `config_dir=None` reads nothing and knows no desk at all: arm space only, no
    desk floor. For a calibration app measuring where the desk is.
    """

    def __init__(self, config_dir: Optional[Union[str, Path]] = CONFIG_DIR) -> None:
        self.config_dir = Path(config_dir) if config_dir is not None else None
        desk_file = self.config_dir / "desk.json" if self.config_dir else None
        self.ignored_desk = _why_unusable(desk_file) if desk_file and desk_file.exists() else None
        self.is_desk_known = self.config_dir is not None
        self.is_desk_saved = desk_file is not None and desk_file.exists() and not self.ignored_desk
        if self.is_desk_saved:
            desk_to_arm = _read_matrix(desk_file, "desk_to_arm", (4, 4))
        elif self.is_desk_known:
            desk_to_arm = default_desk_to_arm()
        else:
            desk_to_arm = np.eye(4)      # no desk: desk space is arm space, and nothing checks it
        self.arm = Affine(np.linalg.inv(desk_to_arm))

    def get(self, space: Space) -> Transform:
        if space is Space.ARM:
            return self.arm
        if not self.is_desk_known:
            raise FileNotFoundError(f"{space.name} coordinates are measured from the desk, and no "
                                    f"calibration is in use (config_dir=None)")
        if space is Space.DESK:
            return Affine(np.eye(4))
        if space is Space.CUSTOM:
            return Affine(_read_matrix(self.config_dir / "custom.json", "custom_to_desk", (4, 4)))
        return Camera.load(self.config_dir / "camera.json")

    def convert(self, point: Position, source: Space, target: Space) -> Position:
        return self.get(target).from_desk(self.get(source).to_desk(point))

    def convert_angle(self, point: Position, angle: float, source: Space, target: Space) -> float:
        """A direction at `point` (in `source`), as `target` has it there."""
        source_space, target_space = self.get(source), self.get(target)
        desk_angle = angle_to_desk(source_space, point, angle)
        return angle_from_desk(target_space, source_space.to_desk(point), desk_angle)


def _why_unusable(desk_file: Path) -> Optional[str]:
    if any(json.loads(desk_file.read_text()).get("joint_offsets_deg", ())):
        return (f"{desk_file} was fitted to the firmware's heights, before the library corrected them, "
                f"so the default desk is used; SAVE it again on the desk page")
    return None


def _direction(convert: Callable[[Position], Position], point: Position, angle: float) -> float:
    """Convert both ends of a unit step along the direction.

    Exact in every space here: affine maps, and a camera's plane-to-image
    homographies, both take straight lines to straight lines.
    """
    radians = math.radians(angle)
    here = convert(point)
    there = convert((point[0] + math.cos(radians), point[1] + math.sin(radians), point[2]))
    return math.degrees(math.atan2(there[1] - here[1], there[0] - here[0]))


def _apply(matrix: np.ndarray, point: Position) -> Position:
    x, y, z, w = matrix @ (point[0], point[1], point[2], 1.0)
    return (float(x / w), float(y / w), float(z / w))


def _read_matrix(path: Path, key: str, shape: Tuple[int, int]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; it holds `{key}`, written by a calibration")
    matrix = np.array(json.loads(path.read_text())[key], dtype=float)
    if matrix.shape != shape:
        raise ValueError(f"`{key}` in {path} is {matrix.shape[0]}x{matrix.shape[1]}, "
                         f"not {shape[0]}x{shape[1]}")
    return matrix
