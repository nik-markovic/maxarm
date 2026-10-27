"""The camera the calibration fits, the cubes it is fitted to, and the least-squares solver.

The camera is a pinhole with square pixels: a focal length and a principal point
in pixels, and where it stands and which way it looks, in true millimetres with z
up from the desk. Lens distortion is not modelled: the library's camera is one
3x4 matrix.

A cube is 40 mm, stands on the desk, and is turned about its vertical axis by
however it was set down. The camera above sees its top and one or two sides;
which, and so which of its edges show, follows from where the camera stands.

Also the pixel geometry the fit measures with: projecting through a 3x4 matrix,
sampling an image between pixels, a point's distance to a polygon, points evenly
round one.
"""

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CUBE_MM = 40.0

# A cube's corners: the four on the desk, then the four on top, each round the
# same way. Each face lists its corners and its outward direction.
_SQUARE = ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5))
_FACES = (
    ((4, 5, 6, 7), (0.0, 0.0, 1.0)),
    ((0, 1, 5, 4), (0.0, -1.0, 0.0)),
    ((1, 2, 6, 5), (1.0, 0.0, 0.0)),
    ((2, 3, 7, 6), (0.0, 1.0, 0.0)),
    ((3, 0, 4, 7), (-1.0, 0.0, 0.0)),
)                                        # the bottom never shows: the camera is above the desk
# All 12 edges: round the bottom, round the top, and the four uprights.
_ALL_EDGES = tuple((k, (k + 1) % 4) for k in range(4)) + tuple((4 + k, 4 + (k + 1) % 4) for k in range(4)) \
    + tuple((k, k + 4) for k in range(4))
# The bottom's diagonals: they cross under the cube's centre, where the arm records it.
BASE_DIAGONALS = ((0, 2), (1, 3))


@dataclass(frozen=True)
class Edge:
    start: int              # corner indices, as `cube_corners()` orders them
    end: int
    is_outline: bool        # on the silhouette: the cube on one side, whatever is behind it on the other


def cube_corners(centre: Sequence[float], turn: float) -> np.ndarray:
    """The 8 corners, true mm: the cube standing on the desk at `centre`, turned `turn` radians."""
    c, s = math.cos(turn), math.sin(turn)
    corners = [(centre[0] + CUBE_MM * (c * dx - s * dy), centre[1] + CUBE_MM * (s * dx + c * dy), z)
               for z in (0.0, CUBE_MM) for dx, dy in _SQUARE]
    return np.array(corners)


def visible_edges(centre: Sequence[float], turn: float, camera_xyz: np.ndarray) -> List[Edge]:
    """The edges of the faces the camera sees, each marked as on the outline or between two faces."""
    corners = cube_corners(centre, turn)
    c, s = math.cos(turn), math.sin(turn)
    seen = {}
    for indices, (nx, ny, nz) in _FACES:
        outward = np.array((c * nx - s * ny, s * nx + c * ny, nz))
        if (camera_xyz - corners[list(indices)].mean(axis=0)) @ outward > 0:
            for k in range(4):
                key = tuple(sorted((indices[k], indices[(k + 1) % 4])))
                seen[key] = seen.get(key, 0) + 1
    return [Edge(start, end, count == 1) for (start, end), count in seen.items()]


def hidden_edges(seen: Sequence[Edge]) -> List[Tuple[int, int]]:
    """The edges the cube's own faces hide from the camera: those of no face it sees."""
    shown = {tuple(sorted((edge.start, edge.end))) for edge in seen}
    return [edge for edge in _ALL_EDGES if tuple(sorted(edge)) not in shown]


def camera_position(matrix: np.ndarray) -> np.ndarray:
    """Where a 3x4 camera stands, in the space it maps from: its null vector."""
    _, _, vectors = np.linalg.svd(matrix)
    return vectors[-1][:3] / vectors[-1][3]


def project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(x, y, z) rows through a 3x4 camera, to pixel rows."""
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def rescaled(matrix: np.ndarray, size: Tuple[int, int], new_size: Tuple[int, int]) -> np.ndarray:
    """A 3x4 camera for the same picture at another size, as the camera scales it: pixel centres kept."""
    sx, sy = new_size[0] / size[0], new_size[1] / size[1]
    scale = np.array([[sx, 0.0, 0.5 * sx - 0.5], [0.0, sy, 0.5 * sy - 0.5], [0.0, 0.0, 1.0]])
    return scale @ matrix


@dataclass(frozen=True)
class Pinhole:
    focal: float            # pixels
    centre: Tuple[float, float]   # the principal point, pixels
    rotation: np.ndarray    # 3-vector, Rodrigues: true mm to the camera's axes
    shift: np.ndarray       # 3-vector, mm: the desk's origin in the camera's axes

    @property
    def matrix(self) -> np.ndarray:
        """3x4: true (x, y, z, 1) to homogeneous pixels."""
        intrinsic = np.array([[self.focal, 0.0, self.centre[0]], [0.0, self.focal, self.centre[1]],
                              [0.0, 0.0, 1.0]])
        return intrinsic @ np.column_stack((self._rotation_matrix(), self.shift))

    @property
    def position(self) -> np.ndarray:
        """Where the camera stands, true mm."""
        return -self._rotation_matrix().T @ self.shift

    def project(self, points: np.ndarray) -> np.ndarray:
        """True (x, y, z) rows to pixel rows."""
        seen = np.asarray(points, float) @ self._rotation_matrix().T + self.shift
        return np.column_stack((self.focal * seen[:, 0] / seen[:, 2] + self.centre[0],
                                self.focal * seen[:, 1] / seen[:, 2] + self.centre[1]))

    def _rotation_matrix(self) -> np.ndarray:
        return cv2.Rodrigues(np.asarray(self.rotation, float))[0]


Residuals = Callable[[np.ndarray], np.ndarray]


def least_squares(residuals: Residuals, start: np.ndarray, steps: np.ndarray, *, huber: float,
                  free: Optional[Sequence[int]] = None,
                  iterations: int = 40) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Levenberg-Marquardt, with residuals beyond `huber` counted linearly so outliers pull less.

    `steps` are each parameter's finite-difference step, in its own units. Returns
    the solution, its residuals, and the normal matrix for the free parameters.
    """
    free = list(range(len(start))) if free is None else list(free)
    solution, damping = np.asarray(start, float).copy(), 1e-3
    current = residuals(solution)
    normal = np.zeros((len(free), len(free)))
    for _ in range(iterations):
        weights = np.where(np.abs(current) <= huber, 1.0, huber / np.maximum(np.abs(current), 1e-12))
        jacobian = np.empty((len(current), len(free)))
        for column, index in enumerate(free):
            nudged = solution.copy()
            nudged[index] += steps[index]
            jacobian[:, column] = (residuals(nudged) - current) / steps[index]
        normal = jacobian.T @ (jacobian * weights[:, None])
        gradient = jacobian.T @ (weights * current)
        before = _huber_cost(current, huber)
        for _ in range(12):
            step = np.linalg.solve(normal + damping * np.diag(np.diag(normal) + 1e-12), -gradient)
            trial = solution.copy()
            trial[free] += step
            trial_residuals = residuals(trial)
            if _huber_cost(trial_residuals, huber) < before:
                solution, current, damping = trial, trial_residuals, max(damping / 3.0, 1e-9)
                break
            damping *= 4.0
        else:
            break                                        # no step lowers the cost: converged
        if np.all(np.abs(step) < 0.01 * steps[free]):
            break
    return solution, current, normal


def to_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Each point's distance to the nearest side of `polygon`."""
    nearest = np.full(len(points), np.inf)
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        side = end - start
        along = np.clip((points - start) @ side / max(side @ side, 1e-12), 0.0, 1.0)
        nearest = np.minimum(nearest, np.linalg.norm(points - (start + along[:, None] * side), axis=1))
    return nearest


def around(polygon: np.ndarray, count: int) -> np.ndarray:
    """`count` points evenly round a closed polygon, however many corners it has."""
    ends = np.roll(polygon, -1, axis=0)
    lengths = np.linalg.norm(ends - polygon, axis=1)
    reach = np.concatenate(([0.0], np.cumsum(lengths)))
    at = np.linspace(0.0, reach[-1], count, endpoint=False)
    side = np.clip(np.searchsorted(reach, at, side="right") - 1, 0, len(polygon) - 1)
    share = (at - reach[side]) / np.maximum(lengths[side], 1e-12)
    return polygon[side] + share[:, None] * (ends[side] - polygon[side])


def sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Bilinear samples of `image` (2-D, or 3-D for channels) at (x, y) rows, clamped to its edges."""
    height, width = image.shape[:2]
    x = np.clip(points[:, 0], 0, width - 1.001)
    y = np.clip(points[:, 1], 0, height - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    if image.ndim == 3:
        fx, fy = fx[:, None], fy[:, None]
    return (image[y0, x0] * (1 - fx) * (1 - fy) + image[y0, x0 + 1] * fx * (1 - fy)
            + image[y0 + 1, x0] * (1 - fx) * fy + image[y0 + 1, x0 + 1] * fx * fy)


def _huber_cost(residuals: np.ndarray, huber: float) -> float:
    size = np.abs(residuals)
    return float(np.sum(np.where(size <= huber, 0.5 * size ** 2, huber * (size - 0.5 * huber))))
