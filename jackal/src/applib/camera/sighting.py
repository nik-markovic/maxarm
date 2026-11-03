"""A cube as one snapshot showed it, and where a camera puts it: its base on the desk, and how tall it looks.

    numbers = measure(matrix, point, start(matrix, point))   # matrix: a camera for the point's frame
    numbers[:2], numbers[3]                                 # true x, y under the cube's centre; its top, mm
    place(matrix, point, start(matrix, point)[:3])          # x, y and turn of the cube as a 40 mm one

A sighting keeps the edges found round the cube in its snapshot: points of the
frame where its colour changes, each on the line of one of the cube's edges.
They are the frame's, so they hold for any camera. Through a given camera the
cube is moved, turned, and made taller or shorter, its base on the desk, until
its edges lie along them, as `locate.py` fits a cube. Through a right camera a
40 mm cube looks 40 mm tall; one that looks taller or shorter says the camera's
sense of height is off there, or the edges found were not all the cube's: a bad
snapshot. Where it stands is measured with the cube 40 mm tall, as it is
(`place()`): as CALIBRATE and FIND RED CUBE have it.

A sighting from before the edges were kept has only the pixel under the cube's
centre, as the camera of its day saw it: where the cube stands, not how tall.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from . import edges
from .fit import EDGE_HUBER_PX
from .model import CUBE_MM, cube_corners, least_squares, project, rescaled

# The edges found round a cube: for each point found, the corners its edge joins (as model.cube_corners() numbers
# them for the cube's turn), and the point, pixels.
Edges = Tuple[np.ndarray, np.ndarray, np.ndarray]

# A cube's numbers: where it stands, true mm; its turn, radians; its height, mm. Their finite-difference steps.
STEPS = np.array([0.01, 0.01, 1e-5, 0.01])
ITERATIONS = 30


@dataclass(frozen=True)
class Sighting:
    x: float                          # desk mm: where the arm says the cup is, over the cube's centre
    y: float
    pixel: Tuple[float, float]        # where the camera of its day saw the desk under the cube's centre
    frame_size: Tuple[int, int]       # the snapshot's: the calibration's, or the same picture smaller
    turn_deg: Optional[float] = None  # the cube's turn, true mm, as its edges' corners are numbered
    edges: Optional[Edges] = None     # None for a sighting from before the edges were kept


def in_frame(true_to_pixel: np.ndarray, frame_size: Tuple[int, int], point: Sighting) -> np.ndarray:
    """A camera for frames of `frame_size`, made the camera for the point's frame: the same, or the picture scaled."""
    if tuple(point.frame_size) == tuple(frame_size):
        return true_to_pixel
    return rescaled(true_to_pixel, frame_size, point.frame_size)


def on_desk(matrix: np.ndarray, point: Sighting) -> np.ndarray:
    """True x, y of the desk at the point's pixel, through a camera for its frame."""
    x, y, w = np.linalg.solve(matrix[:, [0, 1, 3]], (*point.pixel, 1.0))
    return np.array([x / w, y / w])


def start(matrix: np.ndarray, point: Sighting) -> np.ndarray:
    """A cube's numbers to measure from: 40 mm, turned as found, on the desk at the point's pixel."""
    return np.array([*on_desk(matrix, point), math.radians(point.turn_deg), CUBE_MM])


def measure(matrix: np.ndarray, point: Sighting, numbers: np.ndarray) -> np.ndarray:
    """The cube whose edges lie best along those found, through a camera for the point's frame, from `numbers`:
    x, y, turn and height."""
    return least_squares(lambda trial: misses(matrix, point, trial), numbers, STEPS, huber=EDGE_HUBER_PX,
                         iterations=ITERATIONS)[0]


def place(matrix: np.ndarray, point: Sighting, numbers: np.ndarray) -> np.ndarray:
    """The 40 mm cube whose edges lie best along those found, through a camera for the point's frame, from `numbers`:
    x, y and turn."""
    return least_squares(lambda trial: misses(matrix, point, np.append(trial, CUBE_MM)), numbers, STEPS[:3],
                         huber=EDGE_HUBER_PX, iterations=ITERATIONS)[0]


def misses(matrix: np.ndarray, point: Sighting, numbers: np.ndarray) -> np.ndarray:
    """How far each point found lies across the line of its edge, the cube as `numbers` has it: pixels."""
    starts, ends, found = point.edges
    corners = project(matrix, cube_corners(numbers[:2], numbers[2], numbers[3]))
    return edges.signed_distances(corners, starts, ends, found)
