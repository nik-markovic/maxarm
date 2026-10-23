"""The red cube, found by the calibrated camera: where it stands on the desk, and how it is turned.

    found = find_red_cube(frame, fit)       # fit: the calibration, store.load_camera()
    found.x, found.y                        # desk mm: where the arm goes to be over the cube's centre

The camera is known, so a 40 mm cube standing on the desk has one look for each
place and turn. The cube is moved and turned until it looks like the frame's red
blob, the way the calibration's fit moves its camera: first the cube's silhouette
onto the blob's outline, then each edge it shows onto the strongest colour edge
beside it (`edges.py`). Where it stands comes out in true millimetres, the cube's
own; the arm's x/y error, from the calibration, makes it the desk point the arm is
sent to.

The frame may be smaller than the calibration's, as the live view is, as long as it
is the same picture scaled: the camera is scaled with it.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from . import cubes, edges
from .cubes import Blob, CubesNotFound
from .fit import (EDGE_HUBER_PX, OUTLINE_HUBER_PX, OUTLINE_KEEP, OUTLINE_SAMPLES, REACH, TURN_TRIES_DEG, CameraFit,
                  CameraNotFound)
from .model import (CUBE_MM, Edge, around, camera_position, cube_corners, hidden_edges, least_squares, project,
                    rescaled, sample, to_polygon, visible_edges)

# The parameters: where the cube stands, true mm, and its turn, radians; and their finite-difference steps.
STEPS = np.array([0.01, 0.01, 1e-5])
# Covering less of the blob than this, the cube drawn there is not what is red there: something else, or a cube
# not standing on the desk, or partly hidden.
MIN_OVERLAP = 0.6
# Fewer edge points than this, and the cube's edges were not followed.
MIN_EDGE_POINTS = 20
# A frame of another size is the calibration's picture scaled if its shape is the same to this share.
SAME_SHAPE = 0.01


@dataclass(frozen=True)
class FoundCube:
    x: float                          # desk mm, under the cube's centre
    y: float
    turn_deg: float                   # its sides' direction on the desk, 0 to 90 degrees from desk x
    corners: np.ndarray               # 8 x 3, desk mm, as model.cube_corners() orders them
    seen: List[Edge]                  # the edges the camera sees
    hidden: List[Tuple[int, int]]     # and those the cube's own faces hide
    overlap: float                    # how much of the blob the cube drawn there covers, 0 to 1
    edge_px: float                    # how far its edges sit from the frame's: median, pixels of the frame
    pixel: Tuple[float, float]        # where in the frame the desk under its centre is

    def to_json(self) -> dict:
        return {"x": round(self.x, 2), "y": round(self.y, 2), "turn_deg": round(self.turn_deg, 1),
                "pixel": [round(self.pixel[0], 3), round(self.pixel[1], 3)],
                "corners": np.round(self.corners, 2).tolist(),
                "seen": [[edge.start, edge.end, edge.is_outline] for edge in self.seen],
                "hidden": [list(pair) for pair in self.hidden],
                "overlap": round(self.overlap, 3), "edge_px": round(self.edge_px, 2)}


def find_red_cube(frame: np.ndarray, fit: CameraFit) -> FoundCube:
    """The red cube in `frame`. Raises CubesNotFound or CameraNotFound, saying why, if it is not there."""
    matrix = view_camera(fit, frame)
    red = [blob for blob in cubes.find_blobs(frame, is_warm_only=True) if cubes.is_warm(blob)]
    if not red:
        raise CubesNotFound(f"no red cube in view: the frame shows {cubes.describe(cubes.find_blobs(frame))}")
    tries, reasons = [], []
    for blob in red:
        try:
            tries.append(_fit(frame, matrix, blob))
        except CameraNotFound as reason:
            reasons.append(str(reason))
    if not tries:
        raise CameraNotFound(reasons[0])
    overlap, params, distances = max(tries, key=lambda tried: tried[0])
    if overlap < MIN_OVERLAP:
        raise CameraNotFound(f"what is red in view is not a 40 mm cube standing on the desk (a cube drawn there "
                             f"covers {overlap:.0%} of it): is it on something, on its side, or partly hidden?")
    return _found(fit, matrix, params, overlap, distances)


def view_camera(fit: CameraFit, frame: np.ndarray) -> np.ndarray:
    """The calibration's camera, true mm to the pixels of `frame`: its own size, or the same picture scaled."""
    size = (frame.shape[1], frame.shape[0])
    width, height = fit.frame_size
    if size == (width, height):
        return fit.true_to_pixel
    if abs(size[0] * height / (size[1] * width) - 1) > SAME_SHAPE:
        raise ValueError(f"the frame is {size[0]} x {size[1]}, the calibration's {width} x {height}: "
                         f"another shape, so not the same picture")
    return rescaled(fit.true_to_pixel, (width, height), size)


def _fit(frame: np.ndarray, matrix: np.ndarray, blob: Blob) -> Tuple[float, np.ndarray, np.ndarray]:
    """The cube that looks like `blob`: how much of it the cube covers, the parameters, the edges' distances."""
    params = _fit_outline(matrix, blob, _start(matrix, blob))
    params, distances = _fit_edges(frame, matrix, blob, params)
    return edges.overlap(_corners(matrix, params), blob.mask > 0), params, distances


def _start(matrix: np.ndarray, blob: Blob) -> np.ndarray:
    """The cube's middle under the blob's, and the turn that covers the blob best."""
    plane = np.column_stack((matrix[:, 0], matrix[:, 1], matrix[:, 3] + CUBE_MM / 2 * matrix[:, 2]))
    x, y, w = np.linalg.solve(plane, (*blob.centroid, 1.0))
    mask = blob.mask > 0
    turn = max((math.radians(turn) for turn in TURN_TRIES_DEG),
               key=lambda turn: edges.overlap(_corners(matrix, (x / w, y / w, turn)), mask))
    return np.array([x / w, y / w, turn])


def _fit_outline(matrix: np.ndarray, blob: Blob, params: np.ndarray) -> np.ndarray:
    """The cube's silhouette onto the blob's outline, both ways; twice, the second time without what was far off."""
    outline = blob.contour.reshape(-1, 2).astype(float)
    outline = outline[::max(1, len(outline) // 240)]
    box, distance = edges.distance_to_outline(blob)

    def residuals(trial: np.ndarray) -> np.ndarray:
        silhouette = _silhouette(matrix, trial)
        return np.concatenate((to_polygon(outline, silhouette),
                               sample(distance, around(silhouette, OUTLINE_SAMPLES) - box)))

    for _ in range(2):
        params = least_squares(residuals, params, STEPS, huber=OUTLINE_HUBER_PX, iterations=60)[0]
        off = to_polygon(outline, _silhouette(matrix, params))
        outline = outline[off <= max(OUTLINE_HUBER_PX, OUTLINE_KEEP * np.median(off))]
    return params


def _fit_edges(frame: np.ndarray, matrix: np.ndarray, blob: Blob,
               params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Each edge the cube shows onto the frame's edges, round after round as the search narrows."""
    crop = edges.crop(frame, blob, cubes.desk_chroma(frame))
    camera = camera_position(matrix)
    distances = np.zeros(0)
    for reach in REACH:
        seen = visible_edges(params[:2], params[2], camera)
        starts, ends, targets = edges.search(crop, _corners(matrix, params), seen, reach)
        if len(targets) < MIN_EDGE_POINTS:
            raise CameraNotFound("the red cube's edges could not be followed in the frame: is the light even, and "
                                 "the cube clear of other things?")
        params, distances, _ = least_squares(
            lambda trial: edges.signed_distances(_corners(matrix, trial), starts, ends, targets),
            params, STEPS, huber=EDGE_HUBER_PX, iterations=30)
    return params, distances


def _found(fit: CameraFit, matrix: np.ndarray, params: np.ndarray, overlap: float,
           distances: np.ndarray) -> FoundCube:
    to_desk = np.linalg.inv(fit.desk_to_true)
    true = cube_corners(params[:2], params[2])
    corners = np.column_stack((true[:, :2] @ to_desk.T, true[:, 2]))
    x, y = to_desk @ params[:2]
    side = corners[1] - corners[0]
    seen = visible_edges(params[:2], params[2], camera_position(matrix))
    pixel = project(matrix, np.array([[params[0], params[1], 0.0]]))[0]
    return FoundCube(float(x), float(y), math.degrees(math.atan2(side[1], side[0])) % 90.0, corners, seen,
                     hidden_edges(seen), float(overlap), float(np.median(np.abs(distances))),
                     (float(pixel[0]), float(pixel[1])))


def _corners(matrix: np.ndarray, params) -> np.ndarray:
    return project(matrix, cube_corners(params[:2], params[2]))


def _silhouette(matrix: np.ndarray, params: np.ndarray) -> np.ndarray:
    return cv2.convexHull(_corners(matrix, params).astype(np.float32)).reshape(-1, 2).astype(float)
