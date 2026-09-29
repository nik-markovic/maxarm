"""A sheet of square grid paper flat on the desk: where its lattice lies in the frame.

Grid paper is the one thing on the desk whose shape is known everywhere: two
families of lines, evenly spaced and at right angles. Its lines are faint --
a few grey levels on the doc cam -- so they are not traced one by one. The
whole lattice is aligned at once instead, by enhanced correlation (OpenCV's
`findTransformECC`): a rendered grid in lattice units is warped onto the
frame's "how much darker than its surroundings" image until the two agree, so
every pixel of every line counts towards one homography.

It starts from the pitch and direction of the two line families, read off a
Fourier transform of the top-down view that the existing calibration gives.
That calibration only has to be good to a fraction of a square.

The result maps lattice units (one square = 1) to frame pixels. It knows
nothing about millimetres or the arm; the square's size is the paper's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

import topdown
from mapping import DeskMapping

VIEW_PX_PER_MM = 6.0
# The lines' own shadow: blur that is wider than a line and narrower than a square.
DARKNESS_BLUR_PX = 8.0
# Stay clear of the sheet's edges, its torn binding and the holes: the centre only.
EDGE_MARGIN_MM = 15.0
TEMPLATE_PX_PER_SQUARE = 24
LINE_SIGMA_SQUARES = 0.06


@dataclass(frozen=True)
class Lattice:
    square_to_pixel: np.ndarray         # 3x3: (u, v, 1) in squares -> frame pixel
    correlation: float                  # ECC's, 1 is perfect
    squares: Tuple[int, int]            # how many squares the fit covered, u by v

    def to_pixel(self, squares: np.ndarray) -> np.ndarray:
        mapped = np.column_stack((squares, np.ones(len(squares)))) @ self.square_to_pixel.T
        return mapped[:, :2] / mapped[:, 2:]

    def to_squares(self, pixels: np.ndarray) -> np.ndarray:
        mapped = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(self.square_to_pixel).T
        return mapped[:, :2] / mapped[:, 2:]


def darkness(frame: np.ndarray) -> np.ndarray:
    """How much darker each pixel is than its surroundings, in the red channel.

    The lines are pale blue: red is where they show most against white paper.
    """
    red = frame[..., 2].astype(np.float32)
    return cv2.GaussianBlur(red, (0, 0), DARKNESS_BLUR_PX) - red


def paper_mask(image: np.ndarray, px_per_mm: float) -> np.ndarray:
    """The sheet: the largest bright region, shrunk away from its edges."""
    red = cv2.GaussianBlur(image[..., 2].astype(np.float32), (0, 0), 5)
    valid = image.sum(axis=2) > 0
    cut, _ = cv2.threshold(red[valid].astype(np.uint8).reshape(-1, 1), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = ((red > cut) & valid).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    sheet = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    size = 2 * round(EDGE_MARGIN_MM * px_per_mm) + 1
    return cv2.erode(sheet, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))


def frame_mask(frame: np.ndarray, mapping: DeskMapping) -> np.ndarray:
    """The sheet's centre in frame pixels, its margin measured in millimetres on the desk."""
    view = topdown.render(frame, mapping, 0.0, px_per_mm=VIEW_PX_PER_MM)
    mask = paper_mask(view.image, VIEW_PX_PER_MM)
    view_to_frame = mapping.plane_to_pixel(0.0) @ view.view_to_ground
    return cv2.warpPerspective(mask, view_to_frame, frame.shape[1::-1], flags=cv2.INTER_NEAREST)


def first_guess(frame: np.ndarray, mapping: DeskMapping) -> Tuple[np.ndarray, np.ndarray]:
    """The lattice from a Fourier transform of the top-down view: squares -> view mm, and the mask."""
    view = topdown.render(frame, mapping, 0.0, px_per_mm=VIEW_PX_PER_MM)
    mask = paper_mask(view.image, VIEW_PX_PER_MM)
    dark = darkness(view.image) * mask
    rows, cols = np.nonzero(mask)
    top, left = rows.min(), cols.min()
    patch = dark[top:rows.max(), left:cols.max()]
    window = np.hanning(patch.shape[0])[:, None] * np.hanning(patch.shape[1])[None, :]
    size = 2048
    spectrum = np.fft.fft2(patch * window, s=(size, size))
    magnitude = np.abs(spectrum)
    magnitude[0:8, :] = magnitude[:, 0:8] = magnitude[-8:, :] = magnitude[:, -8:] = 0
    # The two strongest peaks not opposite each other are the two line families.
    frequencies, phases = [], []
    for _ in range(2):
        row, col = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        fy = (row if row < size // 2 else row - size) / size
        fx = (col if col < size // 2 else col - size) / size
        frequencies.append((fx, fy))                  # cycles per view pixel
        phases.append(np.angle(spectrum[row, col]))
        for r, c in ((row, col), ((-row) % size, (-col) % size)):
            magnitude[max(0, r - 12):r + 13, max(0, c - 12):c + 13] = 0
    # Lines are where the darkness peaks: u = f . (x - origin) + phase / 2 pi is an integer.
    linear = np.array(frequencies)
    offset = -linear @ np.array((left, top)) + np.array(phases) / (2 * np.pi)
    view_px_to_squares = np.vstack((np.column_stack((linear, offset)), (0, 0, 1)))
    squares_to_frame = mapping.plane_to_pixel(0.0) @ view.view_to_ground @ np.linalg.inv(view_px_to_squares)
    return squares_to_frame, mask


def fit(frame: np.ndarray, mapping: DeskMapping) -> Lattice:
    guess, _ = first_guess(frame, mapping)
    mask = frame_mask(frame, mapping)
    dark = darkness(frame) * mask
    # Squares whose middle is on the sheet, as a u, v rectangle.
    rows, cols = np.nonzero(mask[::8, ::8])
    corners = np.array(np.meshgrid([cols.min() * 8, cols.max() * 8], [rows.min() * 8, rows.max() * 8])).reshape(2, -1).T
    squares = Lattice(guess, 0.0, (0, 0)).to_squares(corners.astype(float))
    low, high = np.floor(squares.min(axis=0)).astype(int), np.ceil(squares.max(axis=0)).astype(int)
    span = high - low
    step = TEMPLATE_PX_PER_SQUARE
    u, v = np.meshgrid(np.arange(span[0] * step) / step, np.arange(span[1] * step) / step)
    distance_u = np.abs(u - np.round(u))
    distance_v = np.abs(v - np.round(v))
    sigma = LINE_SIGMA_SQUARES
    template = (np.exp(-distance_u ** 2 / (2 * sigma ** 2)) + np.exp(-distance_v ** 2 / (2 * sigma ** 2)))
    template = template.astype(np.float32)
    # template pixel -> squares -> frame
    template_to_squares = np.array([[1 / step, 0, low[0]], [0, 1 / step, low[1]], [0, 0, 1]])
    warp = (guess @ template_to_squares).astype(np.float32)
    warp /= warp[2, 2]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7)
    correlation, warp = cv2.findTransformECC(template, dark.astype(np.float32), warp,
                                             cv2.MOTION_HOMOGRAPHY, criteria, mask, 5)
    square_to_pixel = warp.astype(np.float64) @ np.linalg.inv(template_to_squares)
    return Lattice(square_to_pixel / square_to_pixel[2, 2], float(correlation), tuple(int(s) for s in span))
