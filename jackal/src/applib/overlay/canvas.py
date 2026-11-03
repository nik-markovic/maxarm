"""What the overlay draws on, and where desk points are on it.

    view = View(fit.desk_to_pixel, fit.frame_size, (1104, 828), fit.desk_to_true)
    canvas = Canvas((1104, 828), scale=1104 / 900)
    canvas.lines(view.arc(270.0, -120.0, 120.0), RED, width=2, opacity=0.35)
    layer = canvas.layer()            # BGRA, straight alpha: a PNG for a page,
    layer.burn(frame)                 # or into a frame of the same size, for a picture

A `Canvas` is a transparent layer the size of the picture it goes over. Every
mark is drawn antialiased as a coverage mask over its own box and laid over
what is there with its own opacity, so marks show through one another as they
do on a page's canvas. Sizes -- line widths, dots, text -- are design pixels,
the pages' CSS pixels: `scale` is how many of the image's pixels each one is.

A `View` is the camera calibration, desk mm onto the pixels of the picture at
any size: the calibration's own frame, the stream's, or a page's.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from applib.camera.model import rescaled

Ink = Tuple[int, int, int]                  # BGR, as OpenCV draws
Point = Optional[Tuple[float, float]]       # image pixels; None behind the camera

SHIFT = 4                                    # OpenCV's fixed point: a sixteenth of a pixel
# A point drawn further off the picture than this is drawn here instead: one near the horizon projects to millions of
# pixels, past what OpenCV's fixed point holds.
FAR_PX = 100000.0
FONT = cv2.FONT_HERSHEY_DUPLEX
# A Hershey capital is this many pixels tall at font scale 1; a design font size of n pixels is about 0.7 n tall.
FONT_CAP_PX = 22.0
CAP_SHARE = 0.7


def ink(colour: str) -> Ink:
    """'#41c363' as BGR."""
    value = colour.lstrip("#")
    return int(value[4:6], 16), int(value[2:4], 16), int(value[0:2], 16)


class View:
    """Desk mm onto the pixels of an image of the camera's picture, `size` (width, height)."""

    def __init__(self, desk_to_pixel: np.ndarray, frame_size: Tuple[int, int], size: Tuple[int, int],
                 desk_to_true: Optional[np.ndarray] = None) -> None:
        self.size = (int(size[0]), int(size[1]))
        self.matrix = rescaled(np.asarray(desk_to_pixel, float), tuple(frame_size), self.size)
        self.true_to_desk = np.linalg.inv(desk_to_true) if desk_to_true is not None else np.eye(2)

    def at(self, x: float, y: float, z: float = 0.0) -> Point:
        u, v, w = self.matrix @ (x, y, z, 1.0)
        return (u / w, v / w) if w > 0 else None

    def path(self, points: Iterable[Sequence[float]], z: float = 0.0) -> List[Point]:
        """Desk points (x, y) at height `z`, or (x, y, z) each."""
        return [self.at(point[0], point[1], point[2] if len(point) > 2 else z) for point in points]

    def arc(self, radius: float, from_deg: float, to_deg: float, z: float = 0.0) -> List[Point]:
        """Along an arc round the base axis, `from_deg` to `to_deg` from straight out: x < 0, the arm's left, at
        negative bearings."""
        count = max(2, math.ceil(abs(to_deg - from_deg) / 2))
        angles = np.radians(np.linspace(from_deg, to_deg, count + 1))
        return [self.at(radius * math.sin(angle), -radius * math.cos(angle), z) for angle in angles]

    def ring(self, x: float, y: float, radius: float, z: float = 0.0, count: int = 48) -> List[Point]:
        """A circle of `radius` true mm round desk (x, y) at height `z`: the size of a real thing, whatever desk mm
        the calibration holds."""
        turns = np.linspace(0.0, 2 * math.pi, count + 1)
        offsets = self.true_to_desk @ np.vstack((radius * np.cos(turns), radius * np.sin(turns)))
        return [self.at(x + dx, y + dy, z) for dx, dy in offsets.T]


class Layer:
    """A drawn layer: BGRA, straight alpha; burnt into frames of its size, its blend worked out once."""

    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        alpha = image[:, :, 3]
        self.box = None                     # the part drawn on, (rows, columns); None for nothing
        if alpha.any():
            left, top, width, height = cv2.boundingRect(alpha)
            self.box = (slice(top, top + height), slice(left, left + width))
        self._blend = None

    def burn(self, frame: np.ndarray) -> None:
        """Draw the layer into `frame`, BGR, its size, in place."""
        if self.box is None:
            return
        if self._blend is None:
            weight = self.image[self.box][:, :, 3].astype(np.float32) / 255
            self._blend = (np.ascontiguousarray(self.image[self.box][:, :, :3]), weight, 1 - weight)
        colour, weight, rest = self._blend
        frame[self.box] = cv2.blendLinear(frame[self.box], colour, rest, weight)

    def png(self) -> bytes:
        return cv2.imencode(".png", self.image, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()


class Canvas:
    def __init__(self, size: Tuple[int, int], scale: float = 1.0, under: Optional[Layer] = None) -> None:
        """A clear layer `size` (width, height); or, `under` given, a copy of that one to draw over."""
        self.size = (int(size[0]), int(size[1]))
        self.scale = scale
        self.image = under.image.copy() if under is not None else np.zeros((self.size[1], self.size[0], 4), np.uint8)

    def layer(self) -> Layer:
        return Layer(self.image)

    # --- marks --------------------------------------------------------------------------------------------------

    def lines(self, points: Sequence[Point], colour: Ink, width: float, opacity: float,
              dash: Optional[Tuple[float, float]] = None, closed: bool = False) -> None:
        """Points joined up, a gap wherever a point is behind the camera; dashed `dash` = (on, off), design px."""
        runs = _runs(list(points) + list(points[:1]) if closed and points else list(points))
        if dash:
            runs = [piece for run in runs for piece in _dashes(run, dash[0] * self.scale, dash[1] * self.scale)]
        thickness, opacity = self._thickness(width, opacity)
        for run in runs:
            if len(run) > 1:
                self._mark(np.array(run), thickness + 2, opacity, colour,
                           lambda mask, pts: cv2.polylines(mask, [pts], False, 255, thickness, cv2.LINE_AA, SHIFT))

    def fill(self, points: Sequence[Point], colour: Ink, opacity: float) -> None:
        """A shape filled, if every point of it is in front of the camera."""
        if len(points) < 3 or any(point is None for point in points):
            return
        self._mark(np.array(points), 2, opacity, colour,
                   lambda mask, pts: cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA, SHIFT))

    def disc(self, centre: Point, radius: float, colour: Ink, opacity: float) -> None:
        """A filled circle, `radius` design px."""
        if centre is None:
            return
        radius *= self.scale
        self._mark(np.array([centre]), 2 * radius + 2, opacity, colour,
                   lambda mask, pts: cv2.circle(mask, _centre(pts), round(radius * (1 << SHIFT)), 255, -1,
                                                cv2.LINE_AA, SHIFT))

    def circle(self, centre: Point, radius: float, colour: Ink, width: float, opacity: float) -> None:
        """A circle's outline, `radius` and `width` design px."""
        if centre is None:
            return
        radius *= self.scale
        thickness, opacity = self._thickness(width, opacity)
        self._mark(np.array([centre]), 2 * radius + thickness + 2, opacity, colour,
                   lambda mask, pts: cv2.circle(mask, _centre(pts), round(radius * (1 << SHIFT)), 255, thickness,
                                                cv2.LINE_AA, SHIFT))

    def text(self, text: str, at: Point, colour: Ink, size: float, opacity: float = 1.0, bold: bool = False,
             anchor: str = "centre") -> None:
        """`text`, `size` design px, its capitals' middle at `at`: centred on it, or starting there ("left")."""
        if at is None or not text:
            return
        scale, thickness = self._font(size, bold)
        (width, height), _ = cv2.getTextSize(text, FONT, scale, thickness)
        left = at[0] - (width / 2 if anchor == "centre" else 0.0)
        origin = (round(left), round(at[1] + height / 2))
        box = np.array([(left, at[1] - height), (left + width, at[1] + height)])
        self._mark(box, thickness + 2, opacity, colour,
                   lambda mask, offset: cv2.putText(mask, text, (origin[0] - offset[0], origin[1] - offset[1]), FONT,
                                                    scale, 255, thickness, cv2.LINE_AA), is_offset=True)

    def label(self, text: str, at: Point, colour: Ink, size: float = 12.0, anchor: str = "centre",
              pill: Ink = (17, 13, 10)) -> None:
        """`text` on a dark pill outlined in `colour`, kept inside the picture: centred on `at`, or starting there."""
        if at is None:
            return
        width = (self.text_width(text, size) + 14) * self.scale
        height = 22 * self.scale
        left = at[0] - width / 2 if anchor == "centre" else at[0]
        left = min(max(left, 4 * self.scale), self.size[0] - width - 4 * self.scale)
        top = min(max(at[1] - height / 2, 4 * self.scale), self.size[1] - height - 4 * self.scale)
        outline = _rounded(left, top, width, height, 6 * self.scale)
        self.fill(outline, pill, 0.82)
        self.lines(outline, colour, 1.0, 1.0, closed=True)
        self.text(text, (left + 7 * self.scale, top + height / 2), colour, size, anchor="left")

    def text_width(self, text: str, size: float, bold: bool = False) -> float:
        """How wide `text` is drawn, design px."""
        scale, thickness = self._font(size, bold)
        return cv2.getTextSize(text, FONT, scale, thickness)[0][0] / self.scale

    # --- how a mark is laid over the layer --------------------------------------------------------------------------

    def _thickness(self, width: float, opacity: float) -> Tuple[int, float]:
        """A line `width` design px as OpenCV draws it: whole pixels, a line thinner than one fainter instead."""
        pixels = width * self.scale
        return (max(1, round(pixels)), opacity * min(1.0, pixels))

    def _font(self, size: float, bold: bool) -> Tuple[float, int]:
        cap = size * CAP_SHARE * self.scale
        return cap / FONT_CAP_PX, max(1, round(cap / (6.0 if bold else 9.0)))

    def _mark(self, points: np.ndarray, margin: float, opacity: float, colour: Ink, draw, is_offset: bool = False):
        """`draw(mask, points)` on a coverage mask over the points' box, `margin` px round it; then the mask laid
        over the layer in `colour` at `opacity`. With `is_offset`, `draw` is given the box's corner instead."""
        height, width = self.image.shape[:2]
        points = np.clip(points, -FAR_PX, FAR_PX)
        low = np.floor(points.min(axis=0) - margin).astype(int)
        high = np.ceil(points.max(axis=0) + margin).astype(int) + 1
        left, top = max(low[0], 0), max(low[1], 0)
        right, bottom = min(high[0], width), min(high[1], height)
        if right <= left or bottom <= top:
            return
        mask = np.zeros((bottom - top, right - left), np.uint8)
        if is_offset:
            draw(mask, (left, top))
        else:
            draw(mask, np.round((points - (left, top)) * (1 << SHIFT)).astype(np.int32))
        coverage = mask.astype(np.float32) * (opacity / 255)
        if not coverage.any():
            return
        region = self.image[top:bottom, left:right]
        under = region[:, :, 3].astype(np.float32) / 255 * (1 - coverage)
        alpha = coverage + under
        shown = alpha > 0
        mixed = np.zeros(region.shape[:2] + (3,), np.float32)
        mixed[shown] = ((np.array(colour, np.float32) * coverage[shown, None]
                         + region[shown][:, :3].astype(np.float32) * under[shown, None]) / alpha[shown, None])
        region[:, :, :3] = np.clip(np.round(mixed), 0, 255).astype(np.uint8)
        region[:, :, 3] = np.clip(np.round(alpha * 255), 0, 255).astype(np.uint8)


def _centre(points: np.ndarray) -> Tuple[int, int]:
    return int(points[0][0]), int(points[0][1])


def _rounded(left: float, top: float, width: float, height: float, radius: float) -> List[Point]:
    """A rounded rectangle's outline, clockwise from its top left."""
    corners = ((left + width - radius, top + radius, -90), (left + width - radius, top + height - radius, 0),
               (left + radius, top + height - radius, 90), (left + radius, top + radius, 180))
    turns = [(x, y, math.radians(start + step * 15)) for x, y, start in corners for step in range(7)]
    return [(x + radius * math.cos(turn), y + radius * math.sin(turn)) for x, y, turn in turns]


def _runs(points: List[Point]) -> List[List[Tuple[float, float]]]:
    """The points split wherever one is behind the camera."""
    runs: List[List[Tuple[float, float]]] = [[]]
    for point in points:
        if point is None:
            runs.append([])
        else:
            runs[-1].append(point)
    return [run for run in runs if run]


def _dashes(run: List[Tuple[float, float]], on: float, off: float) -> List[List[Tuple[float, float]]]:
    """A polyline cut into dashes `on` long, `off` apart, measured along it."""
    pieces, piece, along, is_on = [], [run[0]], 0.0, True
    for start, end in zip(run, run[1:]):
        length = math.dist(start, end)
        done = 0.0
        while length - done > 1e-9:
            step = min((on if is_on else off) - along, length - done)
            done += step
            along += step
            share = done / length
            point = (start[0] + (end[0] - start[0]) * share, start[1] + (end[1] - start[1]) * share)
            if is_on:
                piece.append(point)
            if along >= (on if is_on else off) - 1e-9:
                if is_on:
                    pieces.append(piece)
                is_on, along = not is_on, 0.0
                piece = [point]
    if is_on and len(piece) > 1:
        pieces.append(piece)
    return pieces
