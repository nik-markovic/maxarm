"""RECORD POINT's last point, kept while the page's dialog shows it: the snapshot the cube was seen in, and where the
servos said the cup was before the arm parked for that snapshot. REDO looks for the cube in the same snapshot again,
the picture tweaked, and keeps that same reading; UNDO forgets the point. The dialog shows the snapshot zoomed in on
the cube: what the camera made of it.

    kept = Kept(point.id, here, sent, frame, found, fit)
    tweak = Tweak.random(rng)               # REDO's: contrast, gamma and blur
    found = cube.look(fit, tweak.applied(kept.frame))
    png = zoomed(kept, width, height, scale)

The zoomed picture, drawn through the calibration the cube was found with: the cube as found, yellow and half see-through as every cube's wireframe is drawn, fainter
where hidden, an X under its centre; the points of the frame's edges the cube was fitted to, cyan; on its top, a
yellow dot over its centre and a green one where the servos said the cup was.
"""

import random
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from applib.camera.fit import CameraFit
from applib.camera.locate import FoundCube
from applib.camera.model import BASE_DIAGONALS, CUBE_MM, project, rescaled

# REDO's tweaks to the picture, each drawn evenly from its range: contrast round the picture's mean, gamma, and a
# Gaussian blur, px (0: none).
CONTRAST = (0.8, 1.25)
GAMMA = (0.7, 1.4)
BLURS_PX = (0.0, 0.0, 0.6, 1.0)
# The zoomed picture: the cube's box and this share of its size round it.
ZOOM_MARGIN = 0.15
WIRE_PX = 0.75
WIRE_OPACITY = 0.5
HIDDEN_OPACITY = 0.25
FOUND_DOT_PX = 1.5
FOUND_OPACITY = 0.8
DOT_PX = 3.0
YELLOW = (63, 210, 255)                 # BGR
GREEN = (99, 195, 65)
CYAN = (255, 230, 60)
SHIFT = 4                               # sub-pixel bits OpenCV draws with


@dataclass(frozen=True)
class Tweak:
    contrast: float
    gamma: float
    blur_px: float

    @classmethod
    def random(cls, rng: random.Random) -> "Tweak":
        return cls(round(rng.uniform(*CONTRAST), 2), round(rng.uniform(*GAMMA), 2), rng.choice(BLURS_PX))

    def applied(self, frame: np.ndarray) -> np.ndarray:
        shade = frame.astype(np.float32) / 255.0
        mean = float(shade.mean())
        shade = np.clip((shade - mean) * self.contrast + mean, 0.0, 1.0) ** self.gamma
        if self.blur_px > 0:
            shade = cv2.GaussianBlur(shade, (0, 0), self.blur_px)
        return np.clip(shade * 255.0 + 0.5, 0, 255).astype(np.uint8)

    def describe(self) -> str:
        blur = f", blurred {self.blur_px:.1f} px" if self.blur_px else ""
        return f"contrast x{self.contrast:.2f}, gamma {self.gamma:.2f}{blur}"


@dataclass(frozen=True)
class Kept:
    id: int
    here: Tuple[float, float, float]    # desk space: where the servos said the cup was, before the arm parked
    sent: Sequence[float]               # where the cup had been sent
    frame: np.ndarray                   # the snapshot, the arm parked
    found: FoundCube                    # the cube as last found in it
    fit: CameraFit                      # the calibration it was found through
    look: int = 1                       # how many times it has been looked for: 1, then each REDO
    tweak: Optional[Tweak] = None       # the last look's, None for the snapshot as taken
    detail: str = ""                    # how the last look went, as the page says it
    stamp: int = 0

    def looked_again(self, found: FoundCube, fit: CameraFit, tweak: Tweak, detail: str) -> "Kept":
        return Kept(self.id, self.here, self.sent, self.frame, found, fit, self.look + 1, tweak, detail,
                    time.time_ns())

    def to_json(self) -> dict:
        return {"id": self.id, "look": self.look, "tweak": self.tweak.describe() if self.tweak else None,
                "detail": self.detail, "stamp": self.stamp}


def zoomed(kept: Kept, width: int, height: int, scale: float) -> bytes:
    """The snapshot round the cube, as large as fits `width` x `height` px at pixel ratio `scale`, with the cube as
    found drawn on it through the calibration it was found with -- not camera.json as it is now, which SAVE may have
    changed since: PNG."""
    frame_height, frame_width = kept.frame.shape[:2]
    matrix = rescaled(kept.fit.desk_to_pixel, kept.fit.frame_size, (frame_width, frame_height))
    corners = project(matrix, kept.found.corners)
    low, high = corners.min(axis=0), corners.max(axis=0)
    margin = ZOOM_MARGIN * float(np.max(high - low))
    x0, y0 = (max(0, int(np.floor(value - margin))) for value in low)
    x1, y1 = min(frame_width, int(np.ceil(high[0] + margin))), min(frame_height, int(np.ceil(high[1] + margin)))
    crop = kept.frame[y0:y1, x0:x1]
    zoom = min(width / crop.shape[1], height / crop.shape[0])
    picture = cv2.resize(crop, (max(1, round(crop.shape[1] * zoom)), max(1, round(crop.shape[0] * zoom))),
                         interpolation=cv2.INTER_CUBIC)

    def at(points: np.ndarray) -> np.ndarray:
        """Frame pixels to the picture's, pixel centres kept."""
        return (np.asarray(points, float) - (x0, y0) + 0.5) * zoom - 0.5

    pixels = at(corners)
    width_px = max(1, round(WIRE_PX * scale))
    edges_seen = [(edge.start, edge.end) for edge in kept.found.seen]
    for pairs, opacity in (([*kept.found.hidden, *BASE_DIAGONALS], HIDDEN_OPACITY), (edges_seen, WIRE_OPACITY)):
        picture = _overlaid(picture, opacity, lambda canvas: [
            cv2.line(canvas, _fixed(pixels[a]), _fixed(pixels[b]), YELLOW, width_px, cv2.LINE_AA, SHIFT)
            for a, b in pairs])
    found_points = at(kept.found.edges[2])
    picture = _overlaid(picture, FOUND_OPACITY, lambda canvas: [
        cv2.circle(canvas, _fixed(point), round(FOUND_DOT_PX * scale * (1 << SHIFT)), CYAN, -1, cv2.LINE_AA, SHIFT)
        for point in found_points])
    top = at(project(matrix, np.array([[kept.found.x, kept.found.y, CUBE_MM],
                                                   [kept.here[0], kept.here[1], CUBE_MM]])))
    for point, colour in zip(top, (YELLOW, GREEN)):
        cv2.circle(picture, _fixed(point), round((DOT_PX + 1) * scale * (1 << SHIFT)), (10, 7, 5), -1, cv2.LINE_AA,
                   SHIFT)
        cv2.circle(picture, _fixed(point), round(DOT_PX * scale * (1 << SHIFT)), colour, -1, cv2.LINE_AA, SHIFT)
    return cv2.imencode(".png", picture)[1].tobytes()


def _overlaid(picture: np.ndarray, opacity: float, draw) -> np.ndarray:
    drawn = picture.copy()
    draw(drawn)
    return cv2.addWeighted(drawn, opacity, picture, 1 - opacity, 0)


def _fixed(point: np.ndarray) -> Tuple[int, int]:
    return int(round(point[0] * (1 << SHIFT))), int(round(point[1] * (1 << SHIFT)))
