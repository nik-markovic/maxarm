"""Synthetic tile faces for training the reader, laid out like the real crops.

Each sample is what `tiles/crop.py` hands the reader: a 22 mm window around a
tile's top face, square-on, the tile's baseline horizontal. The layout is the
owner's description of the tiles, checked against 13 upright crops from the
first scene:

- the face is 17.5 mm along the baseline and 20 mm up it, drawn up to 1.5 mm
  wider and 1 mm shorter, because the view measures the axis the camera looks
  along about that much long and that is not settled;
- the capital is 9-10 mm tall, centred 8.5 mm from the face's left edge -- a
  quarter millimetre left of the middle -- and a little above the middle;
- the score is about a quarter of the capital's height, just right of the
  letter, centred near its baseline.

Everything else is deliberately random -- font and weight, wood and desk
colour and texture, which way the desk is lighter, blur, noise, pose error of
a millimetre and a few degrees -- so the reader learns the letter and its
layout, and nothing about one camera or one desk.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SCORES = dict(A=1, B=3, C=3, D=2, E=1, F=4, G=2, H=4, I=1, J=8, K=5, L=1, M=3,
              N=1, O=1, P=3, Q=10, R=1, S=1, T=1, U=1, V=4, W=4, X=8, Y=4, Z=10)
TURNED = len(LETTERS)            # the class for "not upright"

WINDOW_MM = 22.0
SIZE = 64                        # what the reader sees
SUPERSAMPLE = 4
FONT_DIR = Path(__file__).resolve().parent / "fonts"     # make-reader.sh fetches them


@dataclass(frozen=True)
class Face:
    path: Path
    weight: int                  # 0 for a static font


def list_faces() -> List[Face]:
    faces = []
    for path in sorted(FONT_DIR.rglob("*.ttf")):
        if "italic" in path.name.lower():
            continue
        if "[" in path.name and "wght" in path.name:
            faces += [Face(path, weight) for weight in (300, 400, 500, 600)]
        elif any(style in path.stem for style in ("Regular", "Medium", "SemiBold", "Bold")) \
                or "-" not in path.stem:
            faces.append(Face(path, 0))
    return faces


NEGATIVES = ("off-centre", "blank", "desk")


def render(face: Face, letter: str, turn: int, rng: random.Random, negative: str = "") -> np.ndarray:
    """One 64x64 grey sample: `letter` on a tile, then turned `turn` quarter turns.

    `negative` draws something the reader must call "not an upright letter":
    the window centred on a tile's edge instead of its face, a tile with no
    letter, or bare desk with shadow lines across it -- which is what a letter
    finder turns up on a light desk besides letters.
    """
    scale = SIZE * SUPERSAMPLE / WINDOW_MM             # pixels per mm, supersampled
    size = SIZE * SUPERSAMPLE
    desk, face_tone, ink = _tones(rng)
    canvas = Image.new("L", (size, size), desk)
    draw = ImageDraw.Draw(canvas)
    _texture(draw, size, desk, rng, strength=rng.uniform(0, 40))

    face_w, face_h = rng.uniform(17.5, 19.0), rng.uniform(19.0, 20.5)
    left = (WINDOW_MM - face_w) / 2 + rng.uniform(-1.2, 1.2)
    top = (WINDOW_MM - face_h) / 2 + rng.uniform(-1.2, 1.2)
    if negative == "off-centre":
        away = rng.uniform(0, 2 * np.pi)
        distance = rng.uniform(6.0, 14.0)
        left, top = left + distance * np.cos(away), top + distance * np.sin(away)
    box = [left * scale, top * scale, (left + face_w) * scale, (top + face_h) * scale]
    if negative == "desk":
        for _ in range(rng.randint(1, 3)):
            _shadow_line(draw, size, desk, rng)
        return _finish(np.asarray(canvas, dtype=np.float32), scale, turn, rng)
    if rng.random() < 0.3:                                  # its shadow on the desk
        _shadow_line(draw, size, desk, rng, box)
    if rng.random() < 0.7:                                  # the side band the camera sees
        band = rng.uniform(0.5, 4.0) * scale
        side = rng.choice(((0, band), (0, -band), (band, 0), (-band, 0)))
        shade = int(np.clip(face_tone + rng.uniform(-40, 25), 0, 255))
        draw.rectangle([box[0] + min(0, side[0]), box[1] + min(0, side[1]),
                        box[2] + max(0, side[0]), box[3] + max(0, side[1])], fill=shade)
    draw.rectangle(box, fill=face_tone)
    _grain(draw, box, face_tone, ink, scale, rng)
    if rng.random() < 0.25:                                 # a neighbour touching this tile
        offset = rng.choice((-1, 1)) * (face_w + rng.uniform(0.3, 1.5)) * scale
        draw.rectangle([box[0] + offset, box[1], box[2] + offset, box[3]], fill=face_tone)

    if negative == "blank":
        return _finish(np.asarray(canvas, dtype=np.float32), scale, turn, rng)
    cap = rng.uniform(9.0, 10.0) * scale
    font = _font(face, cap)
    glyph_box = font.getbbox(letter)
    glyph_h = glyph_box[3] - glyph_box[1]
    if glyph_h <= 0:
        return None
    font = _font(face, cap * cap / glyph_h)                 # make the capital exactly `cap` tall
    glyph_box = font.getbbox(letter)
    glyph_w = glyph_box[2] - glyph_box[0]
    centre_x = (left + 8.5 + rng.uniform(-0.8, 0.8)) * scale
    centre_y = (top + face_h / 2 - rng.uniform(0.0, 1.0)) * scale
    x = centre_x - glyph_w / 2 - glyph_box[0]
    y = centre_y - cap / 2 - glyph_box[1]
    draw.text((x, y), letter, font=font, fill=ink)

    digits = str(SCORES[letter])
    small = _font(face, cap * rng.uniform(0.22, 0.3) / 0.72)
    digit_box = small.getbbox(digits)
    sx = centre_x + glyph_w / 2 + rng.uniform(0.2, 1.0) * scale - digit_box[0]
    sy = centre_y + cap / 2 - (digit_box[3] - digit_box[1]) * rng.uniform(0.3, 0.8) - digit_box[1]
    draw.text((sx, sy), digits, font=small, fill=ink)
    if rng.random() < 0.2:                                  # a crack across the wood
        x0 = rng.uniform(box[0], box[2])
        draw.line([(x0, box[1]), (x0 + rng.uniform(-30, 30), box[3])],
                  fill=int(face_tone * rng.uniform(0.5, 0.85)), width=rng.randint(2, 6))

    return _finish(np.asarray(canvas, dtype=np.float32), scale, turn, rng)


def _finish(image: np.ndarray, scale: float, turn: int, rng: random.Random) -> np.ndarray:
    """Pose error, the camera's softness and noise, the turn, and the reader's normalisation."""
    image = _pose_error(image, scale, rng)
    image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.3, 1.4))
    image = image * rng.uniform(0.7, 1.2) + rng.uniform(-25, 25)
    image += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(0, 6), image.shape)
    image = np.rot90(np.clip(image, 0, 255), turn)
    return normalise(image.astype(np.uint8))


def normalise(grey: np.ndarray) -> np.ndarray:
    """Stretch so the face is near white and the ink near black, whatever the exposure.

    The same function runs on real crops, which is the point of it.
    """
    grey = grey.astype(np.float32)
    centre = grey[SIZE // 4:3 * SIZE // 4, SIZE // 4:3 * SIZE // 4]
    low, high = np.percentile(centre, 2), np.percentile(centre, 90)
    return np.clip((grey - low) / max(high - low, 1.0) * 255, 0, 255).astype(np.uint8)


def _tones(rng: random.Random) -> Tuple[int, int, int]:
    face_tone = rng.randint(150, 250)
    if rng.random() < 0.5:
        desk = rng.randint(10, max(11, face_tone - 40))
    else:
        desk = rng.randint(min(face_tone + 5, 250), 255) if rng.random() < 0.5 \
            else rng.randint(max(0, face_tone - 30), min(255, face_tone + 20))
    ink = rng.randint(0, 90)
    return desk, face_tone, ink


def _texture(draw: ImageDraw.ImageDraw, size: int, tone: int, rng: random.Random, strength: float) -> None:
    for _ in range(rng.randint(0, 60)):
        x = rng.uniform(0, size)
        shade = int(np.clip(tone + rng.uniform(-strength, strength), 0, 255))
        draw.line([(x, 0), (x + rng.uniform(-size / 3, size / 3), size)], fill=shade,
                  width=rng.randint(1, 8))


def _shadow_line(draw: ImageDraw.ImageDraw, size: int, desk: int, rng: random.Random,
                 box=None) -> None:
    """A thin dark line: a tile's shadow along one of its edges, or a stray one."""
    shade = int(desk * rng.uniform(0.3, 0.8))
    width = rng.randint(2, 10)
    if box is None:
        x0, y0 = rng.uniform(0, size), rng.uniform(0, size)
        angle = rng.uniform(0, np.pi)
        length = rng.uniform(0.3, 1.0) * size
        draw.line([(x0, y0), (x0 + length * np.cos(angle), y0 + length * np.sin(angle))],
                  fill=shade, width=width)
        return
    edge = rng.randrange(4)
    pad = width / 2 + 1
    left, top, right, bottom = box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad
    ends = (((left, bottom), (right, bottom)), ((left, top), (right, top)),
            ((left, top), (left, bottom)), ((right, top), (right, bottom)))[edge]
    draw.line(ends, fill=shade, width=width)


def _grain(draw: ImageDraw.ImageDraw, box, tone: int, ink: int, scale: float, rng: random.Random) -> None:
    """Wood grain along one of the face's axes: fine faint lines, and sometimes dark bands.

    The bands are what real tiles show. One D measured 1-1.5 mm wide and 40% of
    the way from face to ink, running the whole width of the face along its
    baseline; with only faint lines in training, the reader took that band for
    a tile's edge and the D for "not a letter" on every turn.
    """
    along_baseline = rng.random() < 0.5

    def stripe(shade: int, width: float) -> None:
        wobble = rng.uniform(-15, 15)
        if along_baseline:
            y = rng.uniform(box[1], box[3])
            draw.line([(box[0], y), (box[2], y + wobble)], fill=shade, width=max(1, round(width)))
        else:
            x = rng.uniform(box[0], box[2])
            draw.line([(x, box[1]), (x + wobble, box[3])], fill=shade, width=max(1, round(width)))

    for _ in range(rng.randint(0, 12)):
        stripe(int(np.clip(tone - rng.uniform(0, 25), 0, 255)), rng.randint(1, 4))
    if rng.random() < 0.5:
        for _ in range(rng.randint(1, 3)):
            stripe(int(tone - rng.uniform(0.1, 0.55) * (tone - ink)), rng.uniform(0.3, 1.6) * scale)


def _pose_error(image: np.ndarray, scale: float, rng: random.Random) -> np.ndarray:
    size = image.shape[0]
    centre = (size / 2, size / 2)
    matrix = cv2.getRotationMatrix2D(centre, rng.uniform(-5, 5), 1.0)
    matrix[0, :2] *= rng.uniform(0.94, 1.06)
    matrix[1, :2] *= rng.uniform(0.94, 1.06)
    matrix[:, 2] += (rng.uniform(-1.0, 1.0) * scale, rng.uniform(-1.0, 1.0) * scale)
    matrix[:, 2] += np.array(centre) - matrix[:, :2] @ np.array(centre)
    return cv2.warpAffine(image, matrix, (size, size), borderMode=cv2.BORDER_REFLECT)


_FONTS = {}


def _font(face: Face, pixel_size: float) -> ImageFont.FreeTypeFont:
    key = (face, int(pixel_size))
    if key not in _FONTS:
        font = ImageFont.truetype(str(face.path), max(4, int(pixel_size)))
        if face.weight:
            axes = font.get_variation_axes()
            values = [axis["default"] for axis in axes]
            for index, axis in enumerate(axes):
                if axis["name"] in (b"Weight", "Weight"):
                    values[index] = min(max(face.weight, axis["minimum"]), axis["maximum"])
            font.set_variation_by_axes(values)
        _FONTS[key] = font
    return _FONTS[key]
