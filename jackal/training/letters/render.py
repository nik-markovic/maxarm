"""Synthetic tile faces for training the letter reader: capitals printed by hand in black marker.

Each sample is what `applib/letters/face.py` hands the reader: a window of
`WINDOW_MM` round a tile's top face, square-on, the tile's sides along the
window's, turned by a quarter turn or not. The tiles (PILOT-hyena):

- 25 x 25 mm, a light face with a slightly darker rounded rim, darker side faces;
- one capital, printed by hand with a black marker, roughly in the middle;
- a dot at the letter's bottom right on some, which says which way is up: the
  owner dots M and W, whose shapes are each other's turned.

The letters are strokes, not a font: each capital is a few polylines in a unit
box (`SKELETONS`), every point moved a little, the whole sheared, squeezed and
drawn with a round pen of random width -- so no two are the same, as no two
hand-written letters are. Everything else is random too: the face's tone and its
lighting, the ink's, what lies beyond the face, blur, noise, and a pose error of
a millimetre and a few degrees, so the reader learns the letter and nothing
about one camera, one desk or one hand.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NOT_UPRIGHT = len(LETTERS)          # the class for "not a letter the right way up"

WINDOW_MM = 27.0
SIZE = 64                           # what the reader sees
SUPERSAMPLE = 4
FACE_MM = 25.0

# What a capital turned by np.rot90's quarter turns (anticlockwise) reads as, where that is another letter or itself.
# Undotted only: a dot that has turned away from the bottom right makes any of these "not upright".
# An N turned a quarter is a Z only in outline: a printed N is taller than wide, so turned it is a Z far wider than
# anyone prints one, and a U turned is a C with straight arms. Those are taught as "not upright", so that the reader
# learns the proportions; labelled as the other letter, it read the owner's N as a Z (STATUS-hyena §8.3).
TURNED_READS = {
    1: {"O": "O", "X": "X"},
    2: {"H": "H", "I": "I", "N": "N", "O": "O", "S": "S", "X": "X", "Z": "Z", "M": "W", "W": "M"},
    3: {"O": "O", "X": "X"},
}

Stroke = List[Tuple[float, float]]


def label(letter: str, turn: int, is_dotted: bool) -> int:
    """The class a sample is: the letter it reads as at this quarter turn, or NOT_UPRIGHT."""
    if turn == 0:
        return LETTERS.index(letter)
    reads = None if is_dotted else TURNED_READS[turn].get(letter)
    return LETTERS.index(reads) if reads else NOT_UPRIGHT


# --- the letters -------------------------------------------------------------------------------------------------
# Unit box, x right, y down, (0, 0) the letter's top left. Each returns its strokes, choosing among the ways people
# print that capital.

def _arc(cx: float, cy: float, rx: float, ry: float, start: float, end: float, steps: int = 18) -> Stroke:
    """Degrees, 0 to the right, 90 down (y down), from `start` to `end`."""
    return [(cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a)))
            for a in np.linspace(start, end, steps)]


def _a(r: random.Random) -> List[Stroke]:
    bar = r.uniform(0.55, 0.72)
    top = r.uniform(0.42, 0.58)
    left, right = (top * (1 - bar), top + (1 - top) * bar)
    return [[(0, 1), (top, 0), (1, 1)], [(left, bar), (right, bar)]]


def _b(r: random.Random) -> List[Stroke]:
    middle = r.uniform(0.42, 0.55)
    upper = r.uniform(0.7, 0.9)
    return [[(0, 0), (0, 1)],
            [(0, 0), (upper * 0.6, 0)] + _arc(upper * 0.6, middle / 2, upper * 0.4, middle / 2, -90, 90)
            + [(0, middle)],
            [(0, middle), (0.6, middle)] + _arc(0.6, (1 + middle) / 2, 0.4, (1 - middle) / 2, -90, 90) + [(0, 1)]]


def _c(r: random.Random) -> List[Stroke]:
    gap = r.uniform(35, 60)
    return [_arc(0.5, 0.5, 0.5, 0.5, gap, 360 - gap, 24)]


def _d(r: random.Random) -> List[Stroke]:
    flat = r.uniform(0.15, 0.45)
    return [[(0, 0), (0, 1)], [(0, 0), (flat, 0)] + _arc(flat, 0.5, 1 - flat, 0.5, -90, 90, 20) + [(0, 1)]]


def _e(r: random.Random) -> List[Stroke]:
    middle, reach = r.uniform(0.42, 0.58), r.uniform(0.6, 0.95)
    if r.random() < 0.5:
        return [[(1, 0), (0, 0), (0, 1), (1, 1)], [(0, middle), (reach, middle)]]
    return [[(1, 0), (0, 0)], [(0, 0), (0, 1)], [(0, 1), (1, 1)], [(0, middle), (reach, middle)]]


def _f(r: random.Random) -> List[Stroke]:
    middle, reach = r.uniform(0.4, 0.58), r.uniform(0.55, 0.9)
    return [[(1, 0), (0, 0), (0, 1)], [(0, middle), (reach, middle)]]


def _g(r: random.Random) -> List[Stroke]:
    gap = r.uniform(30, 55)
    end = r.uniform(5, 35)
    arc = _arc(0.5, 0.5, 0.5, 0.5, 360 - gap, end, 24)
    shelf = r.uniform(0.5, 0.6)
    return [arc + [(arc[-1][0], shelf), (r.uniform(0.45, 0.7), shelf)]]


def _h(r: random.Random) -> List[Stroke]:
    middle = r.uniform(0.42, 0.58)
    return [[(0, 0), (0, 1)], [(1, 0), (1, 1)], [(0, middle), (1, middle)]]


def _i(r: random.Random) -> List[Stroke]:
    if r.random() < 0.5:
        return [[(0.5, 0), (0.5, 1)]]
    half = r.uniform(0.25, 0.5)
    return [[(0.5, 0), (0.5, 1)], [(0.5 - half, 0), (0.5 + half, 0)], [(0.5 - half, 1), (0.5 + half, 1)]]


def _j(r: random.Random) -> List[Stroke]:
    hook = _arc(0.5, 0.7, 0.5, 0.3, 0, 180, 14)
    strokes = [[(1, 0), (1, 0.7)] + hook[1:]]
    if r.random() < 0.4:
        strokes.append([(r.uniform(0.3, 0.6), 0), (1.2, 0)])
    return strokes


def _k(r: random.Random) -> List[Stroke]:
    join = r.uniform(0.45, 0.6)
    if r.random() < 0.5:
        return [[(0, 0), (0, 1)], [(1, 0), (0, join), (1, 1)]]
    meet = r.uniform(0.25, 0.45)
    return [[(0, 0), (0, 1)], [(1, 0), (0, join + 0.05)], [(meet, join - 0.1 + meet * 0.2), (1, 1)]]


def _l(r: random.Random) -> List[Stroke]:
    return [[(0, 0), (0, 1), (1, 1)]]


def _m(r: random.Random) -> List[Stroke]:
    splay = r.choice((0.0, r.uniform(0.05, 0.2)))
    dip = r.uniform(0.45, 1.0)
    return [[(0, 1), (splay, 0), (0.5, dip), (1 - splay, 0), (1, 1)]]


def _n(r: random.Random) -> List[Stroke]:
    return [[(0, 1), (0, 0), (1, 1), (1, 0)]]


def _o(r: random.Random) -> List[Stroke]:
    start = r.uniform(-120, -60)
    return [_arc(0.5, 0.5, 0.5, 0.5, start, start + 360 + r.uniform(-8, 15), 30)]


def _p(r: random.Random) -> List[Stroke]:
    middle = r.uniform(0.45, 0.6)
    return [[(0, 1), (0, 0), (0.55, 0)] + _arc(0.55, middle / 2, 0.45, middle / 2, -90, 90) + [(0, middle)]]


def _q(r: random.Random) -> List[Stroke]:
    tail = r.uniform(0.5, 0.7)
    return _o(r) + [[(tail, tail), (1.05, 1.05)]]


def _r(r: random.Random) -> List[Stroke]:
    middle = r.uniform(0.45, 0.58)
    leg_from = r.uniform(0.2, 0.55)
    return [[(0, 1), (0, 0), (0.55, 0)] + _arc(0.55, middle / 2, 0.45, middle / 2, -90, 90) + [(0, middle)],
            [(leg_from, middle), (1, 1)]]


def _s(r: random.Random) -> List[Stroke]:
    upper = _arc(0.5, 0.25, 0.5, 0.25, -20, -270, 14)
    lower = _arc(0.5, 0.75, 0.5, 0.25, -90, 160, 14)
    return [upper + lower[1:]]


def _t(r: random.Random) -> List[Stroke]:
    return [[(0, 0), (1, 0)], [(0.5, 0), (0.5, 1)]]


def _u(r: random.Random) -> List[Stroke]:
    bottom = r.uniform(0.5, 0.75)
    curve = _arc(0.5, bottom, 0.5, 1 - bottom, 180, 0, 14)
    stroke = [(0, 0)] + curve + [(1, 0)]
    if r.random() < 0.3:
        return [stroke[:-1] + [(1, 0)], [(1, bottom), (1, 1)]]
    return [stroke]


def _v(r: random.Random) -> List[Stroke]:
    return [[(0, 0), (r.uniform(0.42, 0.58), 1), (1, 0)]]


def _w(r: random.Random) -> List[Stroke]:
    peak = r.uniform(0.0, 0.55)
    first, second = r.uniform(0.18, 0.3), r.uniform(0.7, 0.82)
    return [[(0, 0), (first, 1), (0.5, peak), (second, 1), (1, 0)]]


def _x(r: random.Random) -> List[Stroke]:
    return [[(0, 0), (1, 1)], [(1, 0), (0, 1)]]


def _y(r: random.Random) -> List[Stroke]:
    join = r.uniform(0.4, 0.6)
    if r.random() < 0.6:
        return [[(0, 0), (0.5, join)], [(1, 0), (0.5, join), (0.5, 1)]]
    return [[(1, 0), (0.25, 1)], [(0, 0), (0.62, join)]]


def _z(r: random.Random) -> List[Stroke]:
    strokes = [[(0, 0), (1, 0), (0, 1), (1, 1)]]
    if r.random() < 0.15:
        strokes.append([(0.25, 0.5), (0.75, 0.5)])
    return strokes


SKELETONS: Dict[str, Callable[[random.Random], List[Stroke]]] = {
    letter: function for letter, function in zip(LETTERS, (
        _a, _b, _c, _d, _e, _f, _g, _h, _i, _j, _k, _l, _m, _n, _o, _p, _q, _r, _s, _t, _u, _v, _w, _x, _y, _z))}
# How wide each capital is printed, for its height: (least, most).
WIDTHS = {letter: (0.55, 0.9) for letter in LETTERS}
WIDTHS.update(M=(0.8, 1.25), W=(0.9, 1.4), O=(0.7, 1.05), Q=(0.7, 1.05), C=(0.6, 0.95), G=(0.65, 1.0),
              I=(0.0, 0.0), J=(0.4, 0.7), L=(0.45, 0.75), X=(0.6, 1.0), A=(0.65, 1.0), V=(0.6, 1.0),
              N=(0.5, 0.9), Z=(0.6, 1.0))


def letter_strokes(letter: str, r: random.Random, height: float) -> List[np.ndarray]:
    """`letter`'s strokes, mm, centred on (0, 0): its skeleton, each point moved a little, sheared and sized."""
    low, high = WIDTHS[letter]
    width = height * r.uniform(low, high) if high else height * r.uniform(0.3, 0.55)
    if letter == "I" and r.random() < 0.5:
        width = height * 0.02
    shear = r.uniform(-0.15, 0.15)
    wobble = r.uniform(0.01, 0.045)
    bend = _smooth_bend(r, r.uniform(0.0, 0.06))
    strokes = []
    for stroke in SKELETONS[letter](r):
        points = np.array(stroke, float)
        if len(points) <= 6:                    # corners: each moved on its own; curves only bend, smoothly
            points += np.array([[r.gauss(0, wobble), r.gauss(0, wobble)] for _ in points])
        points += bend(points)
        points[:, 0] = (points[:, 0] - 0.5) * width
        points[:, 1] = (points[:, 1] - 0.5) * height
        points[:, 0] += shear * -points[:, 1]
        strokes.append(points)
    return strokes


def _smooth_bend(r: random.Random, amount: float) -> Callable[[np.ndarray], np.ndarray]:
    """A gentle warp of the unit box: two slow waves each way, so a curve stays a curve."""
    waves = [(r.uniform(0.5, 2.0), r.uniform(0, 2 * math.pi), r.uniform(0, 2 * math.pi)) for _ in range(4)]

    def bend(points: np.ndarray) -> np.ndarray:
        x, y = points[:, 0], points[:, 1]
        dx = sum(amount * math.cos(angle) * np.sin(2 * math.pi * frequency * (x * math.cos(angle) + y * math.sin(
            angle)) + phase) for frequency, angle, phase in waves[:2])
        dy = sum(amount * math.sin(angle) * np.sin(2 * math.pi * frequency * (x * math.sin(angle) - y * math.cos(
            angle)) + phase) for frequency, angle, phase in waves[2:])
        return np.column_stack((dx, dy))
    return bend


# --- the tile ----------------------------------------------------------------------------------------------------

def render(letter: Optional[str], turn: int, r: random.Random, is_dotted: bool = False,
           scribble: bool = False) -> np.ndarray:
    """One SIZE x SIZE grey sample: `letter` on a tile (None for a blank face), turned `turn` quarter turns."""
    scale = SIZE * SUPERSAMPLE / WINDOW_MM
    size = SIZE * SUPERSAMPLE
    face_tone = r.randint(140, 255)
    beyond = r.randint(20, 230)
    ink = r.randint(0, max(1, min(100, face_tone - 70)))

    canvas = Image.new("L", (size, size), beyond)
    draw = ImageDraw.Draw(canvas)
    _texture(draw, size, beyond, r)
    face = FACE_MM * r.uniform(0.98, 1.03)
    left = (WINDOW_MM - face) / 2
    box = [left * scale, left * scale, (left + face) * scale, (left + face) * scale]
    if r.random() < 0.7:                                    # a side face, where the camera sees one
        band = r.uniform(0.3, 2.5) * scale
        dx, dy = r.choice(((0, band), (0, -band), (band, 0), (-band, 0)))
        shade = int(np.clip(beyond * r.uniform(0.6, 1.3), 0, 255))
        draw.rounded_rectangle([box[0] + min(0, dx), box[1] + min(0, dy), box[2] + max(0, dx), box[3] + max(0, dy)],
                               radius=2 * scale, fill=shade)
    if r.random() < 0.25:                                   # a neighbour, close by
        offset = r.choice((-1, 1)) * (face + r.uniform(0.2, 3.0)) * scale
        horizontal = r.random() < 0.5
        draw.rounded_rectangle([box[0] + offset * horizontal, box[1] + offset * (not horizontal),
                                box[2] + offset * horizontal, box[3] + offset * (not horizontal)],
                               radius=2 * scale, fill=int(face_tone * r.uniform(0.85, 1.0)))
    rim = int(face_tone * r.uniform(0.7, 0.95))
    draw.rounded_rectangle(box, radius=r.uniform(1.0, 2.5) * scale, fill=rim)
    inset = r.uniform(0.3, 1.0) * scale
    draw.rounded_rectangle([box[0] + inset, box[1] + inset, box[2] - inset, box[3] - inset],
                           radius=r.uniform(0.5, 2.0) * scale, fill=face_tone)

    pen = r.uniform(0.7, 1.9) * scale
    if letter:
        height = r.uniform(9.0, 19.0)
        centre = np.array([WINDOW_MM / 2 + r.uniform(-1.8, 1.8), WINDOW_MM / 2 + r.uniform(-1.8, 1.8)])
        strokes = [(points + centre) * scale for points in letter_strokes(letter, r, height)]
        for points in strokes:
            _pen(draw, points, pen, ink)
        if is_dotted:
            everything = np.vstack(strokes)
            right, bottom = everything[:, 0].max(), everything[:, 1].max()
            span = everything.max(axis=0) - everything.min(axis=0)
            dot = (right + r.uniform(-0.15, 0.1) * span[0] + r.uniform(0.3, 2.0) * scale,
                   bottom + r.uniform(-0.3, 0.0) * span[1] + r.uniform(0.0, 1.0) * scale)
            radius = max(pen / 2, r.uniform(0.5, 1.2) * scale)
            draw.ellipse([dot[0] - radius, dot[1] - radius, dot[0] + radius, dot[1] + radius], fill=ink)
    if scribble:
        for _ in range(r.randint(1, 3)):
            points = [(r.uniform(6, 21) * scale, r.uniform(6, 21) * scale) for _ in range(r.randint(2, 4))]
            _pen(draw, np.array(points), pen, ink)
    return _finish(np.asarray(canvas, dtype=np.float32), scale, turn, face_tone, r)


def _pen(draw: ImageDraw.ImageDraw, points: np.ndarray, width: float, ink: int) -> None:
    """A round marker along the points, its ends rounded; a skip or two where the ink ran thin."""
    flat = [tuple(point) for point in points]
    draw.line(flat, fill=ink, width=max(1, round(width)), joint="curve")
    for x, y in (flat[0], flat[-1]):
        draw.ellipse([x - width / 2, y - width / 2, x + width / 2, y + width / 2], fill=ink)


def _texture(draw: ImageDraw.ImageDraw, size: int, tone: int, r: random.Random) -> None:
    strength = r.uniform(0, 40)
    for _ in range(r.randint(0, 40)):
        x = r.uniform(0, size)
        shade = int(np.clip(tone + r.uniform(-strength, strength), 0, 255))
        draw.line([(x, 0), (x + r.uniform(-size / 3, size / 3), size)], fill=shade, width=r.randint(1, 8))


def _finish(image: np.ndarray, scale: float, turn: int, face_tone: int, r: random.Random) -> np.ndarray:
    """Light across the face, pose error, the camera's softness, noise, exposure, the turn, the normalisation."""
    size = image.shape[0]
    if r.random() < 0.6:                                    # light falling off, or a shadow's soft edge
        angle = r.uniform(0, 2 * math.pi)
        rows, columns = np.mgrid[0:size, 0:size] / size - 0.5
        ramp = columns * math.cos(angle) + rows * math.sin(angle)
        if r.random() < 0.3:
            ramp = 1 / (1 + np.exp(-ramp * r.uniform(8, 30)))
        image = image * (1 - r.uniform(0, 0.5) * (ramp - ramp.min()) / max(np.ptp(ramp), 1e-6))
    image = _pose_error(image, scale, r)
    image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    image = cv2.GaussianBlur(image, (0, 0), r.uniform(0.3, 1.3))
    image = image * r.uniform(0.8, 1.5) + r.uniform(-30, 30)       # over-exposed often: the faces clip at 255
    image += np.random.default_rng(r.randrange(1 << 30)).normal(0, r.uniform(0, 6), image.shape)
    image = np.rot90(np.clip(image, 0, 255), turn)
    return normalise(image.astype(np.uint8))


def _pose_error(image: np.ndarray, scale: float, r: random.Random) -> np.ndarray:
    size = image.shape[0]
    centre = (size / 2, size / 2)
    matrix = cv2.getRotationMatrix2D(centre, r.uniform(-4, 4), r.uniform(0.96, 1.04))
    matrix[:, 2] += (r.uniform(-1.0, 1.0) * scale, r.uniform(-1.0, 1.0) * scale)
    return cv2.warpAffine(image, matrix, (size, size), borderMode=cv2.BORDER_REFLECT)


def normalise(grey: np.ndarray) -> np.ndarray:
    """Face near white, ink near black, whatever the exposure: the same function runs on the real faces."""
    grey = grey.astype(np.float32)
    middle = grey[SIZE // 4:3 * SIZE // 4, SIZE // 4:3 * SIZE // 4]
    low, high = np.percentile(middle, 1), np.percentile(middle, 90)
    return np.clip((grey - low) / max(high - low, 1.0) * 255, 0, 255).astype(np.uint8)


def sample(r: random.Random) -> Tuple[np.ndarray, int]:
    """One training sample and its class: about half letters the right way up, the rest turned, blank or scribbled."""
    letter = r.choice(LETTERS)
    kind = r.random()
    is_dotted = r.random() < (0.6 if letter in "MW" else 0.15)
    if kind < 0.06:
        return render(None, r.randrange(4), r, scribble=r.random() < 0.5), NOT_UPRIGHT
    turn = 0 if kind < 0.5 else r.randint(1, 3)
    return render(letter, turn, r, is_dotted), label(letter, turn, is_dotted)


def preview(count: int = 48, seed: int = 0) -> np.ndarray:
    """A sheet of samples, their classes over them: to look at what the reader is taught."""
    r = random.Random(seed)
    tiles: Sequence[np.ndarray] = []
    for _ in range(count):
        image, cls = sample(r)
        image = cv2.cvtColor(cv2.resize(image, (128, 128), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        cv2.putText(image, LETTERS[cls] if cls < NOT_UPRIGHT else "-", (4, 22), 0, 0.7, (0, 0, 255), 2)
        tiles.append(image)
    rows = [np.hstack(tiles[index:index + 8]) for index in range(0, len(tiles), 8)]
    return np.vstack(rows)
