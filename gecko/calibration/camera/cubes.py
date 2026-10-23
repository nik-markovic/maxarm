"""The calibration cubes in a frame: the coloured blobs that could be them, and which colour each would be.

Every decision is measured in the frame itself; nothing is tuned on a camera.

1. **The desk's colour** is the frame's most common chroma (Lab a*, b*): the desk
   fills most of any frame worth calibrating on.
2. **Colourful** is chroma far from the desk's. Where to cut is read off this
   frame's own histogram (Otsu).
3. **Blobs.** A cube's shaded side keeps the cube's hue but has less colour, and
   can fall below the cut: each blob takes in the pixels round it of its own hue
   above part of the cut -- if that leaves it more convex, as a cube's missing side
   does and a glossy desk's tint or dark plastic beside it does not. Blobs of one
   hue that nearly touch are joined.
4. **Cube-shaped.** What hangs off a blob by a thin neck -- a cable, a shadow's
   fringe -- is cut off at the blob's own scale. A cube's silhouette is convex and
   must be whole, so blobs far from convex, or that run off the frame, are dropped.
5. **Names.** Red is the warm one: the hue nearest Lab's red (+a*). Green and blue
   lie across the chroma plane from it, and which is which is left to the fit.

Which blobs are the cubes is not settled here either: the arm's base can look as
green as the green cube. `fit.py` tries each naming `namings()` offers against
where the arm says the cubes stand, and keeps the one a camera can explain.
"""

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterator, List

import cv2
import numpy as np

COLOURS = ("blue", "green", "red")      # as the page lists them

# A cube is convex: its blob covers most of its own convex hull, less where glare
# washes a face out. Anything more ragged is not a cube.
MIN_SOLIDITY = 0.6
# The largest blobs that are tried as cubes; each naming of them costs a fit's start.
MAX_BLOBS = 8
# Hue windows from the colour names, in degrees from +a*: red is the warm side of
# the chroma plane, and green and blue lie across from it.
RED_WITHIN_DEG = 75.0
COOL_BEYOND_DEG = 90.0
# Blobs this far apart, as a share of the smaller one's size, may be one cube's faces.
JOIN_GAP = 0.08
JOIN_HUE_DEG = 40.0
# Anything joined to a blob by a neck thinner than this share of its size is cut off.
TRIM = 0.15
# A blob grows into its shaded side: pixels within this many degrees of its hue, above
# this share of the frame's cut, connected to it, within this share of its size round it.
GROW_HUE_DEG = 30.0
GROW_CUT = 0.6
GROW_MARGIN = 0.3


class CubesNotFound(Exception):
    """The frame does not show three cubes that can be told apart. The message says what it does show."""


@dataclass
class Blob:
    contour: np.ndarray     # the outline, in frame pixels
    mask: np.ndarray        # the frame's size, 255 inside
    area: int
    centroid: np.ndarray    # px, py
    hue: float              # degrees from +a* towards +b*, measured from the desk's colour
    solidity: float         # area over the area of its convex hull

    @property
    def size(self) -> float:
        return math.sqrt(self.area)


def find_blobs(frame: np.ndarray, is_warm_only: bool = False) -> List[Blob]:
    """The coloured blobs that could be cubes, largest first.

    `is_warm_only`: of the warm pixels only, the red cube's hue -- so that a red
    cube standing against a green or a blue one in the picture is a blob of its own.
    """
    chroma = chroma_from_desk(frame, desk_chroma(frame))
    distance = np.hypot(chroma[..., 0], chroma[..., 1])
    scaled = np.clip(distance * 4.0, 0, 255).astype(np.uint8)     # chroma runs about 0-64
    cut, mask = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if is_warm_only:
        hue = np.degrees(np.arctan2(chroma[..., 1], chroma[..., 0]))
        mask[np.abs(hue) > RED_WITHIN_DEG] = 0
    speck = _odd(min(frame.shape[:2]) / 400)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((speck, speck), np.uint8))
    min_area = frame.shape[0] * frame.shape[1] * 5e-4
    height, width = frame.shape[:2]
    smooth = cv2.GaussianBlur(scaled, (0, 0), min(frame.shape[:2]) / 400)      # a face is even; speckle is not
    faint = smooth > GROW_CUT * cut
    blobs = [_grow(blob, chroma, faint) for blob in _blobs(mask, chroma, min_area)]
    blobs = _join_faces(blobs, chroma)
    whole = [blob for blob in blobs if not _touches_edge(blob.contour, width, height)]
    trimmed = [blob for blob in (_trim(blob, chroma, min_area) for blob in whole) if blob]
    cubelike = sorted((blob for blob in trimmed if blob.solidity >= MIN_SOLIDITY), key=lambda blob: -blob.area)
    return cubelike[:MAX_BLOBS]


def namings(blobs: List[Blob]) -> Iterator[Dict[str, Blob]]:
    """Every way three of the blobs can be the red, green and blue cubes that their hues allow.

    Green and blue come both ways round, in hue order first: on some cameras they
    are a few degrees apart. Swapping two cubes mirrors their triangle, which no
    camera above the desk can show, so the fit tells them apart.
    """
    for three in itertools.combinations(blobs, 3):
        red = min(three, key=lambda blob: _apart(blob.hue, 0.0))
        cool = sorted((blob for blob in three if blob is not red), key=lambda blob: (blob.hue - red.hue) % 360)
        if is_warm(red) and all(_apart(blob.hue, red.hue) > COOL_BEYOND_DEG for blob in cool):
            yield {"red": red, "green": cool[0], "blue": cool[1]}
            yield {"red": red, "green": cool[1], "blue": cool[0]}


def is_warm(blob: Blob) -> bool:
    """On the red side of the chroma plane: the hue a red cube could have."""
    return _apart(blob.hue, 0.0) <= RED_WITHIN_DEG


def describe(blobs: List[Blob]) -> str:
    """What was found, for a message: how many blobs, and which are warm and which cool."""
    if not blobs:
        return "no whole, cube-shaped coloured blob"
    kinds = ", ".join("warm" if is_warm(blob) else "cool" for blob in blobs)
    return f"{len(blobs)} whole, cube-shaped coloured blob{'s' if len(blobs) > 1 else ''} ({kinds})"


def desk_chroma(frame: np.ndarray) -> np.ndarray:
    """The desk's a*, b*: the frame's most common chroma, its histogram smoothed."""
    chroma = _chroma(frame)
    histogram, a_edges, b_edges = np.histogram2d(chroma[..., 0].ravel(), chroma[..., 1].ravel(),
                                                 bins=128, range=[[-64, 64], [-64, 64]])
    histogram = cv2.GaussianBlur(histogram.astype(np.float32), (0, 0), 1.5)
    i, j = np.unravel_index(int(histogram.argmax()), histogram.shape)
    return np.array([(a_edges[i] + a_edges[i + 1]) / 2, (b_edges[j] + b_edges[j + 1]) / 2], np.float32)


def chroma_from_desk(image: np.ndarray, desk: np.ndarray) -> np.ndarray:
    """Each pixel's a*, b* less the desk's."""
    return _chroma(image) - desk


def _chroma(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)[..., 1:] - 128.0


def _blobs(mask: np.ndarray, chroma: np.ndarray, min_area: float) -> List[Blob]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= min_area:
            blobs.append(_blob((labels == index).astype(np.uint8) * 255, chroma))
    return blobs


def _blob(mask: np.ndarray, chroma: np.ndarray) -> Blob:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    inside = mask > 0
    rows, cols = np.nonzero(inside)
    total = chroma[inside].sum(axis=0)
    area = int(inside.sum())
    hull = max(cv2.contourArea(cv2.convexHull(contour)), 1.0)
    return Blob(contour=contour, mask=mask, area=area, centroid=np.array([cols.mean(), rows.mean()]),
                hue=math.degrees(math.atan2(total[1], total[0])) % 360, solidity=area / hull)


def _join_faces(blobs: List[Blob], chroma: np.ndarray) -> List[Blob]:
    """Join blobs of one hue that nearly touch: a cube's sides, cut apart where the shade changes.

    Only if the two together are at least as convex as the more convex of them, as
    a cube's faces are, and a cube with a dark thing of its hue beside it is not.
    """
    joined = True
    while joined:
        joined = False
        for first, second in itertools.combinations(blobs, 2):
            if _apart(first.hue, second.hue) > JOIN_HUE_DEG:
                continue
            gap = _odd(JOIN_GAP * min(first.size, second.size))
            box = _shared_box(first.contour, second.contour, gap, first.mask.shape)
            if box is None:
                continue
            kernel = np.ones((gap, gap), np.uint8)
            grown = cv2.dilate(first.mask[box], kernel)
            if not np.any(grown & second.mask[box]):
                continue
            union = first.mask | second.mask
            union[box] = cv2.morphologyEx(union[box], cv2.MORPH_CLOSE, kernel)
            together = _blob(union, chroma)
            if together.solidity < max(first.solidity, second.solidity):
                continue
            blobs = [blob for blob in blobs if blob is not first and blob is not second] + [together]
            joined = True
            break
    return blobs


def _shared_box(first: np.ndarray, second: np.ndarray, gap: int, shape) -> "tuple | None":
    """The frame slice holding both outlines and the gap round them; None if they are further apart than that."""
    ax, ay, aw, ah = cv2.boundingRect(first)
    bx, by, bw, bh = cv2.boundingRect(second)
    if ax > bx + bw + gap or bx > ax + aw + gap or ay > by + bh + gap or by > ay + ah + gap:
        return None
    x0, y0 = max(0, min(ax, bx) - gap), max(0, min(ay, by) - gap)
    x1, y1 = min(shape[1], max(ax + aw, bx + bw) + gap), min(shape[0], max(ay + ah, by + bh) + gap)
    return slice(y0, y1), slice(x0, x1)


def _grow(blob: Blob, chroma: np.ndarray, faint: np.ndarray) -> Blob:
    """The blob with what joins it of its own hue, too faint for the cut: a cube's shaded side.

    Kept only if the blob comes out more convex: a cube's silhouette is, and its
    shaded side completes it, where a tint on the desk or a dark thing behind it
    would make it ragged.
    """
    x, y, w, h = cv2.boundingRect(blob.contour)
    margin = int(GROW_MARGIN * max(w, h))
    y0, y1 = max(0, y - margin), min(blob.mask.shape[0], y + h + margin)
    x0, x1 = max(0, x - margin), min(blob.mask.shape[1], x + w + margin)
    near = chroma[y0:y1, x0:x1]
    hue = np.degrees(np.arctan2(near[..., 1], near[..., 0]))
    alike = faint[y0:y1, x0:x1] & (np.abs((hue - blob.hue + 180.0) % 360.0 - 180.0) <= GROW_HUE_DEG)
    inside = blob.mask[y0:y1, x0:x1] > 0
    count, labels = cv2.connectedComponents((alike | inside).astype(np.uint8), connectivity=8)
    keep = np.isin(labels, np.unique(labels[inside]))
    mask = blob.mask.copy()
    mask[y0:y1, x0:x1] = np.where(keep, 255, 0).astype(np.uint8)
    grown = _blob(mask, chroma)
    return grown if grown.solidity >= blob.solidity else blob


def _trim(blob: Blob, chroma: np.ndarray, min_area: float) -> "Blob | None":
    """The blob without what hangs off it by a thread -- a cable, a shadow's fringe -- at its own scale."""
    x, y, w, h = cv2.boundingRect(blob.contour)
    box = slice(y, y + h), slice(x, x + w)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(TRIM * blob.size),) * 2)
    opened = cv2.morphologyEx(blob.mask[box], cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if count < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < min_area:
        return None
    mask = np.zeros_like(blob.mask)
    mask[box] = (labels == largest).astype(np.uint8) * 255
    return _blob(mask, chroma)


def _touches_edge(contour: np.ndarray, width: int, height: int) -> bool:
    x, y, w, h = cv2.boundingRect(contour)
    return x <= 0 or y <= 0 or x + w >= width or y + h >= height


def _apart(first: float, second: float) -> float:
    """Degrees between two hues, 0 to 180."""
    return abs((first - second + 180.0) % 360.0 - 180.0)


def _odd(value: float) -> int:
    return max(3, int(round(value)) | 1)
