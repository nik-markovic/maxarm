"""The arm's base found in a frame by its own edges: where the arm's axis stands, and which way the base is turned.

    found = find_base(frame, desk_to_pixel)      # desk_to_pixel: a 3x4 from desk mm, its origin near the axis
    found.x, found.y, found.turn_deg             # the base's center -- the arm's axis -- and its turn, desk mm
    found.by_nuts, found.sure_mm                 # laid on its nuts (else the whole shape), and how far to trust it
    found.parts                                  # each nut and cup rim looked for alone

The base's shape, as measured: four suction cups, 41 mm
across, flush on the desk, their centers on a 120 mm square round the arm's axis,
its sides along the arm's x and y; over each, on the black plate's top 10 mm up, a
steel nut 7 mm across its flats and 3 mm tall under a rounded tip, 5 mm across
narrowing to 4, up to 18 mm; and the plate's four straight central edges, 124.5 mm
apart, straight for 28 mm either side of their middles.

1. **The whole shape.** Moved and turned over the desk onto the frame's edges
   (Canny): each of its points costs by how far it lies from the nearest edge, no
   more than TRUNCATE_PX, so a part hidden by the arm or out of the picture costs the
   same wherever the base is. A coarse search round the start, a fine one round the
   best. It only brings the parts near: on two frames of one unmoved camera, the arm
   parked differently, it moved 3.7 mm and 1.2 deg, and wood grain away
   from the base fits the shape as well as the base does.
2. **Each part alone.** Each nut, and each cup's rim, moved alone round where the
   whole shape put it, over edges found in a crop round it with Canny's thresholds
   from the crop's own gradients: a dark nut on the dark plate shows as plainly as a
   shiny one. A part is found where its edges are plainly there: its best place
   inside the search, and far better than the places round it. A part hidden by the
   arm, or not in the picture, is not. On those two frames each nut found came back
   within 0.5 mm.
3. **The base on its nuts.** Nuts found at two corners or more: the base moved and
   turned so that all of them together lie best on their own edges, held 120 mm
   apart -- one nut alone can slide a millimeter or two over its highlight and hex,
   the shape holds them to each other. The arm's axis to about NUTS_SURE_MM. Fewer:
   the whole shape's place, to about WHOLE_SURE_MM. The cups' rims are looked for and
   reported, not used: translucent on a grained desk they are found or not by the
   light.

Nothing is assumed of the frame's colors or brightness: only its edges.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .model import project

# The base, mm, round the arm's axis, sides along its x and y.
CUP_SPACING_MM = 120.0          # the nuts' centers, and the cups' under them
BASE_SUCTION_CUP_RADIUS_MM = 20.5   # the base's four suction cups holding it to the desk (not the arm's cup): 41 mm,
                                   # 161 mm across their outer rims on the 120 mm square
PLATE_TOP_MM = 10.0
NUT_TOP_MM = 13.0               # the hex, 3 mm tall on the plate
NUT_HALF_MM = 3.5               # 7 mm across the flats, drawn as a circle
TIP_BOTTOM_HALF_MM, TIP_TOP_HALF_MM, TIP_TOP_MM, TIP_MM = 2.5, 2.0, 17.0, 18.0
PLATE_EDGE_MM = 124.5           # the plate's straight central edges, apart
PLATE_EDGE_HALF_MM = 28.0       # each straight for this either side of its middle
CORNERS = ((1, 1), (1, -1), (-1, -1), (-1, 1))      # the cups' corners, in halves of CUP_SPACING_MM

# The edges: Gaussian blur, Canny's two thresholds; a shape point further than TRUNCATE_PX from an edge costs no more.
BLUR_PX = 5
CANNY = (30, 90)
TRUNCATE_PX = 6.0
# 1. The whole shape: (step mm, span mm, step deg, span deg), coarse then fine round the best.
WHOLE_SEARCH = ((0.5, 8.0, 0.5, 3.0), (0.1, 1.0, 0.1, 0.5))
# 2. Each part alone, round where the whole shape put it: a coarse search over PART_SPAN_MM, a fine one round its
# best. Found when its best costs at most FOUND_COST_SHARE of the cost round it (the coarse places from RING_SHARE of
# the span outward), is not at the edge of the search, and has IN_PICTURE_SHARE of its points in the picture. Its
# edges: Canny's upper threshold the gradient LOCAL_EDGE_SHARE of the crop's pixels stay under, the lower half of it.
PART_SPAN_MM, PART_STEP_MM = 5.0, 0.5
PART_FINE_SPAN_MM, PART_FINE_STEP_MM = 0.6, 0.1
RING_SHARE = 0.7
FOUND_COST_SHARE = 0.6
IN_PICTURE_SHARE = 0.8
LOCAL_EDGE_SHARE = 0.9
# 3. The nuts together: (step mm, span mm, step deg, span deg), coarse then fine round where they were found alone.
NUTS_SEARCH = ((0.25, 2.5, 0.25, 1.5), (0.05, 0.3, 0.05, 0.3))
# How far the axis is to be trusted, mm: laid on nuts, or on the whole shape.
NUTS_SURE_MM = 1.0
WHOLE_SURE_MM = 4.0


@dataclass(frozen=True)
class Part:
    name: str                   # "nut" or "cup", and its corner: "nut +x+y"
    corner: Tuple[int, int]
    found: bool
    x: float                    # where it was found alone, desk mm (its center); where the shape puts it if not found
    y: float
    contrast: float             # its best place's cost against the places round it: 0 none, 1 all
    off_mm: Optional[float] = None      # found: how far from where the base found puts it

    @property
    def is_nut(self) -> bool:
        return self.name.startswith("nut")


@dataclass(frozen=True)
class FoundBase:
    x: float                    # the base's center, the arm's axis, desk mm
    y: float
    turn_deg: float             # the base turned this much from the desk's axes, + from +x toward +y
    by_nuts: bool               # laid on its nuts found alone; False: the whole shape's place
    shape: Tuple[float, float, float]   # where the whole shape went: x, y, turn_deg
    parts: List[Part] = field(default_factory=list)

    @property
    def sure_mm(self) -> float:
        """How far the axis is to be trusted, mm."""
        return NUTS_SURE_MM if self.by_nuts else WHOLE_SURE_MM

    @property
    def found_parts(self) -> List[Part]:
        return [part for part in self.parts if part.found]

    def describe(self) -> str:
        found = ", ".join(f"{part.name} {part.off_mm:.1f}" for part in self.found_parts)
        how = "laid on its nuts" if self.by_nuts else "the whole shape on the edges (nuts at fewer than two corners)"
        return (f"the base at ({self.x:+.1f}, {self.y:+.1f}) mm, turned {self.turn_deg:+.2f} deg, {how}; "
                f"{'parts found alone, mm from it: ' + found if found else 'no part found alone'}")

    def to_json(self) -> dict:
        return {"x": round(self.x, 3), "y": round(self.y, 3), "turn_deg": round(self.turn_deg, 4),
                "by_nuts": self.by_nuts, "shape": [round(value, 3) for value in self.shape],
                "parts": [{"name": part.name, "found": part.found, "x": round(part.x, 3), "y": round(part.y, 3),
                           "contrast": round(part.contrast, 3),
                           "off_mm": None if part.off_mm is None else round(part.off_mm, 3)} for part in self.parts]}

    @classmethod
    def from_json(cls, stored: dict) -> "FoundBase":
        parts = [Part(part["name"], _corner_of(part["name"]), part["found"], part["x"], part["y"], part["contrast"],
                      part.get("off_mm")) for part in stored.get("parts", [])]
        return cls(stored["x"], stored["y"], stored["turn_deg"], stored["by_nuts"], tuple(stored["shape"]), parts)


def find_base(frame: np.ndarray, desk_to_pixel: np.ndarray, start: Tuple[float, float, float] = (0.0, 0.0, 0.0)
              ) -> FoundBase:
    """The base in `frame`, desk mm through `desk_to_pixel`, searched for round `start` (x, y mm, turn radians)."""
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (BLUR_PX, BLUR_PX), 0)
    distance = cv2.distanceTransform(255 - cv2.Canny(gray, *CANNY), cv2.DIST_L2, 3)
    cost = _Cost(distance, desk_to_pixel)
    whole = shape()
    best = (cost(whole, start), tuple(start))
    for step, span, turn_step, turn_span in WHOLE_SEARCH:
        center = best[1]
        for dx in np.arange(-span, span + 1e-9, step):
            for dy in np.arange(-span, span + 1e-9, step):
                for dt in np.radians(np.arange(-turn_span, turn_span + 1e-9, turn_step)):
                    pose = (center[0] + dx, center[1] + dy, center[2] + dt)
                    trial = cost(whole, pose)
                    if trial < best[0]:
                        best = (trial, pose)
    pose = best[1]
    shapes = {name: points for name, _, points in part_shapes()}
    parts = [_part_alone(gray, desk_to_pixel, name, corner, points, pose) for name, corner, points in part_shapes()]
    nuts = [part for part in parts if part.found and part.is_nut]
    by_nuts = len({part.corner for part in nuts}) >= 2
    laid = _on_nuts(gray, desk_to_pixel, [(part, shapes[part.name]) for part in nuts]) if by_nuts else pose
    parts = [part if not part.found else
             Part(part.name, part.corner, True, part.x, part.y, part.contrast,
                  float(math.dist((part.x, part.y), moved(np.array([[*_corner_mm(part.corner), 0.0]]), laid)[0, :2])))
             for part in parts]
    return FoundBase(laid[0], laid[1], math.degrees(laid[2]), by_nuts,
                     (float(pose[0]), float(pose[1]), math.degrees(pose[2])), parts)


def shape() -> np.ndarray:
    """Points of the whole base, desk mm round the arm's axis, (x, y, z) rows: rims, nuts, tips, plate edges."""
    edge, half = PLATE_EDGE_MM / 2, PLATE_EDGE_HALF_MM
    along = np.linspace(-half, half, 20)
    plate = [(x, y, PLATE_TOP_MM) for t in along for x, y in ((edge, t), (-edge, t), (t, edge), (t, -edge))]
    return np.vstack([points for _, _, points in part_shapes()] + [np.array(plate)])


def part_shapes() -> List[Tuple[str, Tuple[int, int], np.ndarray]]:
    """Each part looked for alone: its name, its corner, and its points, desk mm round the arm's axis."""
    parts = []
    for corner in CORNERS:
        cx, cy = _corner_mm(corner)
        label = f"{'+' if corner[0] > 0 else '-'}x{'+' if corner[1] > 0 else '-'}y"
        nut = np.vstack([ring((cx, cy), radius, z, 16) for radius, z in
                         ((NUT_HALF_MM, PLATE_TOP_MM), (NUT_HALF_MM, NUT_TOP_MM), (TIP_TOP_HALF_MM, TIP_TOP_MM))])
        parts.append((f"nut {label}", corner, nut))
        parts.append((f"cup {label}", corner, ring((cx, cy), BASE_SUCTION_CUP_RADIUS_MM, 0.0, 48)))
    return parts


def ring(center: Sequence[float], radius: float, z: float, count: int) -> np.ndarray:
    turns = np.linspace(0.0, 2 * math.pi, count, endpoint=False)
    return np.column_stack((center[0] + radius * np.cos(turns), center[1] + radius * np.sin(turns),
                            np.full(count, z)))


def moved(points: np.ndarray, pose: Sequence[float]) -> np.ndarray:
    """(x, y, z) rows turned by pose[2] radians about the axis, then moved by pose[0], pose[1]."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.column_stack((c * points[:, 0] - s * points[:, 1] + pose[0], s * points[:, 0] + c * points[:, 1] + pose[1],
                            points[:, 2]))


class _Cost:
    """The mean squared distance, truncated, from shape points through the camera to the nearest edge of a distance
    map whose top left is `origin` in the frame."""

    def __init__(self, distance: np.ndarray, desk_to_pixel: np.ndarray, origin: Tuple[int, int] = (0, 0)) -> None:
        self.distance, self.matrix, self.origin = distance, desk_to_pixel, np.array(origin, float)
        self.height, self.width = distance.shape

    def __call__(self, points: np.ndarray, pose: Sequence[float]) -> float:
        pixels = project(self.matrix, moved(points, pose)) - self.origin
        inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < self.width - 1) & (pixels[:, 1] >= 0)
                  & (pixels[:, 1] < self.height - 1))
        if not inside.any():
            return TRUNCATE_PX ** 2
        d = self.distance[pixels[inside, 1].astype(int), pixels[inside, 0].astype(int)]
        return float(np.mean(np.minimum(d, TRUNCATE_PX) ** 2))


def _local_cost(gray: np.ndarray, desk_to_pixel: np.ndarray, placed: np.ndarray) -> Tuple[_Cost, float]:
    """The edge cost over a crop round the part placed, wide enough for its search, its edges by the crop's own
    gradients; and the share of the part's points in the picture."""
    pixels = project(desk_to_pixel, placed)
    height, width = gray.shape
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)).mean()
    scale = np.linalg.norm(project(desk_to_pixel, placed[:1] + (PART_SPAN_MM, 0.0, 0.0)) - pixels[:1])
    margin = int(np.ceil(1.5 * scale + 3 * TRUNCATE_PX))
    low = np.maximum(np.floor(pixels.min(axis=0)).astype(int) - margin, 0)
    high = np.minimum(np.ceil(pixels.max(axis=0)).astype(int) + margin, (width, height))
    if np.any(high - low < 8):
        return _Cost(np.full((8, 8), TRUNCATE_PX, np.float32), desk_to_pixel), 0.0
    crop = gray[low[1]:high[1], low[0]:high[0]]
    gradient = np.hypot(cv2.Sobel(crop, cv2.CV_32F, 1, 0), cv2.Sobel(crop, cv2.CV_32F, 0, 1))
    upper = max(float(np.quantile(gradient, LOCAL_EDGE_SHARE)), 1.0)
    edges = cv2.Canny(crop, upper / 2, upper, L2gradient=True)
    return _Cost(cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3), desk_to_pixel, tuple(low)), float(inside)


def _part_alone(gray: np.ndarray, desk_to_pixel: np.ndarray, name: str, corner: Tuple[int, int], points: np.ndarray,
                pose: Sequence[float]) -> Part:
    """The part moved alone round where the shape at `pose` puts it."""
    placed = moved(points, pose)
    center = moved(np.array([[*_corner_mm(corner), 0.0]]), pose)[0, :2]
    cost, in_picture = _local_cost(gray, desk_to_pixel, placed)
    offsets = np.arange(-PART_SPAN_MM, PART_SPAN_MM + 1e-9, PART_STEP_MM)
    grid = np.array([[cost(placed, (dx, dy, 0.0)) for dx in offsets] for dy in offsets])
    row, column = np.unravel_index(np.argmin(grid), grid.shape)
    gap = np.hypot(*np.meshgrid(offsets, offsets))
    round_it = float(np.median(grid[gap >= RING_SHARE * PART_SPAN_MM]))
    contrast = float(1.0 - grid[row, column] / max(round_it, 1e-9))
    is_inner = 0 < row < len(offsets) - 1 and 0 < column < len(offsets) - 1
    found = bool(is_inner and in_picture >= IN_PICTURE_SHARE and grid[row, column] <= FOUND_COST_SHARE * round_it)
    if not found:
        return Part(name, corner, False, float(center[0]), float(center[1]), contrast)
    best = (grid[row, column], (offsets[column], offsets[row]))
    fine = np.arange(-PART_FINE_SPAN_MM, PART_FINE_SPAN_MM + 1e-9, PART_FINE_STEP_MM)
    for dx in fine + offsets[column]:
        for dy in fine + offsets[row]:
            trial = cost(placed, (dx, dy, 0.0))
            if trial < best[0]:
                best = (trial, (dx, dy))
    return Part(name, corner, True, float(center[0] + best[1][0]), float(center[1] + best[1][1]), contrast)


def _on_nuts(gray: np.ndarray, desk_to_pixel: np.ndarray, nuts: List[Tuple[Part, np.ndarray]]
             ) -> Tuple[float, float, float]:
    """The base's place, x, y mm and turn radians, at which the nuts found lie best together on their own edges,
    searched round the turn and shift that lays them on where each was found alone."""
    turn, shift = _laid(np.array([_corner_mm(part.corner) for part, _ in nuts]),
                        np.array([(part.x, part.y) for part, _ in nuts]))
    start = (float(shift[0]), float(shift[1]), math.atan2(turn[1, 0], turn[0, 0]))
    costs = [(_local_cost(gray, desk_to_pixel, moved(points, start))[0], points) for _, points in nuts]

    def cost(pose: Sequence[float]) -> float:
        return sum(each(points, pose) for each, points in costs)

    best = (cost(start), start)
    for step, span, turn_step, turn_span in NUTS_SEARCH:
        center = best[1]
        for dx in np.arange(-span, span + 1e-9, step):
            for dy in np.arange(-span, span + 1e-9, step):
                for dt in np.radians(np.arange(-turn_span, turn_span + 1e-9, turn_step)):
                    pose = (center[0] + dx, center[1] + dy, center[2] + dt)
                    trial = cost(pose)
                    if trial < best[0]:
                        best = (trial, pose)
    return best[1]


def _laid(model: np.ndarray, seen: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The turn and shift that lay the `model` rows best on the `seen` rows: seen = turn @ model + shift."""
    model_mean, seen_mean = model.mean(axis=0), seen.mean(axis=0)
    left, _, right = np.linalg.svd((seen - seen_mean).T @ (model - model_mean))
    turn = left @ np.diag([1.0, np.sign(np.linalg.det(left @ right))]) @ right
    return turn, seen_mean - turn @ model_mean


def _corner_mm(corner: Tuple[int, int]) -> Tuple[float, float]:
    return corner[0] * CUP_SPACING_MM / 2, corner[1] * CUP_SPACING_MM / 2


def _corner_of(name: str) -> Tuple[int, int]:
    label = name.split()[-1]
    return (1 if label[0] == "+" else -1, 1 if label[2] == "+" else -1)
