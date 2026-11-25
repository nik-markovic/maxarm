"""The arm laid on the camera's reality: its axis where the camera sees the base, its turn from the cubes' bearings,
and what is left at each point the arm's own error -- the mesh its raw-to-true conversion takes on.

    found = base.find_base(frame, alone.desk_to_pixel)        # the base by its edges, in the cubes' desk space
    tie = fit_tie(found, tie_points(alone, points))           # CALIBRATE: the frame, and the mesh of its cubes
    tie = kept_tie(fit.anchor, found, tie_points(alone, points))   # SAVE without its own camera: CALIBRATE's frame
    anchored(alone, tie)                                      # camera.json, the arm laid by the tie
    tie.mesh(made)                                            # config/arm-true.json: maxarm/mesh.py

The camera is the cubes' edges alone (`fit.py`): reality, its 40 mm cubes its only
length. Everything here is in the desk space the cubes' calibration laid by
their readings (`cube_fit()`), which only has to be near the arm's: the tie says
where the arm really stands in it.

**The frame.** CALIBRATE lays the arm: its axis where the base is, its turn from the
cubes' bearings. SAVE fits the camera again to every point's cube and lays the arm
again the same way, on all the points (`camera/refine.py`); only when it cannot fit
a camera does it keep CALIBRATE's frame and add the points to the correction.

**What is trusted, and how much.** Each point is a cube seen by the camera (where
its edges put it) and the firmware's position over it, made true by the formula
(`maxarm/raw.py`): its bearing and reach from the arm's axis.

- *The base, by its edges* (`applib/camera/base.py`): where the arm's axis stands,
  to about 1 mm laid on its nuts, 4 mm on its whole shape (`FoundBase.sure_mm`); and
  the base's turn, to about BASE_TURN_DEG. The camera, trusted most: it sets the axis.
- *The points' bearings*: each says which way the arm pointed at a place the camera
  saw, to within SEEN_MM across (the cube by its edges, centered under the cup) and
  one base servo pulse; far out, the ~1.5 mm the arm can be off as well
  (FAR_MM); so the middle of the reach counts most. Together they set the turn, and
  where the lines they draw meet says where they put the axis.
- *The points' reaches*: not trusted -- the camera and hand measurements disagree
  by 2-4 mm -- so they set nothing while the base is found; only without the base
  do they, with the bearings, place the axis.

The fit is least squares over all of it, each misfit in its own spread, far misfits
counting less (Huber).

**The mesh.** At each point, the cube where the camera sees it, taken about the
axis and turn, against where the formula puts the cup: further out (mm) and further
round (degrees). Those rows are the correction (`maxarm/mesh.py`): the pattern they
share over the whole desk, and a mesh of what it leaves at each, exact at a point,
fading out beyond the points.

**What does not fit.** Each point is judged without it, by its reach: how much
further out the camera saw its cube than the formula put the cup, against what the
correction of the other points says there. The arm's reach is off by much the same all
over the desk (+1 to +3 mm), its turn is not (the sides up to ~1.5
deg apart), so the reach is what shows a recording gone wrong -- the cube moved
after RECORD, or the cup off it. The worst, if further than DOES_NOT_FIT_MM, is
named, left out of the mesh, and the others judged again without it, until the rest
fit. Two points cannot tell which of them is wrong, so three are needed for one to
be named. CALIBRATE also lays the frame again without the cubes named. A point's
miss without it (`left`) is the whole of it, across too: how far the arm would land
from the cube, had the point not been taken.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from maxarm import raw
from maxarm.mesh import Mesh, Row

from applib.camera.base import FoundBase, find_base
from applib.camera.fit import HANDEDNESS, CameraFit
from applib.camera.model import least_squares
from applib.camera.sighting import Sighting, in_frame, on_desk, place, start

# How far each thing is trusted, mm or degrees (see above).
BASE_TURN_DEG = 0.5             # the base's turn by its edges (two frames, one camera: 0.3 deg apart)
SEEN_MM = 0.4                   # a cube's center by its edges (~0.3), the cube centered under the cup (0.25)
PULSE_DEG = 0.24                # one base servo pulse: the servos stop 1-2 short
FAR_MM, MIDDLE_MM, FULL_MM = 1.5, 200.0, 265.0   # the arm off by up to FAR_MM at FULL_MM, from MIDDLE_MM out
REACH_SPREAD_MM = 0.75          # the reach landing to landing, used only without the base
HUBER = 2.0                     # a misfit beyond this many of its spreads counts less
# The points' own axis is reported only with bearings this far apart: closer, their lines meet anywhere.
AXIS_SPREAD_DEG = 30.0
# A point whose reach is further than this from what its neighbors' mesh says is a recording gone wrong (good
# points came within 3.6 mm; a cube recorded wrong, 12.7).
DOES_NOT_FIT_MM = 5.0
STEPS = np.array([0.01, 0.01, 1e-5])


@dataclass(frozen=True)
class TiePoint:
    id: int
    raw: Tuple[float, float, float]     # the firmware's position over the cube
    seen: Tuple[float, float]           # the cube's center on the desk as the camera sees it, the cubes' desk mm

    @property
    def reading(self) -> np.ndarray:
        """The formula's true x, y: where the arm, by the formula alone, says the cup was."""
        return np.array(raw.formula_true(self.raw)[:2])


@dataclass
class Tie:
    axis: Tuple[float, float]           # the arm's axis, the cubes' desk mm
    turn_deg: float                     # the arm's x, y turned this much in the cubes' desk space
    base: Optional[FoundBase]
    rows: List[Row]                     # the mesh's: the points that fit
    report: dict = field(default_factory=dict)
    misfits: Dict[int, float] = field(default_factory=dict)        # point id: mm from where its neighbors put it
    left: Dict[int, Optional[np.ndarray]] = field(default_factory=dict)   # point id: its reading without it, true

    @property
    def lay(self) -> np.ndarray:
        """4x4: the arm's desk (x, y, z) to the cubes' desk space."""
        turn = math.radians(self.turn_deg)
        lay = np.eye(4)
        lay[:2, :2] = [[math.cos(turn), -math.sin(turn)], [math.sin(turn), math.cos(turn)]]
        lay[:2, 3] = self.axis
        return lay

    def to_arm(self, seen: Sequence[float]) -> np.ndarray:
        """A place in the cubes' desk space, about the arm's axis and turn: the arm's true x, y."""
        return np.linalg.solve(self.lay[:2, :2], np.asarray(seen, float) - self.axis)

    def from_arm(self, true_xy: Sequence[float]) -> np.ndarray:
        return self.lay[:2, :2] @ np.asarray(true_xy, float) + self.axis

    def mesh(self, made: str = "") -> Mesh:
        return Mesh(self.rows, made)

    def to_json(self) -> dict:
        return {"axis": [round(value, 3) for value in self.axis], "turn_deg": round(self.turn_deg, 4),
                "base": self.base.to_json() if self.base else None, "report": self.report}


def tie_points(alone: CameraFit, points) -> List[TiePoint]:
    """The points -- RefinePoints with their raw readings -- as the tie takes them: each cube where the cubes' camera
    sees it (its 40 mm cube on its edges, or the pixel under it for a point from before the edges were kept)."""
    return [TiePoint(point.id, tuple(point.raw), tuple(seen_desk(alone, point.sighting))) for point in points]


def seen_desk(alone: CameraFit, sighting: Sighting) -> np.ndarray:
    """The cube's center on the desk as the camera sees it, the cubes' desk mm."""
    matrix = in_frame(alone.true_to_pixel, alone.frame_size, sighting)
    true = on_desk(matrix, sighting) if sighting.edges is None else place(matrix, sighting,
                                                                           start(matrix, sighting)[:3])[:2]
    return np.linalg.solve(alone.desk_to_true, true)


def fit_tie(found: Optional[FoundBase], points: Sequence[TiePoint]) -> Tie:
    """CALIBRATE: the axis and turn from the base and the points, laid again without the points that do not fit; the
    mesh of the rest, and the report."""
    if not points:
        raise ValueError("no points to tie the arm to the camera with")
    tie = _laid(Tie(*_fit(found, points), found, []), points)
    if tie.misfits:
        fitting = [point for point in points if point.id not in tie.misfits]
        tie = _laid(Tie(*_fit(found, fitting), found, []), points)
    tie.report = _report(tie, found, points)
    return tie


def kept_tie(anchor: dict, found: Optional[FoundBase], points: Sequence[TiePoint]) -> Tie:
    """SAVE: CALIBRATE's axis and turn (camera.json's `anchor`), the mesh of the points that fit, and the report."""
    if not points:
        raise ValueError("no points to tie the arm to the camera with")
    tie = _laid(Tie(tuple(anchor["axis"]), float(anchor["turn_deg"]), found, []), points)
    tie.report = _report(tie, found, points)
    return tie


def base_of(fit: CameraFit, frame: Optional[np.ndarray]) -> Optional[FoundBase]:
    """The base as CALIBRATE found it, in `fit.cube_fit()`'s desk space: kept in camera.json, or for a camera.json
    from before, found again in CALIBRATE's `frame`. None without either."""
    if fit.anchor and fit.anchor.get("base"):
        return FoundBase.from_json(fit.anchor["base"])
    return find_base(frame, fit.cube_fit().desk_to_pixel) if frame is not None else None


def anchored(alone: CameraFit, tie: Tie) -> CameraFit:
    """`alone` -- the cubes' camera, the arm laid on it by their readings -- with the arm laid by `tie` instead."""
    lift = np.eye(4)
    lift[:2, :2] = HANDEDNESS
    true_to_pixel = alone.true_to_pixel @ lift @ tie.lay @ lift
    stretch = {} if alone.arm_stretch is None else {"arm_stretch": alone.arm_stretch.tolist()}
    return CameraFit(true_to_pixel, HANDEDNESS.copy(), alone.cubes, alone.frame_size, alone.edge_px, alone.edges_used,
                     cubes_alone={"true_to_pixel": alone.true_to_pixel.tolist(),
                                  "desk_to_true": alone.desk_to_true.tolist(), **stretch, "cubes": alone.cubes},
                     arm_stretch=alone.arm_stretch, anchor=tie.to_json())


def to_anchored(true_xy: Sequence[float]) -> np.ndarray:
    """The arm's true x, y in the anchored calibration's true x, y (the board's mirror in it)."""
    return HANDEDNESS @ np.asarray(true_xy, float)


# --- the fit -----------------------------------------------------------------

def _spread_across(reach: float) -> float:
    """How far across a point may be off, mm: its cube seen, one base pulse, and the arm far out."""
    far = FAR_MM * max(0.0, (reach - MIDDLE_MM) / (FULL_MM - MIDDLE_MM))
    return math.sqrt(SEEN_MM ** 2 + (reach * math.radians(PULSE_DEG)) ** 2 + far ** 2)


def _misfits(found: Optional[FoundBase], points: Sequence[TiePoint], numbers: np.ndarray, *, base: bool,
             reach: bool) -> np.ndarray:
    """Every misfit in its own spread: the base's axis and turn, each point across, and, if asked, out."""
    axis, turn = numbers[:2], numbers[2]
    c, s = math.cos(turn), math.sin(turn)
    parts = []
    if base and found is not None:
        parts += [(axis[0] - found.x) / found.sure_mm, (axis[1] - found.y) / found.sure_mm,
                  (math.degrees(turn) - found.turn_deg) / BASE_TURN_DEG]
    for point in points:
        q = point.reading
        laid = np.array([c * q[0] - s * q[1], s * q[0] + c * q[1]])
        out = laid / max(np.linalg.norm(laid), 1e-9)
        across = np.array([-out[1], out[0]])
        miss = np.asarray(point.seen) - axis - laid
        parts.append(miss @ across / _spread_across(float(np.linalg.norm(q))))
        if reach:
            parts.append(miss @ out / math.hypot(SEEN_MM, REACH_SPREAD_MM))
    return np.array(parts)


def _solve(found: Optional[FoundBase], points: Sequence[TiePoint], *, base: bool, reach: bool,
           first: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """The axis and turn (radians) the misfits want, and the normal matrix in the spreads' units."""
    first = np.array([found.x, found.y, math.radians(found.turn_deg)] if found is not None else [0.0, 0.0, 0.0]) \
        if first is None else first
    numbers, _, normal = least_squares(lambda trial: _misfits(found, points, trial, base=base, reach=reach), first,
                                       STEPS, huber=HUBER, iterations=40)
    return numbers, normal


def _fit(found: Optional[FoundBase], points: Sequence[TiePoint]) -> Tuple[Tuple[float, float], float]:
    numbers, _ = _solve(found, points, base=True, reach=found is None)
    return (float(numbers[0]), float(numbers[1])), math.degrees(numbers[2])


def _row(tie: Tie, point: TiePoint) -> Row:
    """The point's correction: the cube about the arm's axis and turn against the formula's reading."""
    reading, seen = point.reading, tie.to_arm(point.seen)
    reach, bearing = float(np.linalg.norm(reading)), math.degrees(math.atan2(reading[0], -reading[1]))
    seen_reach, seen_bearing = float(np.linalg.norm(seen)), math.degrees(math.atan2(seen[0], -seen[1]))
    return Row(round(bearing, 4), round(reach, 3), round(seen_reach - reach, 3),
               round((seen_bearing - bearing + 180.0) % 360.0 - 180.0, 4), point.id)


def _corrected(mesh: Mesh, point: TiePoint) -> np.ndarray:
    """The point's reading through the formula and `mesh`: true x, y."""
    true = raw.formula_true(point.raw)
    off = mesh.at(point.raw, true)
    return np.array([true[0] + off[0], true[1] + off[1]])


def _laid(tie: Tie, points: Sequence[TiePoint]) -> Tie:
    """The tie with its mesh rows, the points that do not fit named and left out, and each point without it."""
    rows = {point.id: _row(tie, point) for point in points}
    fitting = list(points)
    while len(fitting) >= 3:
        misses = {point.id: _miss(tie, point, [rows[other.id] for other in fitting if other is not point])
                  for point in fitting}
        worst = max(misses, key=misses.get)
        if misses[worst] <= DOES_NOT_FIT_MM:
            break
        tie.misfits[worst] = round(misses[worst], 2)
        fitting = [point for point in fitting if point.id != worst]
    tie.rows = [rows[point.id] for point in fitting]
    for point in points:
        others = [rows[other.id] for other in fitting if other.id != point.id]
        tie.left[point.id] = _corrected(Mesh(others), point) if others else None
    return tie


def _miss(tie: Tie, point: TiePoint, others: List[Row]) -> float:
    """How far the point's reach is from what the mesh of `others` says there: its cube further out than the formula
    put the cup, against theirs, mm."""
    row = _row(tie, point)
    out, _ = Mesh(others).value(row.bearing_deg, row.reach_mm)
    return abs(row.further_out_mm - out)


# --- the report --------------------------------------------------------------

def _report(tie: Tie, found: Optional[FoundBase], points: Sequence[TiePoint]) -> dict:
    """How each thing trusted fits the rest: the base against the points, each point against its neighbors, the
    points that do not fit, and which fits worst."""
    report = {"points": len(points), "axis": [round(value, 2) for value in tie.axis],
              "turn_deg": round(tie.turn_deg, 3)}
    misfits = []
    if found is not None:
        off = math.dist(tie.axis, (found.x, found.y))
        turn_off = tie.turn_deg - found.turn_deg
        report["base"] = {"axis_off_mm": round(off, 2), "turn_off_deg": round(turn_off, 3)}
        misfits += [("the base's axis", off / found.sure_mm), ("the base's turn", abs(turn_off) / BASE_TURN_DEG)]
    fitting = [point for point in points if point.id not in tie.misfits]
    rows = []
    for point in points:
        row = _row(tie, point)
        across = math.radians(row.further_round_deg) * (row.reach_mm + row.further_out_mm)
        spread = _spread_across(row.reach_mm)
        rows.append({"id": point.id, "bearing": round(row.bearing_deg, 2), "reach": round(row.reach_mm, 1),
                     "out_mm": round(row.further_out_mm, 2), "across_mm": round(across, 2),
                     "across_spreads": round(abs(across) / spread, 2), "fits": point.id not in tie.misfits})
        if point.id not in tie.misfits:
            misfits.append((f"point {point.id} across", abs(across) / spread))
    report["rows"] = rows
    report["does_not_fit_mm"] = {str(key): value for key, value in tie.misfits.items()}
    bearings = [row.bearing_deg for row in tie.rows]
    if len(fitting) >= 3 and max(bearings) - min(bearings) >= AXIS_SPREAD_DEG:
        report["turns_axis"] = _own_axis(found, fitting, tie, reach=False)
        report["reach_axis"] = _own_axis(found, fitting, tie, reach=True)
    report["left_out_mm"] = {str(point.id): round(float(math.dist(tie.left[point.id], tie.to_arm(point.seen))), 2)
                             for point in points if tie.left.get(point.id) is not None}
    if misfits:
        name, size = max(misfits, key=lambda misfit: misfit[1])
        report["worst"] = {"what": name, "spreads": round(size, 2)}
    return report


def _own_axis(found: Optional[FoundBase], points: Sequence[TiePoint], tie: Tie, *, reach: bool) -> dict:
    """Where the points alone put the arm's axis: by their bearings (the lines they point along), or with their
    reaches trusted too; how far from the tie's, and how sure, mm."""
    first = np.array([*tie.axis, math.radians(tie.turn_deg)])
    numbers, normal = _solve(found, points, base=False, reach=reach, first=first)
    try:
        sure = float(math.sqrt(max(np.linalg.eigvalsh(np.linalg.inv(normal)[:2, :2])[-1], 0.0)))
    except np.linalg.LinAlgError:
        sure = math.inf
    return {"axis": [round(float(numbers[0]), 2), round(float(numbers[1]), 2)],
            "off_mm": round(float(math.dist(numbers[:2], tie.axis)), 2), "sure_mm": round(sure, 2),
            "turn_deg": round(math.degrees(numbers[2]), 3)}
