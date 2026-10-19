"""The desk, from the points where the cup touched it: a plane, and the arm's elbow offset.

At each point the cup was lowered onto the desk by hand, and a point is where
the servos then said it was. The desk is the plane z = tilt_x * x + tilt_y * y
+ offset through them: exactly, through three; the least-squares fit through
more, and then how far each point sits off it says how far it can be trusted.

The board's x, y, z are not quite where the cup is. On the owner's arm the
elbow servo sits 4.4 degrees off the zero the firmware assumes, which makes a
flat desk read 8 mm higher near the base than at full reach -- no plane fits
that (STATUS-falcon 5.6). So when the points reach far enough in and out to
tell the two apart, the elbow's offset is fitted too: the plane is then fitted
where the cup really is, and the offset saved beside it for the library.

Desk space is that true space with the plane taken off z: x and y the cup's
own, z the height above the desk.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from maxarm import geometry
from maxarm.geometry import Position

# The owner's minimums for a calibration worth saving (PILOT-falcon).
X_SPAN_MIN_MM = 200.0
Y_SPAN_MIN_MM = 150.0
# Points in a row fit any tilt about that row. This is the RMS distance from
# the line through them that SAVE needs: a sixth of the Y span asked for.
ACROSS_SPREAD_MIN_MM = 25.0
# Off the fitted plane by more than the first, a point is shown yellow; by
# more than the second, red (PILOT-falcon).
DEVIATION_LIMITS_MM = (1.0, 2.0)
# An elbow offset changes the desk's height with reach, a tilt with direction:
# telling them apart takes this many points, spread this far in and out.
ELBOW_POINTS_MIN = 6
ELBOW_REACH_SPREAD_MIN_MM = 80.0
# How far from the firmware's zero the elbow is looked for, and how finely.
ELBOW_SEARCH_DEG = 10.0
ELBOW_RESOLUTION_DEG = 0.01


@dataclass(frozen=True)
class Plane:
    tilt_x: float            # mm of z per mm of x
    tilt_y: float            # mm of z per mm of y
    offset: float            # z at x = y = 0

    def z_at(self, x: float, y: float) -> float:
        return self.tilt_x * x + self.tilt_y * y + self.offset

    @property
    def tilt_deg(self) -> float:
        return math.degrees(math.atan(math.hypot(self.tilt_x, self.tilt_y)))

    def desk_to_arm(self) -> List[List[float]]:
        """Desk (x, y, height, 1) to arm (x, y, z, 1): x and y as they are, the plane added to z."""
        return [[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [self.tilt_x, self.tilt_y, 1.0, self.offset],
                [0.0, 0.0, 0.0, 1.0]]


@dataclass(frozen=True)
class Desk:
    """The plane, where the cup really is, and the elbow offset that says where that is."""

    plane: Plane
    elbow_deg: float

    @property
    def joint_offsets(self) -> Tuple[float, float, float]:
        return (0.0, 0.0, self.elbow_deg)

    def height_of(self, arm_point: Position) -> float:
        """How far above the desk the cup is when the board is sent `arm_point`."""
        x, y, z = geometry.true_position(arm_point, self.joint_offsets)
        return z - self.plane.z_at(x, y)


def fit(points: Sequence[Position]) -> Optional[Desk]:
    """The desk through the points; None for fewer than three, or for three or more in a row."""
    if len(points) < 3 or across_spread(points) < 1.0:
        return None
    elbow = _best_elbow(points) if can_fit_elbow(points) else 0.0
    return Desk(fit_plane(_true(points, elbow)), elbow)


def can_fit_elbow(points: Sequence[Position]) -> bool:
    reach = [math.hypot(x, y) for x, y, _ in points]
    return len(points) >= ELBOW_POINTS_MIN and max(reach) - min(reach) >= ELBOW_REACH_SPREAD_MIN_MM


def fit_plane(points: Sequence[Position]) -> Plane:
    """The least-squares plane through the points, as they are."""
    xyz = np.asarray(points, dtype=float)
    design = np.column_stack((xyz[:, 0], xyz[:, 1], np.ones(len(xyz))))
    (tilt_x, tilt_y, offset), *_ = np.linalg.lstsq(design, xyz[:, 2], rcond=None)
    return Plane(float(tilt_x), float(tilt_y), float(offset))


def deviations(points: Sequence[Position], desk: Desk) -> List[float]:
    """How far above the desk each point is: + where the cup touched higher than the desk says."""
    return [desk.height_of(point) for point in points]


def rms(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _best_elbow(points: Sequence[Position]) -> float:
    """The elbow offset that lets one plane fit the points best: coarse to fine, down to the resolution."""
    best, step = 0.0, 1.0
    while step >= ELBOW_RESOLUTION_DEG:
        candidates = [best + step * index for index in range(-10, 11)]
        candidates = [value for value in candidates if abs(value) <= ELBOW_SEARCH_DEG]
        best = min(candidates, key=lambda elbow: _misfit(points, elbow))
        step /= 10.0
    return round(best, 2)


def _misfit(points: Sequence[Position], elbow: float) -> float:
    true = _true(points, elbow)
    plane = fit_plane(true)
    return rms([z - plane.z_at(x, y) for x, y, z in true])


def _true(points: Sequence[Position], elbow: float) -> List[Position]:
    return [geometry.true_position(point, (0.0, 0.0, elbow)) for point in points]


def across_spread(points: Sequence[Position]) -> float:
    """The RMS distance of the points from the straight line that fits them best: 0 when in a row."""
    xy = np.asarray(points, dtype=float)[:, :2]
    centred = xy - xy.mean(axis=0)
    smallest = np.linalg.eigvalsh(centred.T @ centred / len(xy))[0]
    return float(math.sqrt(max(smallest, 0.0)))


def problem_with(points: Sequence[Position]) -> Optional[str]:
    """Why these points cannot be saved as the desk, or None if they can."""
    if len(points) < 3:
        return f"SAVE needs at least 3 points; there {'is' if len(points) == 1 else 'are'} {len(points)}"
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    if max(xs) - min(xs) < X_SPAN_MIN_MM:
        return (f"the points span {max(xs) - min(xs):.0f} mm in X; "
                f"they must span at least {X_SPAN_MIN_MM:.0f}")
    if max(ys) - min(ys) < Y_SPAN_MIN_MM:
        return (f"the points span {max(ys) - min(ys):.0f} mm in Y; "
                f"they must span at least {Y_SPAN_MIN_MM:.0f}")
    spread = across_spread(points)
    if spread < ACROSS_SPREAD_MIN_MM:
        return (f"the points lie nearly in a row ({spread:.0f} mm either side of it), which fits any "
                f"tilt about that row; add one well off to the side")
    return None
