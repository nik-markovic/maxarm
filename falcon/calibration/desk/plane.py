"""The desk, from the points where the cup touched it: a plane.

At each point the cup was lowered onto the desk by hand, and a point is where
the servos then said it was, in true z: the library corrects the firmware's
heights, which made a flat desk read 8 mm higher near the base than at full
reach. The desk is the plane z = tilt_x * x + tilt_y * y + offset through
them: exactly, through three; the least-squares fit through more, and then how
far each point sits off it says how far it can be trusted.

Desk space is arm space with the plane taken off z: z the height above the desk.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

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


@dataclass(frozen=True)
class Plane:
    tilt_x: float            # mm of z per mm of x
    tilt_y: float            # mm of z per mm of y
    offset: float            # z at x = y = 0

    def z_at(self, x: float, y: float) -> float:
        return self.tilt_x * x + self.tilt_y * y + self.offset

    def height_of(self, arm_point: Position) -> float:
        """How far above the desk the cup is at `arm_point`."""
        return arm_point[2] - self.z_at(arm_point[0], arm_point[1])

    @property
    def tilt_deg(self) -> float:
        return math.degrees(math.atan(math.hypot(self.tilt_x, self.tilt_y)))

    def desk_to_arm(self) -> List[List[float]]:
        """Desk (x, y, height, 1) to arm (x, y, z, 1): x and y as they are, the plane added to z."""
        return [[1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [self.tilt_x, self.tilt_y, 1.0, self.offset],
                [0.0, 0.0, 0.0, 1.0]]


def fit(points: Sequence[Position]) -> Optional[Plane]:
    """The desk through the points; None for fewer than three, or for three or more in a row."""
    if len(points) < 3 or across_spread(points) < 1.0:
        return None
    return fit_plane(points)


def fit_plane(points: Sequence[Position]) -> Plane:
    """The least-squares plane through the points, as they are."""
    xyz = np.asarray(points, dtype=float)
    design = np.column_stack((xyz[:, 0], xyz[:, 1], np.ones(len(xyz))))
    (tilt_x, tilt_y, offset), *_ = np.linalg.lstsq(design, xyz[:, 2], rcond=None)
    return Plane(float(tilt_x), float(tilt_y), float(offset))


def deviations(points: Sequence[Position], desk: Plane) -> List[float]:
    """How far above the desk each point is: + where the cup touched higher than the desk says."""
    return [desk.height_of(point) for point in points]


def rms(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


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
