"""The points, as the tests page shows them, and the calibration they make (`camera/refine.py`).

The points are CALIBRATE's three cubes, 1, 2 and 3, until forgotten, and the red
cube's from this page. Each is shown with how far the arm's reading for it is
from its cube as the camera sees it, across the desk (XY), and how far from 40
mm tall the camera sees the cube (Z), through the saved calibration -- camera.json
and the arm's correction as the library converts with it: "off now". And without
it: its reading through the correction of the other points -- their pattern and
mesh -- against its cube (`camera/anchor.py`), which the page draws beside the cube. A point the others put
further than `anchor.DOES_NOT_FIT_MM` from its cube does not fit: SAVE leaves it out
of the mesh and names it. With no points, SAVE CALIBRATION puts back the three
cubes' own calibration.
"""

import math
from typing import List, Optional, Tuple

import numpy as np

from applib.camera.fit import CameraFit, stretch_of
from applib.camera.model import project
from applib.camera.sighting import in_frame
from applib.camera.store import RefinePoint
from applib.worker import Refused
from calibration.camera import check
from calibration.camera.anchor import Tie
from calibration.camera.refine import Off, PointsDoNotFit, Sighting, off_by, refine_and_leave_out

# SAVE CALIBRATION takes this many points at the least -- each is judged by the mesh of the others, and three are
# needed to tell which of two disagreeing points is wrong -- and the page asks for this many: about 6 over the
# tile area beside CALIBRATE's three.
MIN_POINTS = 4
SUGGESTED_POINTS = 10
# How far off a point is, green to the first, yellow to the second, red beyond: a reading carries about 0.7-0.9 mm, a
# point left out of the arm's laying about 0.9 typically and under 2, checked with the cup on grid paper.
OFF_MM = (1.5, 2.5)


def refined(fit: CameraFit, points: List[RefinePoint], frame: Optional[np.ndarray]) -> Tuple[CameraFit, Optional[Tie]]:
    """The calibration SAVE CALIBRATION writes and the tie whose mesh is the arm's correction (None: no correction),
    or Refused, saying why the points do not make one."""
    trial, tie, _ = _refined(fit, points, frame)
    return trial, tie


def _refined(fit: CameraFit, points: List[RefinePoint], frame: Optional[np.ndarray]
             ) -> Tuple[CameraFit, Optional[Tie], List[Optional[np.ndarray]]]:
    """`refined()`, and each point's reading as the mesh of the other points puts it."""
    if not points and (fit.refined or fit.anchor):
        return refine_and_leave_out(fit, [], frame)
    if len(points) < MIN_POINTS:
        missing = MIN_POINTS - len(points)
        raise Refused(f"record {missing} more point{'s' if missing > 1 else ''} first: {MIN_POINTS} at the least, "
                      f"{SUGGESTED_POINTS} or more spread over the desk to be of use")
    try:
        return refine_and_leave_out(fit, points, frame)
    except PointsDoNotFit as reason:
        raise Refused(str(reason)) from None


def describe(fit: CameraFit, points: List[RefinePoint], frame: Optional[np.ndarray]) -> dict:
    """The points, how far off each is, and what SAVE CALIBRATION would make."""
    try:
        (trial, tie, put), problem = _refined(fit, points, frame), None
    except Refused as reason:
        (trial, tie, put), problem = (None, None, []), str(reason)
    put = put if trial and trial.refined else [None] * len(points)
    misfits = tie.misfits if tie else {}
    rows = [{"id": point.id, "cube": point.cube, "z": point.z, "taken": point.taken, "fits": point.id not in misfits,
             **_row(fit, trial, point.sighting, others)} for point, others in zip(points, put)]
    pattern = tie.mesh().describe_pattern() if tie and tie.rows else None
    return {"points": rows, "fit": _summary(fit, trial, rows, misfits, pattern) if trial else None, "problem": problem,
            "refined": fit.refined}


def _row(fit: CameraFit, trial: Optional[CameraFit], point: Sighting, others: Optional[np.ndarray]) -> dict:
    """Where the arm says, where the saved calibration sees the cube, and how far off it is now -- the reading as the
    library converts it, against the cube through the saved camera.json -- and without it: from where the mesh of the
    `others` puts its reading (true x, y through SAVE's calibration); its top from 40 mm (None for a point without
    edges). `put` is that place in the point's frame, as shares of its width and height, for the page to draw."""
    now = off_by(fit, point)
    width, height = point.frame_size
    without, put = None, None
    if others is not None:
        without = float(math.dist(others, trial.desk_to_true @ off_by(trial, point).seen))
        pixel = project(in_frame(trial.true_to_pixel, trial.frame_size, point), np.array([[*others, 0.0]]))[0]
        put = [round(float((pixel[0] + 0.5) / width), 5), round(float((pixel[1] + 0.5) / height), 5)]
    return {"x": point.x, "y": point.y, "seen": [round(float(now.seen[0]), 2), round(float(now.seen[1]), 2)],
            "view": [(point.pixel[0] + 0.5) / width, (point.pixel[1] + 0.5) / height], "put": put,
            "off": {"now": round(now.across, 2), "without": round(without, 2) if without is not None else None},
            "height": {"now": _rounded(now)}}


def _rounded(off: Optional[Off]) -> Optional[float]:
    return round(off.height, 2) if off and off.height is not None else None


def _summary(fit: CameraFit, trial: CameraFit, rows: List[dict], misfits: dict, pattern: Optional[str]) -> dict:
    """After SAVE: how far each point that fits is from where the others put it, typically and at worst (None with
    none), and the points that do not fit; the pattern they share over the whole desk, in words; their cubes' tops
    from 40 mm; how far it moves the desk, in reach and in view; how the arm measures against the camera (the least
    and most a distance is stretched, and how far off square); and whether it is the cubes' own calibration put
    back."""
    offs = [row["off"]["without"] for row in rows if row["off"]["without"] is not None and row["fits"]]
    heights = [abs(row["height"]["now"]) for row in rows if row["height"]["now"] is not None]
    typical, worst = check.difference_mm(fit, trial)
    low, high, skew = stretch_of(trial.measured_stretch)
    return {"off_mm": [float(np.median(offs)), max(offs)] if offs else None,
            "does_not_fit": [{"id": point, "mm": miss} for point, miss in misfits.items()],
            "pattern": pattern,
            "height_mm": [float(np.median(heights)), max(heights)] if heights else None,
            "change_mm": [round(typical, 2), round(worst, 2)] if math.isfinite(worst) else None,
            "arm": {"stretch": [round(low, 4), round(high, 4)], "skew_deg": round(skew, 2)},
            "is_cubes_alone": trial.refined is None}
