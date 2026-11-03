"""The points, as the tests page shows them, and the calibration they make (`camera/refine.py`).

The points are CALIBRATE's three cubes, 1, 2 and 3, until forgotten, and the red
cube's from this page. Each is shown with how far the arm's reading for it is
from its cube as the camera sees it, across the desk (XY), and how far from 40
mm tall the camera sees the cube (Z): through the three cubes' calibration,
CALIBRATE's, which lays the arm on its three cubes exactly, and through the one
SAVE CALIBRATION would write, the camera fitted to all the points' pictures and
the arm laid on all of them at once. XY is a point's against the others: 0 for
CALIBRATE's three, and for any three. With no points, SAVE CALIBRATION puts back
the three cubes' own calibration.
"""

import math
from typing import List, Optional

import numpy as np

from applib.camera.fit import CameraFit, stretch_of
from applib.camera.store import RefinePoint
from camera import check
from camera.refine import Off, PointsDoNotFit, Sighting, off_by, refine
from common.worker import Refused

# SAVE CALIBRATION takes this many points at the least -- the arm is laid by six numbers and each point gives two, so
# three fit it exactly and the fourth is the first that can disagree -- and the page asks for this many (the owner's:
# 10 at least).
MIN_POINTS = 4
SUGGESTED_POINTS = 10
# How far off a point is, coloured as on the desk page: green to the first, yellow to the second, red beyond.
OFF_MM = (1.0, 2.0)


def refined(fit: CameraFit, points: List[RefinePoint]) -> CameraFit:
    """The calibration SAVE CALIBRATION writes, or Refused, saying why the points do not make one."""
    if not points and fit.refined:
        return refine(fit, [])
    if len(points) < MIN_POINTS:
        missing = MIN_POINTS - len(points)
        raise Refused(f"record {missing} more point{'s' if missing > 1 else ''} first: {MIN_POINTS} at the least, "
                      f"{SUGGESTED_POINTS} or more spread over the desk to be of use")
    try:
        return refine(fit, [point.sighting for point in points])
    except PointsDoNotFit as reason:
        raise Refused(str(reason)) from None


def describe(fit: CameraFit, points: List[RefinePoint]) -> dict:
    """The points, how far off each is, and what SAVE CALIBRATION would change."""
    try:
        trial, problem = refined(fit, points), None
    except Refused as reason:
        trial, problem = None, str(reason)
    rows = [{"id": point.id, "cube": point.cube, "z": point.z, "taken": point.taken,
             **_row(fit, trial, point.sighting)} for point in points]
    return {"points": rows, "fit": _summary(fit, trial, rows) if trial else None, "problem": problem,
            "refined": fit.refined}


def _row(fit: CameraFit, trial: Optional[CameraFit], point: Sighting) -> dict:
    """Where the arm says, where the saved calibration sees the cube, and how far off it is through CALIBRATE's
    calibration and after SAVE: across the desk, and its top from 40 mm (None for a point without edges)."""
    saved = off_by(fit, point)
    calibrate = saved if fit.cubes_alone is None else off_by(fit.cube_fit(), point)
    after = off_by(trial, point) if trial else None
    seen = saved.seen
    width, height = point.frame_size
    return {"x": point.x, "y": point.y, "seen": [round(float(seen[0]), 2), round(float(seen[1]), 2)],
            "view": [(point.pixel[0] + 0.5) / width, (point.pixel[1] + 0.5) / height],
            "off": {"calibrate": round(calibrate.across, 2), "after": round(after.across, 2) if after else None},
            "height": {"calibrate": _rounded(calibrate), "after": _rounded(after)}}


def _rounded(off: Optional[Off]) -> Optional[float]:
    return round(off.height, 2) if off and off.height is not None else None


def _summary(fit: CameraFit, trial: CameraFit, rows: List[dict]) -> dict:
    """After SAVE: how far off the points are across the desk, and their cubes' tops from 40 mm, typically and at
    worst (None with none); how far it moves the desk, in reach and in view; how the arm measures against the
    camera (the least and most a distance is stretched, and how far off square); and whether it is the cubes' own
    calibration put back."""
    offs = [row["off"]["after"] for row in rows]
    heights = [abs(row["height"]["after"]) for row in rows if row["height"]["after"] is not None]
    typical, worst = check.difference_mm(fit, trial)
    low, high, skew = stretch_of(trial.desk_to_true)
    return {"off_mm": [float(np.median(offs)), max(offs)] if offs else None,
            "height_mm": [float(np.median(heights)), max(heights)] if heights else None,
            "change_mm": [round(typical, 2), round(worst, 2)] if math.isfinite(worst) else None,
            "arm": {"stretch": [round(low, 4), round(high, 4)], "skew_deg": round(skew, 2)},
            "is_cubes_alone": trial.refined is None}
