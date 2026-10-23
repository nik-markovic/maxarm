"""The red cube's points, as the tests page shows them, and the calibration they make (`camera/refine.py`).

Each point, and each of the three cubes the camera was fitted to, is shown with
how far the camera puts the cube's centre from where the arm says the cup was:
through the saved calibration, and through the one SAVE CALIBRATION would write.
With no points, SAVE CALIBRATION puts back the three cubes' own calibration.
"""

import math
from typing import List, Optional

import numpy as np

from camera import check, store
from camera.fit import CameraFit
from camera.refine import RefineRefused, Sighting, cube_sightings, off_mm, refine, seen_desk
from camera.store import RefinePoint
from common.worker import Refused

# SAVE CALIBRATION takes this many points at the least; the page asks for this many (the owner's: 10 at least).
MIN_POINTS = 3
SUGGESTED_POINTS = 10
# How far off a point is, coloured as on the desk page: green to the first, yellow to the second, red beyond.
OFF_MM = (1.0, 2.0)


def refined(fit: CameraFit, points: List[RefinePoint]) -> CameraFit:
    """The calibration SAVE CALIBRATION writes, or Refused, saying why the points do not make one."""
    if not points and fit.refined:
        return fit.cube_fit()
    if len(points) < MIN_POINTS:
        missing = MIN_POINTS - len(points)
        raise Refused(f"record {missing} more point{'s' if missing > 1 else ''} first: {MIN_POINTS} at the least, "
                      f"{SUGGESTED_POINTS} or more spread over the desk to be of use")
    try:
        return refine(fit, [point.sighting for point in points])
    except RefineRefused as reason:
        raise Refused(str(reason)) from None


def describe(fit: CameraFit, points: List[RefinePoint]) -> dict:
    """The points and the cubes, how far off each is, and what SAVE CALIBRATION would change."""
    try:
        trial, problem = refined(fit, points), None
    except Refused as reason:
        trial, problem = None, str(reason)
    rows = {
        "points": [{"id": point.id, "z": point.z, "taken": point.taken, **_row(fit, trial, point.sighting)}
                   for point in points],
        "cubes": [{"colour": colour, **_row(fit, trial, sighting)}
                  for colour, sighting in zip(fit.cubes, cube_sightings(fit))],
    }
    return {**rows, "fit": _summary(fit, trial, rows) if trial else None, "problem": problem, "refined": fit.refined}


def _row(fit: CameraFit, trial: Optional[CameraFit], point: Sighting) -> dict:
    """Where the arm says, where the saved calibration sees the cube, and how far off it is now and after SAVE."""
    seen = seen_desk(fit, point)
    width, height = point.frame_size
    return {"x": point.x, "y": point.y, "seen": [round(float(seen[0]), 2), round(float(seen[1]), 2)],
            "view": [(point.pixel[0] + 0.5) / width, (point.pixel[1] + 0.5) / height],
            "off": {"now": round(off_mm(fit, point), 2), "after": round(off_mm(trial, point), 2) if trial else None}}


def _summary(fit: CameraFit, trial: CameraFit, rows: dict) -> dict:
    """After SAVE: how far off the points are, typically and at worst; how far it moves the desk, in reach and in
    view; the arm's error; and whether it is the cubes' own calibration put back."""
    offs = [row["off"]["after"] for row in rows["points"] + rows["cubes"]]
    typical, worst = check.difference_mm(fit, trial)
    return {"off_mm": [float(np.median(offs)), max(offs)],
            "change_mm": [round(typical, 2), round(worst, 2)] if math.isfinite(worst) else None,
            "arm": store.describe_arm(trial), "is_cubes_alone": trial.refined is None}
