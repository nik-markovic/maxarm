#!/usr/bin/env python3
"""SAVE on the owner's real points: CALIBRATE's three cubes of 30 Oct 10:08 and the three of 09:58 (the blue one
recorded before the blue cube was moved, 16 mm: STATUS §14.4), all with their edges; then without that one.
Through CALIBRATE again on the 10:08 frame (the new code). Nothing is written.

    .venv/bin/python impala/agenttools/height/real-six.py [config-dir]
"""

import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera import check, store                                     # noqa: E402
from camera.fit import fit_camera, stretch_of                       # noqa: E402
from camera.refine import off_by, refine                            # noqa: E402


def main():
    config = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "config"
    arm = {colour: (cube["x"], cube["y"])
           for colour, cube in json.loads((config / "camera-points.json").read_text())["cubes"].items()}
    fit = fit_camera(cv2.imread(str(config / "camera-frame.png")), arm)
    now = store.cube_refine_points(fit, {}, "10:08")
    before = [store.RefinePoint(**{**point, "id": point["id"] + 3}) for point in
              json.loads((config / "camera-refine-points-previous.json").read_text())["points"]]
    names = {point.id: f"{point.cube} {'10:08' if point.id <= 3 else '09:58'}" for point in now + before}
    for label, points in (("all six", now + before), ("without blue 09:58", now + before[1:])):
        started = time.monotonic()
        saved = refine(fit, [point.sighting for point in points])
        low, high, skew = stretch_of(saved.desk_to_true)
        print(f"SAVE, {label} ({time.monotonic() - started:.1f} s): focal {saved.focal:.1f} (CALIBRATE's "
              f"{fit.focal:.1f}); the desk {check.difference_mm(fit, saved)} mm from CALIBRATE's; the arm stretched "
              f"{low:.4f} to {high:.4f}, {skew:.2f} deg")
        for point in points:
            off = off_by(saved, point.sighting)
            calibrate = off_by(fit, point.sighting)
            print(f"  {point.id}: {names[point.id]:12s} at CALIBRATE XY {calibrate.across:6.2f}  after SAVE XY "
                  f"{off.across:6.2f}  Z {off.height:+.2f}")


if __name__ == "__main__":
    main()
