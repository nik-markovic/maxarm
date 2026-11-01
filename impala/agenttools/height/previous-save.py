#!/usr/bin/env python3
"""The new SAVE on a set-aside calibration: CALIBRATE's frame fitted again, its three cubes as points with their
edges, the Fine Tune points as they were kept (the pixel alone: no edges then).

    .venv/bin/python impala/agenttools/height/previous-save.py [config-dir] [suffix]

Nothing is written. Prints each point through CALIBRATE's camera, the saved one, and the new SAVE.
"""

import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera import check, store                     # noqa: E402
from camera.fit import CameraFit, fit_camera        # noqa: E402
from camera.refine import off_by, refine            # noqa: E402


def main():
    config = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "config"
    suffix = sys.argv[2] if len(sys.argv) > 2 else "-previous"
    frame = cv2.imread(str(config / f"camera-frame{suffix}.png"))
    saved = CameraFit.from_json(json.loads((config / f"camera{suffix}.json").read_text()))
    arm = {colour: (cube["x"], cube["y"])
           for colour, cube in json.loads((config / f"camera-points{suffix}.json").read_text())["cubes"].items()}
    calibrated = fit_camera(frame, arm)
    print(f"CALIBRATE again: {check.difference_mm(saved.cube_fit(), calibrated)} mm from the file's cubes alone")
    stored = [store.RefinePoint(**point)
              for point in json.loads((config / f"camera-refine-points{suffix}.json").read_text())["points"]]
    cubes = {point.cube: point for point in store.cube_refine_points(calibrated, {}, "now")}
    points = [cubes[point.cube] if point.cube else point for point in stored]
    started = time.monotonic()
    new = refine(calibrated, [point.sighting for point in points])
    print(f"new SAVE {time.monotonic() - started:.1f} s: {new.refined}; desk from the saved camera.json "
          f"{check.difference_mm(saved, new)} mm, focal {new.focal:.1f} (saved {saved.focal:.1f})")
    print("  id  cube   CALIBRATE across/top   saved across/top   new SAVE across/top")
    for point in points:
        offs = [off_by(fit, point.sighting) for fit in (calibrated, saved, new)]
        cells = "   ".join(f"{off.across:5.2f} {'' if off.height is None else f'{off.height:+.2f}':>6s}"
                           for off in offs)
        print(f"  {point.id:3d} {point.cube or '':6s} {cells}")


if __name__ == "__main__":
    main()
