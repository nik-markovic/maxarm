#!/usr/bin/env python3
"""Fine Tune's SAVE CALIBRATION as the app runs it (tests/api.py, TestsSession._save), on a copy of the owner's
calibration made by the new CALIBRATE: the 10:08 frame and cubes, and the 09:58 green and red as points 4 and 5.
Writes only to the scratch directory given.

    .venv/bin/python impala/agenttools/height/save-flow.py SCRATCH-DIR
"""

import json
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]

from camera import store                                   # noqa: E402
from camera.fit import fit_camera                          # noqa: E402
from tests import refine as table                          # noqa: E402
from tests.api import TestsSession                         # noqa: E402


def main():
    config, scratch = ROOT / "config", Path(sys.argv[1])
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    frame = cv2.imread(str(config / "camera-frame.png"))
    arm = {colour: (cube["x"], cube["y"])
           for colour, cube in json.loads((config / "camera-points.json").read_text())["cubes"].items()}
    fit = fit_camera(frame, arm)
    before = [store.RefinePoint(**point) for point in
              json.loads((config / "camera-refine-points-previous.json").read_text())["points"]][1:]
    points = store.cube_refine_points(fit, {}, "30 Oct 10:08") + [
        store.RefinePoint(**{**vars(point), "id": index + 4, "cube": None}) for index, point in enumerate(before)]
    store.save_refine_points(scratch, points)
    store.save_frame(scratch, frame)
    store.save_camera(scratch, fit)
    print("camera page:", store.describe_fit(fit))
    session = TestsSession(None, None, scratch)
    loaded, problem = session._calibration()
    print("loaded:", len(session.points), "points;", problem)
    described = table.describe(loaded, session.points)
    print("fit line:", described["fit"], described["problem"])
    print("SAVE:", session._save(None))
    saved = store.load_camera(scratch)
    print("camera.json now:", store.describe_fit(saved))
    print("table after:", [(row["id"], row["off"], row["height"]) for row in table.describe(saved, session.points)["points"]])


if __name__ == "__main__":
    main()
