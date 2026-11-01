#!/usr/bin/env python3
"""The app's own handlers once through, on drawn frames and the fake arm (test_tests_page's harness): CALIBRATE,
four red points, the Fine Tune table, SAVE. Only that each runs and what it says; nothing from the fake arm counts.

    .venv/bin/python impala/agenttools/height/app-flow.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]

import test_tests_page as page                                                 # noqa: E402


def main():
    with page.quick_app() as (post, state, board, camera, config, server):
        page.calibrate_by_eye(post, camera, config)
        print("CALIBRATE:", state()["camera"].get("calibration", {}) if isinstance(state()["camera"], dict) else "")
        for reply, spot in page.record_points(post, state, camera, page.REFINE_SPOTS[:4]):
            print("RECORD POINT at", spot, "->", reply.get("status"), reply.get("detail", "")[:160])
        tests = state().get("tests") or {}
        refine = tests.get("refine") or {}
        print("table:", [(row["id"], row["off"], row["height"]) for row in refine.get("points", [])])
        print("fit:", refine.get("fit"), refine.get("problem"))
        saved = post("tests/save")
        print("SAVE ->", saved.get("status"), saved.get("detail"))


if __name__ == "__main__":
    main()
