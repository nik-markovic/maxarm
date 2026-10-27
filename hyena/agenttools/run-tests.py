#!/usr/bin/env python3
"""Every offline check: the library, the tile finder, the letters and the calibration app. No arm, no camera, no motion.

    ../.venv/bin/python agenttools/run-tests.py            # everything
    ../.venv/bin/python agenttools/run-tests.py path       # one suite

Exit code 0 only if every check passes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_arm
import test_calibration
import test_camera
import test_fit
import test_letters
import test_path
import test_spaces
import test_tests_page
import test_tiles
import test_true_z
from harness import run

SUITES = {
    "true-z": test_true_z.TESTS,
    "spaces": test_spaces.TESTS,
    "path": test_path.TESTS,
    "arm": test_arm.TESTS,
    "calibration": test_calibration.TESTS,
    "fit": test_fit.TESTS,
    "camera": test_camera.TESTS,
    "tests-page": test_tests_page.TESTS,
    "tiles": test_tiles.TESTS,
    "letters": test_letters.TESTS,
}


def main() -> int:
    wanted = sys.argv[1:] or list(SUITES)
    unknown = [name for name in wanted if name not in SUITES]
    if unknown:
        print(f"unknown suite(s): {', '.join(unknown)}; have {', '.join(SUITES)}")
        return 2
    return max(run(SUITES[name], name) for name in wanted)


if __name__ == "__main__":
    sys.exit(main())
