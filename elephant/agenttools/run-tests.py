#!/usr/bin/env python3
"""Every offline check for elephant's library. No hardware, no motion.

    ../.venv/bin/python agenttools/run-tests.py            # everything
    ../.venv/bin/python agenttools/run-tests.py path       # one suite

Exit code 0 only if every check passes. `training/tiles/check-reader.py` covers
the tile finder on its own baseline scenes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_arm
import test_example
import test_kinematics
import test_path
import test_spaces
from harness import run

SUITES = {
    "kinematics": test_kinematics.TESTS,
    "spaces": test_spaces.TESTS,
    "path": test_path.TESTS,
    "arm": test_arm.TESTS,
    "example": test_example.TESTS,
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
