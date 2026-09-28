#!/usr/bin/env python3
"""Run every offline check for condor's library. No hardware, no motion.

    ./run-tests.py            # everything
    ./run-tests.py motion     # one group

Exit code is 0 only if every check passes, so this is safe to put in a hook.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_arm
import test_example
import test_geometry
import test_motion
from harness import run

SUITES = {
    "geometry": test_geometry.TESTS,
    "motion": test_motion.TESTS,
    "arm": test_arm.TESTS,
    "example": test_example.TESTS,
}

# `test-scene.py` covers `calibration/arm.py` and is run on its own -- its name
# has a hyphen in it, so it cannot be imported the way these are.


def main() -> int:
    wanted = sys.argv[1:] or list(SUITES)
    unknown = [name for name in wanted if name not in SUITES]
    if unknown:
        print(f"unknown suite(s): {', '.join(unknown)}; have {', '.join(SUITES)}")
        return 2
    return max(run(SUITES[name], name) for name in wanted)


if __name__ == "__main__":
    sys.exit(main())
