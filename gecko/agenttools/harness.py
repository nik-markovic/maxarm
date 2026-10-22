#!/usr/bin/env python3
"""A check function and a runner. Deliberately not a test framework.

These run on a laptop with no hardware attached and on whatever Python the
board's owner happens to have, so they depend on nothing outside the standard
library. `check()` prints its own line because the useful output of a run is
the list of what was verified, not a dot.
"""

import functools
from pathlib import Path
from typing import Callable, List, Sequence

from maxarm import raw

TestCase = Callable[[], bool]

# The calibration the library checks read: a fixed copy of what elephant shipped.
# config/ is the live one, and the desk calibration rewrites it.
BASELINE = Path(__file__).resolve().parent.parent / "training" / "tiles" / "baseline"


def check(label: str, is_ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if is_ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(is_ok)


def with_raw_heights(test: TestCase) -> TestCase:
    """`test` with the library's height correction off, as `raw.debug_use_raw()` turns it off.

    For checks written in the firmware's own heights: what they check -- plans,
    timings, the pages' moves -- does not depend on it. `test_true_z.py` checks
    the correction.
    """
    @functools.wraps(test)
    def in_raw_heights() -> bool:
        raw.debug_use_raw(True)
        try:
            return test()
        finally:
            raw.debug_use_raw(False)
    return in_raw_heights


def run(tests: Sequence[TestCase], title: str = "") -> int:
    """Run every test, print a summary, and return a process exit code."""
    if title:
        print(f"\n=== {title} ===")
    failures: List[str] = []
    for test in tests:
        try:
            if not test():
                failures.append(test.__name__)
        except Exception as error:               # a crash is a failure, not a stop
            print(f"  FAIL  {test.__name__} raised {error!r}")
            failures.append(test.__name__)
    if failures:
        print(f"\n{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print(f"\nall {len(tests)} groups pass")
    return 0
