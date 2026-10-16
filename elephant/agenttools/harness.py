#!/usr/bin/env python3
"""A check function and a runner. Deliberately not a test framework.

These run on a laptop with no hardware attached and on whatever Python the
board's owner happens to have, so they depend on nothing outside the standard
library. `check()` prints its own line because the useful output of a run is
the list of what was verified, not a dot.
"""

from typing import Callable, List, Sequence

TestCase = Callable[[], bool]


def check(label: str, is_ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if is_ok else 'FAIL'}  {label}{'  ' + detail if detail else ''}")
    return bool(is_ok)


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
