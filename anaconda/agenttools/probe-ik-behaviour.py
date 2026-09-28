#!/usr/bin/env python3
"""Ask the board's IK what it does at its edges. Zero motion.

`__espmax` is compiled, so the kit's Arduino `_espmax.cpp` has to stand in for
its source. It is a faithful twin (same L0..L4, same algebra), but C and
MicroPython part ways exactly where we care: `acos()` out of domain returns NaN
in C and raises in Python, and the C version *prints* on an invalid angle where
the Python one may raise. Which of those happens decides whether a limit shows
up as `False` from `set_position()` or as a silent clamp.

Nothing is commanded, nothing is written, no name is bound on the board.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "datacollection"))

from maxarm_link import MaxArmLink, ReplError  # noqa: E402

# (label, expression). Each is evaluated bare so we see stdout and tracebacks
# rather than a tidy verdict.
PROBES = [
    ("link constants", "(__espmax.forward(16.8, (120.0, 90.0, 0.0)),)"),
    ("ORIGIN as espmax computes it", "arm.origin"),
    ("ORIGIN via forward+inverse", "arm.position_to_pulses(arm.origin)"),
    # Straight out along -Y at shoulder height: inside reach, then past it.
    ("reachable far (-Y)", "arm.position_to_pulses((0, -280.0, 84.4))"),
    ("past full extension (-Y)", "arm.position_to_pulses((0, -300.0, 84.4))"),
    # Back-right quadrant: the base-angle branch that looks out of servo range.
    ("back-right quadrant", "arm.position_to_pulses((-100.0, 100.0, 100.0))"),
    ("back-left quadrant", "arm.position_to_pulses((100.0, 100.0, 100.0))"),
    ("front-right quadrant", "arm.position_to_pulses((-100.0, -100.0, 100.0))"),
    # High and low, where servo 3 runs out of travel.
    ("high pose", "arm.position_to_pulses((0, -200.0, 204.0))"),
    ("low pose", "arm.position_to_pulses((0, -200.0, 50.0))"),
    # Does deg_to_pulse raise, or only print, when handed a bad angle?
    ("deg_to_pulse(-10 deg)", "__espmax.deg_to_pulse((-10.0, 90.0, 0.0))"),
    ("deg_to_pulse(260 deg)", "__espmax.deg_to_pulse((260.0, 90.0, 0.0))"),
    ("pulse_to_deg(1200)", "__espmax.pulse_to_deg((1200.0, 500.0, 500.0))"),
]


def main() -> int:
    with MaxArmLink() as link:
        for label, expression in PROBES:
            print(f"--- {label}")
            print(f"    expr: {expression}")
            try:
                # run(), not evaluate(): we want the board's own stdout kept
                # next to the value instead of filtered out by the marker.
                out = link.run(f"print(repr({expression}))")
                print(f"    stdout+value: {out!r}")
            except ReplError as exc:
                print(f"    raised: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
