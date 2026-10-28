"""Which desk points a jump from the park pose cannot get to, as the pages' moves plan it: offline arithmetic.

    ../.venv/bin/python agenttools/scan-jumps.py          # through config/, every 10 mm
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from maxarm import MaxArm, Space                      # noqa: E402
from maxarm.arm import home_position                  # noqa: E402
from maxarm.path import Refused                       # noqa: E402

HOVER_MM, STEP_MM = 54.0, 5.0
arm = MaxArm(config_dir=ROOT / "config")
arm.use_coordinates(Space.DESK)
home = arm._spaces.arm.to_desk(home_position())
refused, total = [], 0
for x in range(-270, 271, 10):
    for y in range(-270, 271, 10):
        if not arm.can_reach(x, y, HOVER_MM):
            continue
        total += 1
        height = max(home[2], HOVER_MM)
        while height > HOVER_MM and not arm.can_reach(x, y, height):
            height = max(height - STEP_MM, HOVER_MM)
        try:
            arm._planner.plan(home, (x, y, height), None)
        except Refused as reason:
            refused.append((x, y, round(height), math.degrees(math.atan2(x, -y)), str(reason)))
print(f"{len(refused)} of {total} points in reach at {HOVER_MM:.0f} mm refused from the park pose")
for x, y, height, bearing, reason in refused:
    print(f"  ({x:4d}, {y:4d}) at {height}, bearing {bearing:+.0f} deg: {reason}")
