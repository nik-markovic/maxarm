#!/usr/bin/env python3
"""The planner: what is refused, and the path round clearance, the arm-base limit and sag.

Arithmetic only, no board. Most checks plan in a desk that is the arm's own
frame (an identity calibration), so the numbers read directly; the sag and
the base checks also run through elephant's calibration.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maxarm import path                                        # noqa: E402
from maxarm.path import Planner, Refused                       # noqa: E402
from maxarm.spaces import Affine, Spaces                       # noqa: E402
from harness import BASELINE, check                            # noqa: E402

FLAT = Planner(Affine(np.eye(4)), is_desk_known=True)
SHIPPED = Spaces(BASELINE)
CALIBRATED = Planner(SHIPPED.arm, is_desk_known=True)


def test_ordinary_move_is_one_leg() -> bool:
    print("an ordinary move:")
    legs = FLAT.plan((150.0, -150.0, 120.0), (200.0, -120.0, 110.0), crossing_height=None)
    guarded = FLAT.plan((150.0, -150.0, 60.0), (-150.0, -150.0, 60.0), crossing_height=40.0)
    return all([
        check("high over the desk it is one leg, straight to the target",
              legs == [(200.0, -120.0, 110.0)], f"{legs}"),
        check("above the crossing height a clearance changes nothing", len(guarded) == 1),
    ])


def test_refusals() -> bool:
    print("what is refused, before anything moves:")
    reasons = {name: FLAT.problem_with(point) for name, point in {
        "good": (100.0, -200.0, 60.0),
        "far": (0.0, -280.0, 60.0),
        "behind": (0.0, 160.0, 60.0),
        "base": (40.0, -40.0, 60.0),
        "sunk": (100.0, -200.0, -10.0),
        "high": (0.0, -160.0, 230.0),
    }.items()}
    loose = Planner(Affine(np.eye(4)), is_desk_known=False).problem_with((100.0, -200.0, -10.0))
    try:
        FLAT.plan((100.0, -200.0, 60.0), (40.0, -40.0, 60.0), None)
        is_plan_refused = False
    except Refused:
        is_plan_refused = True
    return all([
        check("a good target has no problem", reasons["good"] is None, f"{reasons['good']}"),
        check(f"past {path.REACH_MAX_MM:.0f} mm", "reach limit" in (reasons["far"] or ""),
              f"{reasons['far']}"),
        check("behind the base", "behind" in (reasons["behind"] or ""), f"{reasons['behind']}"),
        check("inside the arm-base limit", "arm-base" in (reasons["base"] or ""),
              f"{reasons['base']}"),
        check("10 mm into a calibrated desk", "below the desk" in (reasons["sunk"] or ""),
              f"{reasons['sunk']}"),
        check("but not into an unknown one", loose is None, f"{loose}"),
        check("above the firmware's z clamp", "cannot hold" in (reasons["high"] or ""),
              f"{reasons['high']}"),
        check("and plan() raises it", is_plan_refused),
    ])


def test_clearance() -> bool:
    print("clearance: climb, cross, come down:")
    low = FLAT.plan((150.0, -150.0, 4.0), (-150.0, -150.0, 4.0), crossing_height=23.0)
    down = FLAT.plan((0.0, -163.0, 212.0), (150.0, -150.0, 4.0), crossing_height=23.0)
    up = FLAT.plan((150.0, -150.0, 4.0), (0.0, -163.0, 150.0), crossing_height=23.0)
    straight_down = FLAT.plan((150.0, -150.0, 60.0), (150.5, -150.0, 4.0), crossing_height=23.0)
    return all([
        check("low to low: up, across at the crossing height, down",
              low == [(150.0, -150.0, 23.0), (-150.0, -150.0, 23.0), (-150.0, -150.0, 4.0)],
              f"{low}"),
        check("high to low: across to over the target, then down",
              down == [(150.0, -150.0, 23.0), (150.0, -150.0, 4.0)], f"{down}"),
        check("low to high: up first, then on", up[0] == (150.0, -150.0, 23.0) and len(up) == 2,
              f"{up}"),
        check("half a millimetre sideways is not crossing the desk", len(straight_down) == 1),
    ])


def test_round_the_base() -> bool:
    print("the arm-base limit: by a front corner, at half-way height:")
    out = path.BASE_LIMIT_HALF_MM + path.BASE_CORNER_MARGIN_MM
    across = FLAT.plan((-120.0, -60.0, 40.0), (120.0, -60.0, 80.0), None)
    behind = FLAT.plan((-150.0, 30.0, 60.0), (150.0, 30.0, 60.0), None)
    leaving = FLAT.plan((30.0, -60.0, 100.0), (150.0, -150.0, 100.0), None)
    through_calibration = CALIBRATED.plan(SHIPPED.arm.to_desk((-150.0, 30.0, 90.0)),
                                          SHIPPED.arm.to_desk((150.0, 30.0, 90.0)), None)
    in_arm = [SHIPPED.arm.from_desk(point) for point in through_calibration]
    return all([
        check("a line across the front goes by a corner, z half-way",
              _is_clear(FLAT, (-120.0, -60.0, 40.0), across) and len(across) >= 2
              and any(abs(abs(p[0]) - out) < 1e-6 and abs(p[1] + out) < 1e-6
                      and abs(p[2] - 60.0) < 1e-6 for p in across), f"{_rounded(across)}"),
        check("side to side behind the front edge takes both corners",
              _is_clear(FLAT, (-150.0, 30.0, 60.0), behind)
              and [round(p[0]) for p in behind[:2]] == [-round(out), round(out)],
              f"{_rounded(behind)}"),
        check("leaving the square from inside is allowed, straight out", leaving[-1] == (
            150.0, -150.0, 100.0) and len(leaving) == 1, f"{_rounded(leaving)}"),
        check("through the shipped calibration, the corners are the arm's own",
              _is_clear(CALIBRATED, SHIPPED.arm.to_desk((-150.0, 30.0, 90.0)), through_calibration)
              and any(abs(abs(p[0]) - out) < 1e-6 and abs(p[1] + out) < 1e-6 for p in in_arm),
              f"{_rounded(in_arm)}"),
    ])


def test_no_leg_sags_past_the_margin() -> bool:
    """The real path sags on long moves in or out; the planner halves those legs."""
    print("sag: long moves low over the desk, through the shipped calibration:")
    worst, legs_used, moves = 0.0, 0, 0
    for height in (2.0, 10.0, 23.0, 40.0):
        for r_start in (110.0, 170.0, 265.0):
            for r_end in (110.0, 170.0, 265.0):
                for bearing in (-60.0, 0.0, 60.0):
                    start = _desk_at(r_start, bearing, height)
                    end = _desk_at(r_end, -bearing / 2.0, height)
                    plan = CALIBRATED.plan(start, end, None)
                    cursor = start
                    for point in plan:
                        worst = max(worst, CALIBRATED._sag(cursor, point))
                        cursor = point
                    legs_used = max(legs_used, len(plan))
                    moves += 1
    unsplit = CALIBRATED._sag(_desk_at(110.0, 0.0, 2.0), _desk_at(265.0, 0.0, 2.0))
    return all([
        check("one long low leg would sag past the margin", unsplit > path.CLEARANCE_MARGIN_MM,
              f"{unsplit:.1f} mm"),
        check(f"planned, no leg of {moves} moves sags past it", worst <= path.CLEARANCE_MARGIN_MM,
              f"worst {worst:.1f} mm, at most {legs_used} legs"),
    ])


def _desk_at(radius: float, bearing: float, height: float):
    """The desk point over the arm-space polar position, at a desk height."""
    heading = math.radians(bearing)
    return CALIBRATED._at_desk_height(radius * math.sin(heading), -radius * math.cos(heading), height)


def _is_clear(planner: Planner, start, plan) -> bool:
    cursor = start
    for point in plan:
        if planner._crosses_base(cursor, point):
            return False
        cursor = point
    return True


def _rounded(points):
    return [tuple(round(value, 1) for value in point) for point in points]


TESTS = [test_ordinary_move_is_one_leg, test_refusals, test_clearance, test_round_the_base,
         test_no_leg_sags_past_the_margin]
