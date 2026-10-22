#!/usr/bin/env python3
"""True z against raw z: the formula, and that the library sends and reads through it.

Each check compares what goes down the wire, or what comes back, with the
formula written out here. None of them trusts where the fake board says its
arm is: it is a REPL that executes commands, not a model of the arm.
"""

import contextlib
import io
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maxarm import geometry, raw                               # noqa: E402
from maxarm import protocol as protocol_module                 # noqa: E402
from maxarm.path import Planner                                # noqa: E402
from maxarm.spaces import Affine                               # noqa: E402
from fake_arm import fake_arm                                  # noqa: E402
from harness import check                                      # noqa: E402

import numpy as np                                             # noqa: E402


def firmware_z(x: float, y: float, true_z: float) -> float:
    """The PILOT's formula, written out here and not taken from the library, so that this checks it."""
    out = math.hypot(x, y) - 207.0
    height = max(true_z - 44.0, 0.0)
    return true_z - 0.05 * out + (height / 120.0) ** 2 * (3.4 + 0.0126 * out)


def test_formula() -> bool:
    print("the formula, both ways:")
    near, level = raw.true_position((0.0, -107.0, 49.0)), raw.true_position((120.0, -168.6, 44.0))
    trips = [math.dist(raw.raw_position(raw.true_position(point)), point)
             for point in ((0.0, -100.0, 50.0), (200.0, -150.0, 30.0), (-250.0, 60.0, 120.0))]
    return all([
        check("100 mm in from the level reach, true is 5 mm under raw", near == (0.0, -107.0, 44.0), f"{near}"),
        check("at 207 mm true and raw agree", abs(level[2] - 44.0) < 0.01, f"{level}"),
        check("x and y untouched, and the round trip exact", max(trips) < 1e-12, f"{max(trips):.1e}"),
        check("logs show both", raw.describe((0.0, -107.0, 44.0)) == "(0.0, -107.0, 44.0 [raw 49.0])",
              raw.describe((0.0, -107.0, 44.0))),
    ])


def test_above_the_desk() -> bool:
    print("above the desk, by the height and the reach:")
    stack = {reach: raw.raw_position((0.0, -reach, 164.0))[2] for reach in (110.0, 207.0, 240.0)}
    spots = [(math.sin(turn) * reach, -math.cos(turn) * reach, z) for turn in (-1.5, 0.0, 0.7)
             for reach in (95.0, 160.0, 207.0, 280.0) for z in (-20.0, 44.0, 44.5, 84.0, 164.0, 205.0)]
    misses = [abs(raw.raw_position(spot)[2] - firmware_z(*spot)) for spot in spots]
    solved = [abs(raw.true_position((x, y, firmware_z(x, y, z)))[2] - z) for x, y, z in spots]
    return all([
        check("120 mm up, 207 mm out: true 164 is sent as 3.4 mm higher", abs(stack[207.0] - 167.4) < 1e-9,
              f"{stack[207.0]:.4f}"),
        check("and that much more further out, less nearer: the formula at 110 and 240 mm",
              abs(stack[110.0] - firmware_z(0.0, -110.0, 164.0)) < 1e-9
              and abs(stack[240.0] - firmware_z(0.0, -240.0, 164.0)) < 1e-9, f"{stack}"),
        check("anywhere, at any height: as written out here", max(misses) < 1e-9, f"{max(misses):.1e}"),
        check("a reading solved back to the true z it came from", max(solved) < 1e-9, f"{max(solved):.1e}"),
        check("on and below the desk the reach line alone", raw.raw_position((0.0, -207.0, 30.0))[2] == 30.0
              and raw.raw_position((0.0, -207.0, 44.0))[2] == 44.0),
    ])


def test_on_the_wire() -> bool:
    print("what the board is sent and what a reading becomes:")
    with fake_arm(None) as (arm, board, _):
        start = arm.get_position()
        pulses = board.namespace["bus_servo"]
        firmware = geometry.forward(geometry.pulse_to_deg(tuple(pulses.get_position(i) for i in (1, 2, 3))))
        trace = io.StringIO()
        protocol_module.IS_TRACED = True
        try:
            with contextlib.redirect_stderr(trace):
                arm.move(0.0, -107.0, 44.0)
        finally:
            protocol_module.IS_TRACED = False
        sent = board.moves()[-1]
        board.log.clear()
        raw.debug_use_raw(True)
        try:
            arm.move(0.0, -150.0, 60.0)
            sent_raw = board.moves()[-1]
        finally:
            raw.debug_use_raw(False)
    expected = firmware_z(0.0, -107.0, 44.0)
    return all([
        check("a reading is the true z the formula takes to the firmware's",
              abs(firmware_z(*start) - firmware[2]) < 1e-9,
              f"read {start[2]:.2f}, the firmware's own {firmware[2]:.2f}"),
        check("arm z 44, 107 mm out, is sent as raw 49", abs(sent[2] - expected) < 0.005, f"{sent}"),
        check("and the trace says both", "44.0 [raw 49.0]" in trace.getvalue()),
        check("debug_use_raw(True): sent as given", sent_raw[2] == 60.0, f"{sent_raw}"),
    ])


def test_firmware_limits_in_raw() -> bool:
    print("the firmware's limits are checked at the raw z it will be sent:")
    planner = Planner(Affine(np.eye(4)), is_desk_known=False)
    disagreements = []
    for reach in (100.0, 180.0, 260.0):
        for z in np.arange(150.0, 232.0, 0.5):
            point = (0.0, -reach, float(z))
            is_refused = planner.problem_with(point) is not None
            is_refused_raw = geometry.limit_reason(raw.raw_position(point)) is not None
            if is_refused != is_refused_raw:
                disagreements.append(point)
    return check("refused exactly where the firmware would refuse the raw z", not disagreements,
                 f"{disagreements[:3]}")


TESTS = [test_formula, test_above_the_desk, test_on_the_wire, test_firmware_limits_in_raw]
