#!/usr/bin/env python3
"""True positions against raw ones: the formula, and that the library sends and reads through it.

Each check compares what goes down the wire, or what comes back, with the
formula written out here. None of them trusts where the fake board says its
arm is: it is a REPL that executes commands, not a model of the arm.
"""

import contextlib
import io
import math
import random
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


def firmware_z(raw_x: float, raw_y: float, true_z: float) -> float:
    """The PILOT's formula, written out here and not taken from the library, so that this checks it: by the
    firmware's own reach."""
    out = math.hypot(raw_x, raw_y) - 207.0
    height = max(true_z - 44.0, 0.0)
    return true_z - 0.05 * out + (height / 120.0) ** 2 * (3.4 + 0.0126 * out)


def true_xy(raw_x: float, raw_y: float, true_z: float) -> tuple:
    """Where the cup really is across, for the firmware's x and y: further out, and further round. Written out."""
    height = min(max(true_z - 44.0, 0.0), 44.0)
    reach, bearing = math.hypot(raw_x, raw_y), math.atan2(raw_x, -raw_y)
    reach += 6.6 - 0.118 * height
    bearing += math.radians(-1.36 * math.sin(2 * bearing) + 0.33 * (math.cos(2 * bearing) - 1))
    return reach * math.sin(bearing), -reach * math.cos(bearing)


def test_formula() -> bool:
    print("the formula, both ways:")
    random.seed(7)
    spots = [(math.sin(turn) * reach, -math.cos(turn) * reach, z) for turn, reach, z in
             ((random.uniform(-2.1, 2.1), random.uniform(80.0, 290.0), random.uniform(-10.0, 230.0)) for _ in range(500))]
    sent = [raw.raw_position(spot) for spot in spots]
    across = max(math.dist(true_xy(x, y, spot[2]), spot[:2]) for (x, y, _), spot in zip(sent, spots))
    heights = max(abs(z - firmware_z(x, y, spot[2])) for (x, y, z), spot in zip(sent, spots))
    trips = max(math.dist(raw.true_position(raw_point), spot) for raw_point, spot in zip(sent, spots))
    out_on_desk, out_on_cube = raw.raw_position((0.0, -200.0, 44.0)), raw.raw_position((0.0, -200.0, 88.0))
    sides = [raw.raw_position((reach * math.sin(math.radians(turn)), -reach * math.cos(math.radians(turn)), 44.0))
             for turn, reach in ((45.0, 200.0), (-45.0, 200.0), (90.0, 200.0))]
    turned = [math.degrees(math.atan2(x, -y)) for x, y, _ in sides]
    return all([
        check("a target's raw x and y come back to it through the formula", across < 1e-9, f"{across:.1e}"),
        check("and its raw z is the formula's, by the firmware's reach", heights < 1e-9, f"{heights:.1e}"),
        check("a reading of that raw position is the target again", trips < 1e-9, f"{trips:.1e}"),
        check("200 mm straight out on the desk is sent 6.6 mm in, and not turned",
              abs(out_on_desk[1] + 193.4) < 1e-9 and out_on_desk[0] == 0.0, f"{out_on_desk}"),
        check("44 mm up, 1.4 mm in", abs(out_on_cube[1] + 198.592) < 1e-9, f"{out_on_cube}"),
        check("at 45 degrees either side the base is sent further round, and less so at 90",
              46.6 < turned[0] < 46.8 and -46.1 < turned[1] < -45.9 and 90.5 < turned[2] < 90.7, f"{turned}"),
        check("logs show both", raw.describe((0.0, -107.0, 44.0)) == "(0.0, -107.0, 44.0 [raw 0.0, -100.4, 49.3])",
              raw.describe((0.0, -107.0, 44.0))),
    ])


def test_above_the_desk() -> bool:
    print("above the desk, by the height and the reach:")
    level = 207.0 + 6.6 - 0.118 * 44.0                     # 120 mm up, where the firmware's reach is 207
    stack = {reach: raw.raw_position((0.0, -reach, 164.0)) for reach in (110.0, level, 240.0)}
    spots = [(math.sin(turn) * reach, -math.cos(turn) * reach, z) for turn in (-1.5, 0.0, 0.7)
             for reach in (95.0, 160.0, 207.0, 280.0) for z in (-20.0, 44.0, 44.5, 84.0, 164.0, 205.0)]
    misses = [abs(raw.raw_position(spot)[2] - firmware_z(*raw.raw_position(spot)[:2], spot[2])) for spot in spots]
    solved = [abs(raw.true_position((*raw.raw_position(spot)[:2], firmware_z(*raw.raw_position(spot)[:2], spot[2])))[2]
                  - spot[2]) for spot in spots]
    on_desk = raw.raw_position((0.0, -207.0 - 6.6, 44.0))
    return all([
        check("120 mm up, 207 mm out by the firmware: true 164 is sent as 3.4 mm higher",
              abs(stack[level][2] - 167.4) < 1e-9, f"{stack[level]}"),
        check("and that much more further out, less nearer: the formula at 110 and 240 mm",
              all(abs(stack[reach][2] - firmware_z(*stack[reach][:2], 164.0)) < 1e-9 for reach in (110.0, 240.0)),
              f"{stack}"),
        check("anywhere, at any height: as written out here", max(misses) < 1e-9, f"{max(misses):.1e}"),
        check("a reading solved back to the true z it came from", max(solved) < 1e-9, f"{max(solved):.1e}"),
        check("on the desk the reach line alone: 207 mm out by the firmware, true and raw z agree",
              abs(on_desk[2] - 44.0) < 1e-9 and abs(on_desk[1] + 207.0) < 1e-9, f"{on_desk}"),
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
    expected = (0.0, -100.4, firmware_z(0.0, -100.4, 44.0))
    return all([
        check("a reading is the true position the formula takes to the firmware's",
              abs(firmware_z(*firmware[:2], start[2]) - firmware[2]) < 1e-9
              and math.dist(true_xy(*firmware[:2], start[2]), start[:2]) < 1e-9,
              f"read {start}, the firmware's own {firmware}"),
        check("arm (0, -107, 44) is sent as raw (0, -100.4, 49.3)", math.dist(sent, expected) < 0.005, f"{sent}"),
        check("and the trace says both", "44.0 [raw 0.0, -100.4, 49.3]" in trace.getvalue()),
        check("debug_use_raw(True): sent as given", tuple(sent_raw) == (0.0, -150.0, 60.0), f"{sent_raw}"),
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
