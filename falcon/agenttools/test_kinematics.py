#!/usr/bin/env python3
"""Kinematics, and the facts about the arm's real path that `path.py` is built on.

`test_matches_validated_reference` holds `geometry` to anaconda's port, which
was checked against the live board over 336 points: if they disagree, the
library is wrong. The path checks hold the planner's two assumptions to the
kinematics: the real path never comes nearer the base than its nearer end,
and it does sag below its ends, which is why `path.py` halves long legs.
"""

import importlib.util
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maxarm import geometry, path                              # noqa: E402
from harness import check                                      # noqa: E402

REFERENCE_PATH = (Path(__file__).resolve().parents[2]
                  / "anaconda" / "agenttools" / "kinematics_reference.py")


def test_matches_validated_reference() -> bool:
    print("agreement with the board-validated reference:")
    if not REFERENCE_PATH.exists():
        return check("reference port is available", False, str(REFERENCE_PATH))
    spec = importlib.util.spec_from_file_location("kinematics_reference", REFERENCE_PATH)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    disagreements, worst_angle = [], 0.0
    for x in range(-280, 281, 20):
        for y in range(-280, 281, 20):
            for z in (48, 90, 150, 200):
                pose = (float(x), float(y), float(z))
                if geometry.limit_reason(pose) != reference.limit_reason(pose):
                    disagreements.append(pose)
                if geometry.limit_reason(pose) is None:
                    worst_angle = max(worst_angle, max(
                        abs(a - b) for a, b in zip(geometry.inverse(pose), reference.inverse(pose))))
    return all([
        check("every limit verdict agrees, 3 364 poses", not disagreements,
              f"{len(disagreements)} differ"),
        check("every joint angle agrees", worst_angle < 1e-9, f"worst {worst_angle:.2e} deg"),
    ])


def test_firmware_limits() -> bool:
    print("what the firmware refuses, predicted:")
    return all([
        check("home is a pose the arm can hold", geometry.limit_reason(geometry.HOME) is None),
        check("inside the 50 mm cylinder", geometry.limit_reason((20.0, -30.0, 100.0))
              == "blind_cylinder"),
        check("above the z clamp", geometry.limit_reason((0.0, -160.0, 226.0)) == "z_clamped"),
        check("past the links' span", geometry.limit_reason((0.0, -300.0, 60.0))
              == "out_of_reach"),
        check("the back wedge", geometry.limit_reason((0.0, 160.0, 100.0)) is not None),
    ])


def test_reach_limit_is_inside_the_firmware() -> bool:
    """REACH_MAX_MM is where the owner stops the arm; the firmware stops it later, honestly."""
    print(f"the {path.REACH_MAX_MM:.0f} mm reach limit against the firmware's own:")
    first_refusals = []
    silent_clamps = 0
    for z in range(30, 111, 10):
        for bearing in range(-110, 111, 20):
            radius = path.REACH_MAX_MM
            while geometry.limit_reason(_polar(radius, bearing, z)) is None:
                radius += 0.5
            first_refusals.append(radius)
            silent_clamps += any(
                geometry.limit_reason(_polar(r, bearing, z)) in ("servo2_clamped", "servo3_clamped")
                for r in range(60, int(path.REACH_MAX_MM) + 1, 5))
    return all([
        check("at desk heights the firmware's reach ends 5 mm or more past it",
              min(first_refusals) >= path.REACH_MAX_MM + 5.0, f"first at {min(first_refusals):.1f}"),
        check("and no silent servo clamp anywhere inside it", silent_clamps == 0,
              f"{silent_clamps} fans with one"),
    ])


def test_real_path() -> bool:
    """The arm interpolates its servos, so it swings -- but away from the base, never towards it."""
    print("the arm's real path, through the kinematics:")
    worst_inward, worst_sag, worst_end = 0.0, 0.0, 0.0
    moves = 0
    for z in (50.0, 70.0, 90.0, 110.0):
        for r_start in (110.0, 170.0, 230.0, 265.0):
            for r_end in (110.0, 170.0, 230.0, 265.0):
                for b_start in (-100.0, -40.0, 0.0, 50.0):
                    for b_end in (-80.0, -10.0, 30.0, 100.0):
                        start, end = _polar(r_start, b_start, z), _polar(r_end, b_end, z)
                        trail = geometry.real_path(start, end)
                        worst_end = max(worst_end, math.dist(trail[0], start), math.dist(trail[-1], end))
                        worst_inward = max(worst_inward, max(
                            min(r_start, r_end) - math.hypot(p[0], p[1]) for p in trail))
                        worst_sag = max(worst_sag, max(z - p[2] for p in trail))
                        moves += 1
    return all([
        check("it starts and ends where it is sent", worst_end < 1e-9, f"{worst_end:.1e} mm"),
        check(f"never nearer the base than the nearer end, {moves} level moves",
              worst_inward < 0.5, f"worst {worst_inward:.2f} mm inside"),
        check("but it does sag below its ends -- what path.py halves legs for",
              worst_sag > path.CLEARANCE_MARGIN_MM, f"worst {worst_sag:.1f} mm"),
    ])


def _polar(radius: float, bearing_deg: float, z: float):
    heading = math.radians(bearing_deg)
    return (radius * math.sin(heading), -radius * math.cos(heading), float(z))


TESTS = [test_matches_validated_reference, test_firmware_limits,
         test_reach_limit_is_inside_the_firmware, test_real_path]
