#!/usr/bin/env python3
"""Kinematics checks. No hardware, no fake board -- just arithmetic.

The important one is `test_matches_validated_reference`: the anaconda pilot's
`kinematics_reference.py` was checked against the live board over 336 points,
so it, not this file, is the ground truth. If the two ever disagree, this
library is wrong.
"""

import importlib.util
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maxarm import geometry                                   # noqa: E402
from harness import check                                     # noqa: E402

REFERENCE_PATH = (Path(__file__).resolve().parents[2]
                  / "anaconda" / "agenttools" / "kinematics_reference.py")

# A spread of poses that are all comfortably inside the envelope.
SAMPLE_POSES = [(0.0, -160.0, 150.0), (150.0, -150.0, 60.0), (-120.0, -90.0, 100.0),
                (200.0, 100.0, 60.0), (60.0, -90.0, 200.0), (0.0, -250.0, 90.0)]


def test_round_trip() -> bool:
    print("forward/inverse round trip:")
    worst = 0.0
    for pose in SAMPLE_POSES:
        recovered = geometry.forward(geometry.inverse(pose))
        worst = max(worst, math.dist(pose, recovered))
    home = geometry.forward(geometry.HOME_ANGLES)
    return all([
        check("inverse then forward returns the pose", worst < 1e-9, f"worst {worst:.2e} mm"),
        check("home angles put the tip at the documented origin",
              math.dist(home, (0.0, -162.94, 212.8)) < 0.01, f"{_format(home)}"),
        check("pulses round trip through angles",
              max(abs(a - b) for a, b in zip(
                  geometry.HOME_ANGLES,
                  geometry.pulse_to_deg(geometry.deg_to_pulse(geometry.HOME_ANGLES)))) < 1e-9),
    ])


def test_joint_frame() -> bool:
    """A renderer draws the chain, so the chain has to have the right links."""
    print("joint frame:")
    frame = geometry.joint_frame(geometry.inverse((150.0, -150.0, 60.0)))
    upper = math.dist(frame.shoulder, frame.elbow)
    fore = math.dist(frame.elbow, frame.wrist)
    wrist = math.dist(frame.wrist, frame.tip)
    return all([
        check("shoulder sits at the link-0 height", abs(frame.shoulder[2] - geometry.L0) < 1e-9),
        check("upper arm is L2", abs(upper - geometry.L2) < 1e-9, f"{upper:.3f}"),
        check("forearm is L3", abs(fore - geometry.L3) < 1e-9, f"{fore:.3f}"),
        check("wrist offset is L4", abs(wrist - geometry.L4) < 1e-9, f"{wrist:.3f}"),
        check("tip is the commanded pose",
              math.dist(frame.tip, (150.0, -150.0, 60.0)) < 1e-9),
        check("chain has five points for a polyline", len(frame.chain) == 5),
        check("bearing is 45 degrees to the right", abs(frame.bearing_deg - 45.0) < 1e-9,
              f"{frame.bearing_deg:.2f}"),
    ])


def test_firmware_limits() -> bool:
    print("firmware limits:")
    back = geometry.limit_reason((0.0, 160.0, 120.0))
    edge_in = geometry.limit_reason(_polar(200.0, 119.0, 84.0))
    edge_out = geometry.limit_reason(_polar(200.0, 121.0, 84.0))
    return all([
        check("blind cylinder is named",
              geometry.limit_reason((0.0, -30.0, 100.0)) == "blind_cylinder"),
        check("z above the silent clamp is named",
              geometry.limit_reason((0.0, -160.0, 240.0)) == "z_clamped"),
        check("beyond the links is out of reach",
              geometry.limit_reason((0.0, -300.0, 60.0)) == "out_of_reach"),
        check("straight back is outside the base fan", back is not None, f"{back}"),
        check("just inside the fan is fine", edge_in is None, f"{edge_in}"),
        check("just outside the fan is not", edge_out is not None, f"{edge_out}"),
        check("good poses are reachable",
              all(geometry.is_firmware_reachable(p) for p in SAMPLE_POSES)),
    ])


def test_envelope() -> bool:
    """The envelope must match what the sweep measured on the real arm."""
    print("reach envelope:")
    low = geometry.radius_span(84.0)
    high = geometry.radius_span(204.0)
    ceiling = geometry.max_reachable_z()
    link_sum = geometry.MAX_LINK_REACH
    return all([
        check("peak reach is just short of full extension",
              low is not None and link_sum - 5.0 < low[1] <= link_sum,
              f"r_max {low[1]:.1f} against link sum {link_sum:.1f}"),
        check("inner edge at shoulder height is the blind cylinder",
              abs(low[0] - geometry.BLIND_RADIUS) < 1.0, f"r_min {low[0]:.1f}"),
        check("reach collapses near the top, as measured (~206 at z=204)",
              high is not None and 200.0 < high[1] < 215.0, f"r_max {high[1]:.1f}"),
        check("the annulus is genuinely narrow up there",
              high[0] > 100.0, f"r_min {high[0]:.1f}"),
        check("nothing is reachable above ~213", 210.0 < ceiling < 214.0, f"{ceiling:.1f}"),
        check("no radius at all above the ceiling",
              geometry.radius_span(ceiling + 1.0) is None),
    ])


def test_matches_validated_reference() -> bool:
    """Ground truth: the anaconda port that was checked against the board."""
    print("agreement with the board-validated reference:")
    if not REFERENCE_PATH.exists():
        return check("reference port is available", False, str(REFERENCE_PATH))
    spec = importlib.util.spec_from_file_location("kinematics_reference", REFERENCE_PATH)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)

    disagreements = []
    worst_angle = 0.0
    for x in range(-280, 281, 20):
        for y in range(-280, 281, 20):
            for z in (48, 90, 150, 200):
                pose = (float(x), float(y), float(z))
                if geometry.limit_reason(pose) != reference.limit_reason(pose):
                    disagreements.append(pose)
                if geometry.is_firmware_reachable(pose):
                    worst_angle = max(worst_angle, max(
                        abs(a - b) for a, b in zip(geometry.inverse(pose),
                                                   reference.inverse(pose))))
    return all([
        check("every limit verdict agrees", not disagreements,
              f"{len(disagreements)} differ"),
        check("every joint angle agrees", worst_angle < 1e-9, f"worst {worst_angle:.2e} deg"),
    ])


def _polar(radius: float, bearing_deg: float, z: float):
    heading = math.radians(bearing_deg)
    return (radius * math.sin(heading), -radius * math.cos(heading), z)


def _format(position) -> str:
    return "({:.2f}, {:.2f}, {:.2f})".format(*position)


TESTS = [test_round_trip, test_joint_frame, test_firmware_limits, test_envelope,
         test_matches_validated_reference]
