#!/usr/bin/env python3
"""Check the host model against the live board. Commands no motion at all.

`set_position()` is never called. The board is only *asked* things: where its
servos are, whether a pose solves, and what pulses a pose implies. So the only
movement in the whole run is the one the firmware does to itself when the port
opens and `main.py` re-runs `go_home()` -- which happens on every connect over
this transport and cannot be suppressed.

What it proves, in order of how much it matters:

  1. `read_servo_pulses()` works on real hardware. It is new in this library and
     nothing in the anaconda pilot used it, so it is the least evidenced thing
     in the stack.
  2. The host kinematics agree with the board's own `verify_position()` and
     `position_to_pulses()` across the envelope -- the same check the anaconda
     pilot ran, repeated against this library's copy.
  3. The joint chain a 3D view would draw matches the pose the board reports.

Run it before any motion test. If anything here fails, nothing above it in the
library can be trusted.
"""

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maxarm import geometry                                   # noqa: E402
from maxarm.config import ArmConfig                           # noqa: E402
from maxarm.protocol import ReplProtocol                      # noqa: E402
from maxarm.transport import SerialTransport                  # noqa: E402
from harness import check                                     # noqa: E402

# Spread across the envelope and deliberately across its edges: reachable,
# just outside, the blind cylinder, the back wedge, and the top.
PROBE_POSES = [
    (0.0, -160.0, 150.0), (150.0, -150.0, 60.0), (-120.0, -90.0, 100.0),
    (200.0, 100.0, 60.0), (60.0, -90.0, 200.0), (0.0, -250.0, 90.0),
    (0.0, -285.0, 60.0), (0.0, -292.0, 60.0), (0.0, -300.0, 60.0),
    (0.0, -30.0, 100.0), (0.0, 160.0, 120.0), (0.0, -160.0, 213.0),
    (250.0, -100.0, 84.0), (100.0, -100.0, 48.0), (80.0, -80.0, 205.0),
    (0.0, -100.0, 190.0), (-200.0, -60.0, 70.0), (120.0, -220.0, 120.0),
]


def main() -> int:
    options = parse_arguments()
    config = ArmConfig(device=options.device).resolved()
    transport = SerialTransport(config.device, config.baud)
    protocol = ReplProtocol(transport)

    print(f"connecting to {config.device} -- the board resets and homes itself")
    started = time.time()
    protocol.connect()
    print(f"  up in {time.time() - started:.1f} s")

    try:
        results = [
            test_readback(protocol),
            test_joint_readback(protocol),
            test_solver_agreement(protocol),
            test_pulse_agreement(protocol),
            test_board_health(protocol),
        ]
    finally:
        protocol.disconnect()
        print("disconnected; the board is back in its friendly REPL")

    passed = all(results)
    print("\nALL PASS -- the host model matches this board" if passed else "\nFAILURES")
    return 0 if passed else 1


def test_readback(protocol: ReplProtocol) -> bool:
    print("\nposition readback:")
    settled = protocol.read_settled_position()
    commanded = protocol.get_commanded_position()
    samples = [protocol.read_position() for _ in range(5)]
    good = [sample for sample in samples if sample is not None]
    spread = (max(max(abs(a[i] - b[i]) for i in range(3)) for a in good for b in good)
              if len(good) > 1 else 0.0)
    return all([
        check("the servos answer", settled is not None, f"{_format(settled)}"),
        check("every read came back", len(good) == len(samples), f"{len(good)}/{len(samples)}"),
        check("readback is quantised to whole mm",
              all(float(value).is_integer() for sample in good for value in sample)),
        check("repeat reads are stable to a few mm", spread <= 3.0, f"spread {spread:.0f} mm"),
        check("the board's commanded pose is near the readback",
              commanded is not None and math.dist(commanded, settled) < 8.0,
              f"commanded {_format(commanded)}"),
    ])


def test_joint_readback(protocol: ReplProtocol) -> bool:
    """The new one: pulses straight off the servo bus, and the chain they imply."""
    print("\njoint readback (new in this library):")
    pulses = protocol.read_servo_pulses()
    if pulses is None:
        return check("bus_servo.get_position answered for all three servos", False)

    angles = geometry.pulse_to_deg(pulses)
    tip = geometry.forward(angles)
    measured = protocol.read_settled_position()
    frame = geometry.joint_frame(angles)
    return all([
        check("bus_servo.get_position answered for all three servos", True,
              f"pulses {_format(pulses)}"),
        check("pulses are in the servos' 0..1000 range",
              all(0.0 <= pulse <= 1000.0 for pulse in pulses)),
        check("the angles they imply are inside the joint limits",
              all(geometry.ANGLE_MIN <= angle <= geometry.ANGLE_MAX for angle in angles),
              f"angles {_format(angles)}"),
        check("forward kinematics on them reproduce the board's own readback",
              measured is not None and math.dist(tip, measured) < 3.0,
              f"ours {_format(tip)} vs board {_format(measured)}"),
        check("the joint chain has the right link lengths",
              abs(math.dist(frame.shoulder, frame.elbow) - geometry.L2) < 0.01
              and abs(math.dist(frame.elbow, frame.wrist) - geometry.L3) < 0.01),
    ])


def test_solver_agreement(protocol: ReplProtocol) -> bool:
    """Our verdict against the board's, pose by pose. No motion, one round trip each."""
    print("\nreachability, ours against the board's:")
    disagreements = []
    for pose in PROBE_POSES:
        # `verify_position` is only the IK and the joint-angle range -- the
        # blind cylinder, the z clamp and the servo clamps all live elsewhere
        # in `set_position`. So compare against the same two steps, not against
        # `limit_reason()`, which answers the larger question and short-circuits
        # on the cylinder before the IK is ever tried.
        try:
            geometry.position_to_pulses(pose)
            ours = True
        except geometry.Unreachable:
            ours = False
        theirs = protocol.is_position_solvable(pose)
        if ours != theirs:
            disagreements.append(f"{_format(pose)} ours={ours} board={theirs}")
        print(f"  {_format(pose)}  board solves={'yes' if theirs else 'no '}  "
              f"library says: {geometry.limit_reason(pose) or 'reachable'}")
    return check("every pose gets the same verdict", not disagreements,
                 "; ".join(disagreements))


def test_pulse_agreement(protocol: ReplProtocol) -> bool:
    """Our arithmetic against the board's, where the IK actually solves."""
    print("\nservo pulses, ours against the board's:")
    worst = 0.0
    compared = 0
    for pose in PROBE_POSES:
        if geometry.limit_reason(pose) is not None:
            continue
        x, y, z = pose
        reply = protocol.evaluate(f"arm.position_to_pulses(({-x}, {y}, {z}))")
        theirs = tuple(float(value) for value in reply)
        ours = geometry.position_to_pulses(pose)
        worst = max(worst, max(abs(a - b) for a, b in zip(ours, theirs)))
        compared += 1
    return all([
        check("there were poses to compare", compared >= 5, f"{compared} poses"),
        # The board computes in single precision; a pulse is ~0.24 degrees, so
        # anything under a tenth of a pulse is arithmetic noise, not a model error.
        check("pulses agree to well under one step", worst < 0.1, f"worst {worst:.4f} pulses"),
    ])


def test_board_health(protocol: ReplProtocol) -> bool:
    print("\nboard health:")
    free = protocol.get_free_memory()
    has_globals = protocol.evaluate(
        "[name in globals() for name in ('arm', 'nozzle', 'bus_servo', 'gc', 'time')]")
    # Stock main.py leaves ~66 names behind (`from USBDevice import *` and
    # friends), so the count means nothing on its own. What matters is that a
    # session of ours does not change it.
    before = protocol.evaluate("sorted(globals())")
    protocol.read_position()
    protocol.is_position_solvable((0.0, -160.0, 150.0))
    after = protocol.evaluate("sorted(globals())")
    return all([
        check("stock main.py's globals are all live", all(has_globals), f"{has_globals}"),
        check("heap has room to work in", free > 20_000, f"{free} bytes free"),
        check("a session of ours leaves the namespace untouched", before == after,
              f"{len(before)} names, {sorted(set(after) - set(before))} added"),
    ])


def _format(values) -> str:
    if values is None:
        return "(none)"
    return "(" + ", ".join(f"{value:.1f}" for value in values) + ")"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default=None, help="serial port (default /dev/ttyUSB0)")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
