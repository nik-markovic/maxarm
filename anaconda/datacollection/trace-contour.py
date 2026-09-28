#!/usr/bin/env python3
"""Trace the arm's reach boundary per Z plane and write an X,Y,Z CSV to plot.

    ./trace-contour.py --out reach.csv
    ./trace-contour.py --out reach.csv --z 144 --z-to 144      # one plane
    ./trace-contour.py --out reach.csv --resume

Described in polar terms, which is how this arm actually moves. Per plane, and
per bearing from the base axis:

  1. Ask the board which radii along that bearing are reachable. No motion --
     a whole ray in one round trip.
  2. Probe outward to find where the arm really stops reaching out.
  3. Probe inward to find where it stops folding in.

Both ends matter. Near the top of the envelope the reachable set is a narrow
annulus, not a disc: at the reset pose the arm spans X -66..+66 but Y only
-173..-151, and that inner edge is the arm refusing to fold further, not a
limit of ours. Describing the boundary as min/max X per Y row misses it
entirely, which is why this works in bearings instead.

Bearings run clockwise for the outward pass, then back counter-clockwise for
the inward one, so the arm traces the ring once in each direction.

Only points where the ARM refused are boundary. Limits we impose -- the X>=0
testing floor, the no-fly square around the base -- are recorded but marked, so
they can be filtered out rather than drawn as walls that do not exist.
"""

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

from envelope_model import EnvelopeModel
from maxarm_link import DEVICE, MaxArmLink, ReplError
from reach_probe import (MIN_RADIUS, NO_FLY_PLANNING_HALF, Outcome, ReachProbe,
                         Z_MIN, describe)

CSV_HEADER = ["x", "y", "z", "edge", "outcome", "limit_kind", "plane_z",
              "predicted_x", "predicted_y"]

ARM_LIMITS = (Outcome.REFUSED, Outcome.STALLED, Outcome.CLAMPED, Outcome.Z_SAG)

RAY_MAX = 320.0   # past any reach the links allow; the arm's refusal is the limit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="reach.csv")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--z", type=float, default=224.0, help="first Z plane")
    parser.add_argument("--z-to", type=float, default=Z_MIN, help="last Z plane")
    parser.add_argument("--z-step", type=float, default=20.0)
    parser.add_argument("--z-start", type=float, default=100.0,
                        help="plane to begin from; others are ordered outward from it")
    parser.add_argument("--grid", type=float, default=4.0,
                        help="radial resolution when mapping a bearing, mm")
    parser.add_argument("--angle-step", type=float, default=7.5,
                        help="bearing spacing around the ring, degrees")
    parser.add_argument("--step", type=float, default=2.0, help="fine probe step, mm")
    parser.add_argument("--coarse-step", type=float, default=15.0)
    parser.add_argument("--margin", type=float, default=15.0,
                        help="mm inside the predicted edge to start each probe")
    parser.add_argument("--overshoot", type=float, default=20.0,
                        help="mm to probe past the predicted edge, per attempt")
    parser.add_argument("--max-overshoot", type=float, default=80.0,
                        help="give up extending past the prediction after this many mm")
    parser.add_argument("--tolerance", type=float, default=12.0)
    parser.add_argument("--z-drift", type=float, default=10.0)
    parser.add_argument("--desk-z-drift", type=float, default=3.0)
    parser.add_argument("--desk-last-below", type=float, default=25.0)
    parser.add_argument("--step-ms", type=int, default=80)
    parser.add_argument("--settle-ms", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def say(message: str = "") -> None:
    try:
        print(message)
    except BrokenPipeError:
        pass


def float_range(start: float, stop: float, step: float) -> List[float]:
    step = abs(step)
    direction = 1.0 if stop >= start else -1.0
    count = int(math.floor(abs(stop - start) / step))
    values = [round(start + direction * step * n, 2) for n in range(count + 1)]
    if abs(values[-1] - stop) > 1e-6:
        values.append(round(stop, 2))
    return values


def plane_order(args) -> List[float]:
    """Comfortable middle first; desk-adjacent planes strictly last."""
    planes = float_range(args.z, args.z_to, args.z_step)
    desk_ceiling = Z_MIN + args.desk_last_below
    return sorted(planes, key=lambda p: (1, -p) if p <= desk_ceiling
                  else (0, abs(p - args.z_start)))


def no_fly_radius(angle: float) -> float:
    """Where the base exclusion square ends along this bearing.

    The square is not a circle, so its edge is nearer on the axes than at the
    corners; anything inward of this is ours, not the arm's.
    """
    reach = max(abs(math.cos(angle)), abs(math.sin(angle)))
    return NO_FLY_PLANNING_HALF / max(reach, 1e-6)


def bearings(args) -> List[float]:
    """Clockwise from straight ahead (-Y) round to straight back (+Y).

    Only the right half is measured; the left is its mirror image and the desk
    has no room there.
    """
    step = math.radians(args.angle_step)
    count = int(round(math.pi / step))
    return [-math.pi / 2 + step * n for n in range(count + 1)]


def map_ring(model: EnvelopeModel, z: float, args) -> Dict[float, Tuple[float, float]]:
    """Reachable radius range per bearing, from the board's kinematics. No motion."""
    ring: Dict[float, Tuple[float, float]] = {}
    for angle in bearings(args):
        floor = max(no_fly_radius(angle), MIN_RADIUS) + 1.0
        radii = float_range(round(floor, 2), RAY_MAX, args.grid)
        span = model.find_ray_span(angle, z, radii)
        if span is not None:
            ring[angle] = span
    return ring


def limit_kind(outcome: Outcome, fence: str) -> str:
    if outcome in ARM_LIMITS:
        return "arm"
    if outcome is Outcome.Z_DESK:
        return "desk"
    if outcome in (Outcome.REACHED, Outcome.SKIPPED):
        return fence
    return "incomplete"


class ContourWriter:
    def __init__(self, path: str, is_appending: bool) -> None:
        self.handle = open(path, "a" if is_appending else "w", newline="")
        self.writer = csv.writer(self.handle)
        self.row_count = 0
        if not is_appending:
            self.writer.writerow(CSV_HEADER)
            self.handle.flush()

    def add(self, edge: str, position, z: float, outcome: Outcome,
            fence: str, predicted) -> None:
        self.writer.writerow([round(position[0], 1), round(position[1], 1),
                              round(position[2], 1), edge, outcome.value,
                              limit_kind(outcome, fence), z,
                              round(predicted[0], 1), round(predicted[1], 1)])
        self.handle.flush()
        self.row_count += 1

    def close(self) -> None:
        self.handle.close()


def load_done(path: str) -> set:
    done = set()
    try:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                px, py = float(row["predicted_x"]), float(row["predicted_y"])
                done.add((float(row["plane_z"]), row["edge"],
                          round(math.degrees(math.atan2(py, px))),
                          round(math.hypot(px, py))))
    except FileNotFoundError:
        pass
    return done


def radial(x: float, y: float) -> Tuple[float, float]:
    """Unit vector from the base axis outward through (x, y)."""
    reach = math.hypot(x, y)
    if reach < 1e-6:
        return (1.0, 0.0)
    return (x / reach, y / reach)


def our_own_fence(angle: float, radius: float, edge: str,
                  tolerance: float = 6.0) -> Optional[str]:
    """Name the fence this point sits on, if it is ours rather than the arm's.

    Only the inward end can be ours, and only when it stops at the base square
    rather than at the arm's own folding limit. Getting this wrong in either
    direction is costly: calling an arm limit a fence loses a real boundary
    point, and calling a fence an arm limit draws a wall that is not there.
    """
    if edge != "r_min":
        return None
    if radius <= max(no_fly_radius(angle), MIN_RADIUS) + tolerance:
        return "no_fly"
    return None


def probe_point(probe: ReachProbe, writer: ContourWriter, angle: float,
                radius: float, edge: str, z: float, args, on_step) -> None:
    """Approach along the bearing from inside the limit, then push until it stops."""
    unit = (math.cos(angle), math.sin(angle))
    x, y = radius * unit[0], radius * unit[1]
    direction = 1.0 if edge == "r_max" else -1.0

    fence = our_own_fence(angle, radius, edge)
    if fence is not None:
        writer.add(edge, (x, y, z), z, Outcome.SKIPPED, fence, (x, y))
        return
    fence = "sanity"

    start = (round(x - unit[0] * args.margin * direction, 2),
             round(y - unit[1] * args.margin * direction, 2), z)
    approach = probe.walk_around_to(start, on_step=on_step, step_mm=args.coarse_step)
    if not approach.is_reached:
        say(f"  {edge} at ({x:6.1f},{y:7.1f})  approach failed"
            f" ({approach.outcome.value})")
        return

    # Push out in overshoot-sized bites until the ARM stops us. Ending on our
    # own cap means the model under-predicted and the real edge is further on;
    # recording that would lose the boundary at this bearing entirely.
    reached_out = 0.0
    while True:
        reached_out += args.overshoot
        target = (round(x + unit[0] * reached_out * direction, 2),
                  round(y + unit[1] * reached_out * direction, 2), z)
        walk = probe.walk_to(target, on_step=on_step, step_mm=args.step)
        if walk.outcome is not Outcome.REACHED:
            break
        if reached_out >= args.max_overshoot:
            say(f"  {math.degrees(angle):+6.1f} deg {edge}"
                f"  still moving after {reached_out:.0f} mm past prediction"
                " -- stopping at our cap")
            break
    settled = probe.settle() if walk.outcome in (
        Outcome.STALLED, Outcome.REACHED, Outcome.REFUSED) else walk.edge

    writer.add(edge, settled, z, walk.outcome, fence, (x, y))
    say(f"  {math.degrees(angle):+6.1f} deg {edge}"
        f"  predicted r={radius:6.1f}  measured ({settled[0]:6.1f},{settled[1]:7.1f})"
        f" r={math.hypot(settled[0], settled[1]):6.1f}  ({walk.outcome.value})")
    if walk.outcome in (Outcome.STALLED, Outcome.CLAMPED):
        probe.back_off_from_limit(20.0, on_step)


def trace_plane(probe: ReachProbe, model: EnvelopeModel, writer: ContourWriter,
                z: float, args, on_step, done) -> None:
    ring = map_ring(model, z, args)
    if not ring:
        say(f"  plane {z:.0f}: nothing reachable -- no motion")
        return

    angles = sorted(ring)
    widths = [ring[a][1] - ring[a][0] for a in angles]
    say(f"  reachable over {len(angles)} bearings,"
        f" {math.degrees(angles[0]):+.0f} to {math.degrees(angles[-1]):+.0f} deg;"
        f" ring width {min(widths):.0f}-{max(widths):.0f} mm (no motion yet)")

    # Outward clockwise, then inward back the other way: one lap each direction.
    schedule = [(a, ring[a][1], "r_max") for a in angles]
    schedule += [(a, ring[a][0], "r_min") for a in reversed(angles)]

    for angle, radius, edge in schedule:
        key = (z, edge, round(math.degrees(angle)), round(radius))
        if key in done:
            continue
        probe_point(probe, writer, angle, radius, edge, z, args, on_step)


def main() -> int:
    args = parse_args()
    done = load_done(args.out) if args.resume else set()
    writer = ContourWriter(args.out, bool(args.resume and done))
    on_step = (lambda step: say("    " + describe(step))) if args.verbose else None

    try:
        with MaxArmLink(args.device) as link:
            probe = ReachProbe(link, step_mm=args.step, tolerance_mm=args.tolerance,
                               z_drift_mm=args.z_drift,
                               desk_z_drift_mm=args.desk_z_drift,
                               step_ms=args.step_ms, settle_ms=args.settle_ms)
            model = EnvelopeModel(link)
            probe.escape_no_fly(on_step)
            say(f"start pose {tuple(round(v) for v in probe.position)}")

            order = plane_order(args)
            say("plane order: " + " -> ".join(f"{v:.0f}" for v in order))
            say(f"  (desk-adjacent planes <= {Z_MIN + args.desk_last_below:.0f} last)")
            for index, z in enumerate(order, start=1):
                say(f"\n--- Z plane {z:.0f}  [{index} of {len(order)}] ---")
                trace_plane(probe, model, writer, z, args, on_step, done)

    except KeyboardInterrupt:
        say("\ninterrupted -- rerun with --resume to continue")
    except (ReplError, OSError) as exc:
        say(f"link error: {exc}")
        writer.close()
        return 1

    say(f"\nwrote {writer.row_count} boundary points to {args.out}")
    say("  plot rows where limit_kind == 'arm'; the rest are fences we imposed")
    writer.close()
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        try:
            sys.stdout.flush()
        except BrokenPipeError:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    sys.exit(exit_code)
