#!/usr/bin/env python3
"""Test whether one or more (X,Y,Z) endpoints are actually reachable.

    ./test-endpoint.py 120 -180 60
    ./test-endpoint.py 120 -180 60  --home-first --verbose
    ./test-endpoint.py --targets targets.csv --out results.csv

The target is first predicted for free by envelope_model, with no motion. The
arm then covers the distance in coarse steps and switches to fine steps for the
last --margin mm, reading the pose back on every step. Ctrl-C is safe: the arm
stops where it is.
"""

import argparse
import csv
import sys
from typing import List, Optional

from envelope_model import EnvelopeModel
from maxarm_link import DEVICE, MaxArmLink, Position, ReplError
from reach_probe import (HOME, Outcome, ReachProbe, WalkResult, describe,
                         distance)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("coords", nargs="*", type=float, metavar="X Y Z",
                        help="one target, in millimetres")
    parser.add_argument("--targets", help="CSV of x,y,z rows to test in order")
    parser.add_argument("--out", help="write results to this CSV")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--step", type=float, default=2.0,
                        help="fine step in mm, used for the last stretch")
    parser.add_argument("--coarse-step", type=float, default=15.0,
                        help="step in mm while still far from the target")
    parser.add_argument("--margin", type=float, default=12.0,
                        help="mm from the target at which to switch to fine steps")
    parser.add_argument("--tolerance", type=float, default=12.0,
                        help="gross tracking-loss guard; NOT how limits are found")
    parser.add_argument("--z-drift", type=float, default=3.0,
                        help="readback BELOW the commanded Z that triggers a back-out")
    parser.add_argument("--step-ms", type=int, default=80, help="servo duration per step")
    parser.add_argument("--settle-ms", type=int, default=40, help="extra wait before readback")
    parser.add_argument("--home-first", action="store_true",
                        help="go_home() before testing (arm sweeps to origin)")
    parser.add_argument("--return-home", action="store_true",
                        help="walk back to the origin after the last target")
    parser.add_argument("--no-bounds", action="store_true",
                        help="skip the desk-safety envelope check from PILOT-anaconda.md")
    parser.add_argument("--verbose", "-v", action="store_true", help="print every step")
    return parser.parse_args()


def load_targets(args: argparse.Namespace) -> List[Position]:
    targets: List[Position] = []
    if args.coords:
        if len(args.coords) % 3 != 0:
            raise SystemExit("coordinates must come in triples of X Y Z")
        for n in range(0, len(args.coords), 3):
            targets.append(tuple(args.coords[n:n + 3]))
    if args.targets:
        with open(args.targets, newline="") as handle:
            for row in csv.reader(handle):
                if not row or row[0].lstrip().startswith("#") or row[0].strip() == "x":
                    continue
                targets.append(tuple(float(v) for v in row[:3]))
    if not targets:
        raise SystemExit("nothing to test: give X Y Z or --targets")
    return targets


def format_position(position: Optional[Position]) -> str:
    return "({:.1f}, {:.1f}, {:.1f})".format(*position) if position else "unknown"


def approach_target(probe: ReachProbe, target: Position, args: argparse.Namespace,
                    on_step) -> WalkResult:
    """Cover the distance coarsely, then close the last `margin` mm finely.

    Only the final stretch is worth stepping slowly -- that is where a limit
    would be, and it is the only part that can put the nozzle into the desk.
    """
    start = probe.position
    span = distance(start, target)
    if span > args.margin:
        fraction = (span - args.margin) / span
        staging = tuple(start[i] + (target[i] - start[i]) * fraction for i in range(3))
        crossing = probe.walk_to(staging, on_step=on_step, step_mm=args.coarse_step)
        if not crossing.is_reached:
            return crossing
    return probe.walk_to(target, on_step=on_step, step_mm=args.step)


def main() -> int:
    args = parse_args()
    targets = load_targets(args)
    rows = []

    try:
        with MaxArmLink(args.device) as link:
            if args.home_first:
                print("homing...")
                link.go_home()

            probe = ReachProbe(
                link,
                step_mm=args.step,
                tolerance_mm=args.tolerance,
                z_drift_mm=args.z_drift,
                step_ms=args.step_ms,
                settle_ms=args.settle_ms,
                is_bounds_enforced=not args.no_bounds,
            )
            print(f"start pose {format_position(probe.position)}")

            model = EnvelopeModel(link)

            for target in targets:
                predicted = model.is_predicted_reachable(target)
                print(f"\ntarget {format_position(target)}  predicted={predicted}")
                on_step = (lambda step: print("  " + describe(step))) if args.verbose else None
                result = approach_target(probe, target, args, on_step)

                print(f"  verdict: {result.outcome.value.upper()}"
                      f"  after {len(result.steps)} steps"
                      f"  reached {format_position(result.edge)}"
                      f"  (commanded {format_position(result.last_good)})")
                if result.outcome is Outcome.STUCK:
                    print("  arm did not recover Z on back-out -- stopping", file=sys.stderr)
                    rows.append((target, result))
                    break
                rows.append((target, result))

            if args.return_home:
                print("\nwalking back to origin...")
                probe.walk_to(HOME, step_mm=args.coarse_step)

    except KeyboardInterrupt:
        print("\ninterrupted -- arm left where it stopped", file=sys.stderr)
    except (ReplError, OSError) as exc:
        print(f"link error: {exc}", file=sys.stderr)
        return 1

    if args.out:
        with open(args.out, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["target_x", "target_y", "target_z", "outcome",
                             "reached_x", "reached_y", "reached_z",
                             "commanded_x", "commanded_y", "commanded_z", "steps"])
            for target, result in rows:
                writer.writerow([*target, result.outcome.value,
                                 *(round(v, 1) for v in result.edge),
                                 *result.last_good, len(result.steps)])
        print(f"\nwrote {len(rows)} rows to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
