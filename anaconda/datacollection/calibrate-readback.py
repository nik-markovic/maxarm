#!/usr/bin/env python3
"""Measure how faithfully the arm follows small commands, well above the desk.

    ./calibrate-readback.py
    ./calibrate-readback.py --axes xyz --step 2 --span 30
    ./calibrate-readback.py --axes z --step 4 --step-ms 200 --out drift.csv

The sweep records readback coordinates, so a standing offset between command
and feedback does not matter to the result. What does matter is whether the arm
still *advances* when told to, because that is how an edge is detected. This
answers, per axis:

  * how big is the standing offset, and does it differ by axis?
  * does the arm keep moving after the first readback (settling lag), or has it
    genuinely stopped short (deadband)?
  * how much does it actually travel per commanded step?

Each axis is driven out and back at a safe height. Nothing goes near the desk,
and nothing is written to the board.
"""

import argparse
import sys
from typing import Dict, List

from calibration import (cube_probe, format_position, joint_margins, leg, ramp_probe,
                         repeat_probe, report_repeat,
                         reach_sweep, report, report_cube, report_joints, report_ramp,
                         report_reach, suggest, write_cube_csv, write_csv)
from maxarm_link import DEVICE, MaxArmLink, ReplError
from calibration import DEFAULT_SAFE_FLOOR, cube_centre, walk_into_cube
from reach_probe import clamp_to_bounds

AXIS_NAMES = {"x": 0, "y": 1, "z": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--axes", default="zxy", help="axes to exercise, e.g. 'z' or 'xyz'")
    parser.add_argument("--step", type=float, default=2.0, help="increment to test, mm")
    parser.add_argument("--span", type=float, default=30.0, help="travel per leg, mm")
    parser.add_argument("--step-ms", type=int, default=120, help="servo duration per step")
    parser.add_argument("--settle-ms", type=int, default=60)
    parser.add_argument("--extra-reads", type=int, default=3,
                        help="readings after each move, to expose settling lag")
    parser.add_argument("--read-gap-ms", type=int, default=120)
    parser.add_argument("--mode", choices=("cube", "axes", "reach", "ramp", "joints", "repeat"),
                        default="cube",
                        help="cube: varied hops in mid-space (default);"
                             " axes: one axis at a time; reach: extend out and watch Z;"
                             " ramp: consecutive same-direction steps, to see what accumulates")
    parser.add_argument("--ramp-sizes", default="1,2,3,5,8",
                        help="step sizes to ramp, mm")
    parser.add_argument("--ramp-steps", type=int, default=20,
                        help="consecutive steps per ramp")
    parser.add_argument("--ramp-axis", default="x", choices=("x", "y", "z"))
    parser.add_argument("--centre", help="cube centre as X,Y,Z instead of near the arm")
    parser.add_argument("--repeats", type=int, default=6,
                        help="how many times to re-command each coordinate")
    parser.add_argument("--repeat-radii", default="150,175,200,225",
                        help="reach radii to test re-commanding at, mm")
    parser.add_argument("--repeat-z", type=float, default=90.0)
    parser.add_argument("--trials", type=int, default=5,
                        help="interleaved trials per arm, per radius")
    parser.add_argument("--safe-floor", type=float, default=DEFAULT_SAFE_FLOOR,
                        help="lowest Z any calibration move may command, mm")
    parser.add_argument("--margin-warn", type=float, default=40.0,
                        help="pulses of headroom below which a pose counts as tight")
    parser.add_argument("--sizes", default="1,2,3,5,8,12,20,30",
                        help="comma-separated step sizes for the cube probe, mm")
    parser.add_argument("--half", type=float, default=35.0,
                        help="half-size of the safe cube, mm")
    parser.add_argument("--rounds", type=int, default=3,
                        help="passes through the size list")
    parser.add_argument("--seed", type=int, default=7, help="so a run is repeatable")
    parser.add_argument("--reach-sweep", action="store_true",
                        help="extend outward at constant Z and chart sag against reach")
    parser.add_argument("--reach-step", type=float, default=5.0,
                        help="X increment for the reach sweep, mm")
    parser.add_argument("--max-sag", type=float, default=12.0,
                        help="stop the reach sweep once sag exceeds this, mm")
    parser.add_argument("--out", help="write every sample to this CSV")
    return parser.parse_args()


def chosen_centre(args: argparse.Namespace, start) -> tuple:
    """--centre if given, else a cube that is not jammed against the ceiling."""
    if args.centre:
        return tuple(float(v) for v in args.centre.split(","))
    return cube_centre(start, args.half, args.safe_floor)


def main() -> int:
    args = parse_args()
    axes = [AXIS_NAMES[c] for c in args.axes.lower() if c in AXIS_NAMES]
    if not axes:
        print(f"no valid axes in {args.axes!r}; use letters from xyz", file=sys.stderr)
        return 1

    by_axis: Dict[int, List[Sample]] = {}
    try:
        with MaxArmLink(args.device) as link:
            start = clamp_to_bounds(link.get_position() or link.get_cached_position())
            print(f"start pose {format_position(start)}")
            if start[2] - args.span < args.safe_floor:
                print(f"start Z too low; move the arm above"
                      f" {args.safe_floor + args.span:.0f} first", file=sys.stderr)
                return 1

            if args.mode == "repeat":
                print("\n--- repeat: does re-commanding the same coordinate help? ---")
                report_repeat(repeat_probe(link, args))
                return 0

            if args.mode == "joints":
                print("\n--- joint margins: is this region against a mechanical limit? ---")
                centre = chosen_centre(args, start)
                report_joints(joint_margins(link, args, centre, args.half),
                              args.margin_warn)
                return 0

            if args.mode == "ramp":
                print("\n--- ramp: do consecutive small steps accumulate? ---")
                centre = walk_into_cube(link, args, start, cube_centre(start, args.half))
                if centre is None:
                    return 1
                report_ramp(ramp_probe(link, args))
                return 0

            if args.mode == "cube":
                print("\n--- cube probe: varied moves in safe mid-space ---")
                cube = cube_probe(link, args)
                report_cube(cube)
                if args.out:
                    write_cube_csv(args.out, cube)
                return 0

            if args.mode == "reach" or args.reach_sweep:
                print("\n--- reach sweep: extending outward at constant Z ---")
                reach = reach_sweep(link, args)
                report_reach(reach)
                by_axis[3] = reach
                if not axes:
                    return 0

            for axis in axes:
                name = "xyz"[axis]
                print(f"\n--- axis {name}, outbound ---")
                out = leg(link, args, axis, -1.0 if axis == 2 else +1.0)
                print(f"--- axis {name}, return ---")
                back = leg(link, args, axis, +1.0 if axis == 2 else -1.0)
                by_axis[axis] = out + back

    except KeyboardInterrupt:
        print("\ninterrupted -- arm left where it stopped", file=sys.stderr)
    except (ReplError, OSError) as exc:
        print(f"link error: {exc}", file=sys.stderr)
        return 1

    for axis in axes:
        report(axis, by_axis.get(axis, []))
    suggest(by_axis, args.step)
    if args.out:
        write_csv(args.out, by_axis)
    return 0


if __name__ == "__main__":
    sys.exit(main())
