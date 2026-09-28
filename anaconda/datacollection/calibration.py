#!/usr/bin/env python3
"""Measurement routines behind calibrate-readback.py.

Three ways of asking the same question -- how faithfully does the arm follow a
command, and where does ordinary error end and a real limit begin:

  cube  : varied step sizes hopping around safe mid-space. A standing offset
          looks the same at every size; a deadband makes small steps cover
          proportionally less ground than large ones.
  axes  : one axis at a time, out and back, exposing per-axis offset and the
          cross-talk onto the axes that were meant to hold still.
  reach : extend outward at constant Z until Z gives way. This is what
          separates the arm's normal Z error from the droop it develops only
          when over-extended.

Everything here stays well above the desk and writes nothing to the board.
"""

import argparse
import csv
import math
import random
import statistics
import time
from typing import Dict, List, NamedTuple, Optional

from maxarm_link import MaxArmLink, Position
from reach_probe import X_MAX, X_MIN, Y_MAX, Y_MIN, Z_MAX, is_within_bounds

# Z=48 is the operator-enforced floor (nozzle just above the desk). 55 keeps a
# little air under it while letting the cube sit low, where the arm has by far
# the most joint freedom -- roughly 5x the servo-3 headroom it has up near 209.
DEFAULT_SAFE_FLOOR = 55.0
AXIS_NAMES = {"x": 0, "y": 1, "z": 2}


class CubeSample(NamedTuple):
    """One move of a known size, somewhere inside the safe cube."""
    requested_mm: float
    origin: Position
    commanded: Position
    first_read: Position
    settled_read: Position

    @property
    def commanded_travel(self) -> float:
        return math.dist(self.origin, self.commanded)

    @property
    def actual_travel(self) -> float:
        return math.dist(self.origin, self.settled_read)

    @property
    def travel_ratio(self) -> float:
        """How much of the commanded distance the arm actually covered."""
        asked = self.commanded_travel
        return self.actual_travel / asked if asked else 0.0

    def error(self, axis: int) -> float:
        return self.settled_read[axis] - self.commanded[axis]

    @property
    def worst_error(self) -> float:
        return max(abs(self.error(i)) for i in range(3))

    @property
    def settling(self) -> float:
        return math.dist(self.first_read, self.settled_read)


class Sample(NamedTuple):
    axis: int
    commanded: Position
    first_read: Position
    settled_read: Position

    @property
    def offset(self) -> float:
        """Settled minus commanded, on the axis being driven."""
        return self.settled_read[self.axis] - self.commanded[self.axis]

    @property
    def settling(self) -> float:
        """How much further it moved while we waited after the first read."""
        return self.settled_read[self.axis] - self.first_read[self.axis]

    @property
    def cross_talk(self) -> float:
        """Worst drift on the two axes that were supposed to stay put."""
        others = [abs(self.settled_read[i] - self.commanded[i])
                  for i in range(3) if i != self.axis]
        return max(others)


def format_position(position: Optional[Position]) -> str:
    return "({:.1f}, {:.1f}, {:.1f})".format(*position) if position else "unknown"


def cube_centre(origin: Position, half: float,
                safe_floor: float = DEFAULT_SAFE_FLOOR) -> Position:
    """Sit the cube as low as the safe floor allows.

    Low is deliberate. Servo 3 runs out of travel as Z rises, so a cube near
    Z_MAX sits against a clamp where commands are accepted and ignored -- which
    measures as under-travel and is nothing of the kind. Down at the floor the
    arm has several times the joint headroom, so what is measured there is
    really the servos.
    """
    return (min(max(origin[0], X_MIN + half), X_MAX - half),
            min(max(origin[1], Y_MIN + half), Y_MAX - half),
            min(max(origin[2], safe_floor + half), safe_floor + half))


def walk_into_cube(link: MaxArmLink, args: argparse.Namespace, origin: Position,
                   centre: Position, step_mm: float = 15.0) -> Optional[Position]:
    """Step to the cube centre rather than lunging at it."""
    span = math.dist(origin, centre)
    if span < 1.0:
        return origin
    print(f"  moving to the cube centre, {span:.0f} mm away")
    count = max(int(math.ceil(span / step_mm)), 1)
    position = origin
    for n in range(1, count + 1):
        waypoint = tuple(round(origin[i] + (centre[i] - origin[i]) * n / count, 2)
                         for i in range(3))
        verdict, measured = link.probe(waypoint, args.step_ms or 100, args.settle_ms or 60)
        if not verdict or measured is None:
            print(f"  could not reach {format_position(waypoint)} -- aborting")
            return None
        position = measured
    return position


def pick_target(rng, origin: Position, size: float, centre: Position,
                half: float) -> Position:
    """A point `size` mm from origin, kept inside the cube.

    Directions that would leave the box are reflected rather than clamped, so
    the requested step size is preserved -- otherwise short moves near a face
    would quietly become shorter ones and skew the numbers.
    """
    while True:
        vector = [rng.gauss(0.0, 1.0) for _ in range(3)]
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 1e-6:
            break
    direction = [v / norm for v in vector]

    target = []
    for axis in range(3):
        step = direction[axis] * size
        value = origin[axis] + step
        if abs(value - centre[axis]) > half:
            value = origin[axis] - step      # reflect off the face
        target.append(round(value, 2))
    return tuple(target)


def cube_probe(link: MaxArmLink, args: argparse.Namespace) -> List[CubeSample]:
    """Hop around a safe cube with varying step sizes, measuring the follow.

    Variable sizes are the point: a standing offset looks the same at every
    size, while a deadband shows up as small steps covering proportionally
    less ground than large ones.
    """
    samples: List[CubeSample] = []
    origin = link.get_position()
    if origin is None:
        print("  servo read failed -- cannot start")
        return samples

    half = args.half
    centre = cube_centre(origin, half)
    if not is_within_bounds(centre):
        print(f"  cube centre {format_position(centre)} is not reachable -- aborting")
        return samples
    sizes = [float(v) for v in args.sizes.split(",") if v.strip()]
    rng = random.Random(args.seed)
    print(f"  cube centred at {format_position(centre)}, half-size {half:.0f} mm")
    print(f"  step sizes: {', '.join(f'{v:g}' for v in sizes)} mm,"
          f" {args.rounds} round(s)\n")

    current = walk_into_cube(link, args, origin, centre)
    if current is None:
        return samples

    for _ in range(args.rounds):
        for size in sizes:
            target = pick_target(rng, current, size, centre, half)
            if not is_within_bounds(target) or target[2] < args.safe_floor:
                continue
            verdict, first = link.probe(target, args.step_ms, args.settle_ms)
            if not verdict or first is None:
                print(f"  refused {format_position(target)} -- skipping")
                continue

            settled = first
            for _ in range(args.extra_reads):
                time.sleep(args.read_gap_ms / 1000.0)
                reading = link.get_position()
                if reading is not None:
                    settled = reading

            sample = CubeSample(size, current, target, first, settled)
            samples.append(sample)
            print(f"  step {size:5.1f}  to {format_position(target)}"
                  f"  err x={sample.error(0):+5.1f} y={sample.error(1):+5.1f}"
                  f" z={sample.error(2):+5.1f}"
                  f"  moved {sample.actual_travel:6.2f}/{sample.commanded_travel:5.2f}"
                  f" ({sample.travel_ratio:4.2f})  lag={sample.settling:4.1f}")
            current = settled
    return samples


def report_cube(samples: List[CubeSample]) -> None:
    if not samples:
        print("\ncube probe: no samples")
        return

    print("\n" + "=" * 74)
    print("per-axis error (settled readback minus command), over all moves")
    for axis in range(3):
        errors = [s.error(axis) for s in samples]
        print(f"  {'xyz'[axis]}: mean {statistics.fmean(errors):+6.2f}"
              f"  range [{min(errors):+6.1f}, {max(errors):+6.1f}]"
              f"  worst |err| {max(abs(e) for e in errors):5.1f} mm")

    print("\nby commanded step size")
    print("  size   n   travel ratio   worst |err|   mean z err")
    by_size: Dict[float, List[CubeSample]] = {}
    for sample in samples:
        by_size.setdefault(sample.requested_mm, []).append(sample)
    for size in sorted(by_size):
        group = by_size[size]
        ratio = statistics.fmean(s.travel_ratio for s in group)
        worst = max(s.worst_error for s in group)
        z_error = statistics.fmean(s.error(2) for s in group)
        flag = "  <- under-travelling" if ratio < 0.7 else ""
        print(f"  {size:5.1f}  {len(group):3d}   {ratio:11.2f}   {worst:9.1f}"
              f"   {z_error:+9.2f}{flag}")

    sizes = sorted({s.requested_mm for s in samples})
    small = [s for s in samples if s.requested_mm <= 3.0]
    large = [s for s in samples if s.requested_mm >= max(sizes[-1], 5.0)]
    print("\nreading")
    if not (small and large):
        print("  need both small (<=3 mm) and larger steps to compare; widen --sizes")
    else:
        small_ratio = statistics.fmean(s.travel_ratio for s in small)
        large_ratio = statistics.fmean(s.travel_ratio for s in large)
        print(f"  small steps (<=3 mm) cover {small_ratio:.2f} of what is asked;"
              f" large (>=10 mm) cover {large_ratio:.2f}")
        if small_ratio < large_ratio - 0.15:
            print("  -> deadband: small steps lose ground. Raise --step for the sweep to")
            print(f"     the smallest size whose ratio is healthy.")
        else:
            print("  -> no deadband worth worrying about; small steps track as well as large")

    z_errors = [s.error(2) for s in samples]
    z_spread = max(z_errors) - min(z_errors)
    print(f"\n  Z error here is {statistics.fmean(z_errors):+.1f} mm"
          f" +/- {z_spread / 2:.1f} -- this is the arm's NORMAL Z behaviour in mid-space.")
    print("  Anything worse than this out at reach is over-extension, not drift.")
    print(f"  Suggested --z-drift: {max(z_spread, 2.0) + 1:.0f}"
          "  (above the normal wobble, below real droop)")
    print("=" * 74)


def leg(link: MaxArmLink, args: argparse.Namespace, axis: int,
        direction: float) -> List[Sample]:
    """Drive one axis one way, sampling each step."""
    samples: List[Sample] = []
    start = link.get_position()
    if start is None:
        print("  servo read failed at leg start -- skipping")
        return samples

    for n in range(1, int(abs(args.span) / args.step) + 1):
        commanded = list(start)
        commanded[axis] = round(start[axis] + direction * args.step * n, 2)
        commanded = tuple(commanded)
        if commanded[2] < args.safe_floor or not is_within_bounds(commanded):
            print(f"  {format_position(commanded)} leaves the safe box -- ending leg")
            break

        verdict, first = link.probe(commanded, args.step_ms, args.settle_ms)
        if not verdict or first is None:
            print(f"  arm refused {format_position(commanded)} -- ending leg")
            break

        settled = first
        for _ in range(args.extra_reads):
            time.sleep(args.read_gap_ms / 1000.0)
            reading = link.get_position()
            if reading is not None:
                settled = reading

        sample = Sample(axis, commanded, first, settled)
        samples.append(sample)
        name = "xyz"[axis]
        print(f"  cmd {name}={commanded[axis]:7.2f}  first={first[axis]:7.1f}"
              f"  settled={settled[axis]:7.1f}  offset={sample.offset:+5.1f}"
              f"  lag={sample.settling:+4.1f}  cross-talk={sample.cross_talk:4.1f}")
    return samples


def report(axis: int, samples: List[Sample]) -> None:
    name = "xyz"[axis]
    if not samples:
        print(f"\naxis {name}: no samples")
        return

    offsets = [s.offset for s in samples]
    lags = [s.settling for s in samples]
    cross = [s.cross_talk for s in samples]
    travel = [abs(samples[n].settled_read[axis] - samples[n - 1].settled_read[axis])
              for n in range(1, len(samples))]

    print(f"\naxis {name}: {len(samples)} steps")
    print(f"  offset   mean {statistics.fmean(offsets):+6.2f}  range"
          f" [{min(offsets):+.1f}, {max(offsets):+.1f}]  spread"
          f" {max(offsets) - min(offsets):.1f} mm")
    print(f"  settling mean {statistics.fmean(lags):+6.2f}  worst {max(lags, key=abs):+.1f} mm")
    print(f"  cross-talk on the other axes: max {max(cross):.1f} mm")
    if travel:
        stalled = sum(1 for t in travel if t < 0.5)
        print(f"  travel per {samples[0].commanded[axis] and ''}"
              f"commanded step: mean {statistics.fmean(travel):.2f} mm"
              f"  (asked for {abs(samples[1].commanded[axis] - samples[0].commanded[axis]):.1f})")
        print(f"  steps where it barely moved (<0.5 mm): {stalled} of {len(travel)}")


def suggest(by_axis: Dict[int, List[Sample]], step_mm: float) -> None:
    every = [s for samples in by_axis.values() for s in samples]
    if not every:
        return

    offsets = [s.offset for s in every]
    worst_sag = -min(s.settled_read[2] - s.commanded[2] for s in every)
    worst_deviation = max(max(abs(s.settled_read[i] - s.commanded[i]) for i in range(3))
                          for s in every)
    travels = []
    for axis, samples in by_axis.items():
        travels += [abs(samples[n].settled_read[axis] - samples[n - 1].settled_read[axis])
                    for n in range(1, len(samples))]
    mean_travel = statistics.fmean(travels) if travels else 0.0

    print("\n" + "=" * 68)
    print("suggested settings")
    print(f"  --tolerance  {max(worst_deviation * 1.5, 6):.0f}"
          "    loose on purpose: it only catches tracking being lost, not limits")
    print(f"  --z-drift    {max(worst_sag + 2, 3):.0f}"
          "    one-sided; only readback BELOW the command means the desk")

    if mean_travel < step_mm * 0.6:
        advised = max(step_mm * 2, 4.0)
        print(f"  --step       {advised:.0f}"
              f"    arm travelled only {mean_travel:.1f} mm per {step_mm:.0f} mm step;"
              " smaller steps are below what it resolves")
    else:
        print(f"  --step       {step_mm:.0f}    arm tracks {step_mm:.0f} mm steps fine"
              f" ({mean_travel:.1f} mm actual)")

    print(f"\n  offset spread across all axes: {max(offsets) - min(offsets):.1f} mm")
    print("  (edges are recorded from the readback, so a constant offset is harmless --")
    print("   what matters is that the arm keeps advancing until it truly stops)")
    print("=" * 68)



def ramp(link: MaxArmLink, args: argparse.Namespace, axis: int, size: float,
         count: int) -> Optional[dict]:
    """Take `count` consecutive steps of `size` along one axis, and total them.

    This is the measurement that separates a servo deadband from readback
    quantisation. A single 1 mm step is invisible either way -- the feedback
    only resolves whole millimetres. But over twenty steps, real motion
    accumulates to 20 mm while a deadband accumulates to nothing.
    """
    start = link.get_position()
    if start is None:
        return None

    measured = [start]
    for n in range(1, count + 1):
        commanded = list(start)
        commanded[axis] = round(start[axis] + size * n, 2)
        commanded = tuple(commanded)
        if not is_within_bounds(commanded) or commanded[2] < args.safe_floor:
            break
        verdict, reading = link.probe(commanded, args.step_ms, args.settle_ms)
        if not verdict or reading is None:
            break
        measured.append(reading)

    taken = len(measured) - 1
    if taken < 1:
        return None
    asked = size * taken
    got = measured[-1][axis] - start[axis]
    deltas = [measured[n][axis] - measured[n - 1][axis] for n in range(1, len(measured))]
    dead = sum(1 for d in deltas if abs(d) < 0.5)
    return {"size": size, "steps": taken, "asked": asked, "got": got,
            "ratio": got / asked if asked else 0.0, "dead_steps": dead,
            "start": start, "end": measured[-1]}


def ramp_probe(link: MaxArmLink, args: argparse.Namespace) -> List[dict]:
    """Ramp out and back at each step size, to see what accumulates."""
    results = []
    axis = AXIS_NAMES.get(args.ramp_axis, 0)
    sizes = [float(v) for v in args.ramp_sizes.split(",") if v.strip()]
    print(f"  ramping along {args.ramp_axis} in {args.ramp_steps} consecutive steps")

    for size in sizes:
        outcome = ramp(link, args, axis, size, args.ramp_steps)
        if outcome is None:
            print(f"  size {size:4.1f}: could not run")
            continue
        results.append(outcome)
        print(f"  size {size:4.1f} x{outcome['steps']:3d} steps:"
              f" asked {outcome['asked']:6.1f} mm, moved {outcome['got']:6.1f} mm"
              f"  ratio {outcome['ratio']:4.2f}"
              f"  ({outcome['dead_steps']} of {outcome['steps']} steps read as no motion)")
        # Return in one large move, which we know the arm honours.
        link.probe(outcome["start"], args.step_ms * 3, args.settle_ms * 2)
    return results


def report_ramp(results: List[dict]) -> None:
    if not results:
        print("\nramp: no results")
        return
    print("\n" + "=" * 74)
    print("cumulative travel over consecutive same-direction steps")
    print("  size   steps   asked    moved   ratio   steps reading as 'no motion'")
    for r in results:
        print(f"  {r['size']:4.1f}   {r['steps']:5d}  {r['asked']:6.1f}  {r['got']:7.1f}"
              f"   {r['ratio']:5.2f}   {r['dead_steps']:3d} of {r['steps']}")

    fine = [r for r in results if r["size"] <= 3.0]
    print("\nreading")
    if not fine:
        print("  no small sizes tested")
    elif all(r["ratio"] > 0.8 for r in fine):
        print("  Small steps ACCUMULATE properly -- the arm is moving, the 1 mm readback")
        print("  just cannot see a single step. Fine stepping is sound; edge resolution is")
        print("  limited by feedback quantisation (~1 mm), not by the arm.")
    elif all(r["ratio"] < 0.4 for r in fine):
        print("  Small steps go NOWHERE even when accumulated -- a real servo deadband.")
        print("  Raise the sweep's --step above the smallest size that accumulates, or")
        print("  use a zig-zag approach so each command is large while the net step is small.")
    else:
        worst = min(fine, key=lambda r: r["ratio"])
        best = max((r for r in results if r["ratio"] > 0.8), key=lambda r: -r["size"],
                   default=None)
        print(f"  Partial: {worst['size']:.0f} mm steps accumulate only"
              f" {worst['ratio']:.0%} of what is asked.")
        if best:
            print(f"  Smallest size that accumulates cleanly: {best['size']:.0f} mm"
                  "  <- use that as the sweep's --step")
    print("=" * 74)


def joint_margins(link: MaxArmLink, args: argparse.Namespace,
                  centre: Position, half: float) -> List[dict]:
    """Chart how close a region sits to the silent joint clamps. No motion.

    A pose with little margin accepts commands and does not move, which is
    indistinguishable from a deadband unless you look at the pulses.
    """
    from envelope_model import EnvelopeModel
    model = EnvelopeModel(link)
    rows = []
    print(f"  probing the corners and faces of the cube at {format_position(centre)}"
          f" (half {half:.0f} mm), no motion\n")
    print("     x       y       z    servo2 room   servo3 room   verdict")
    for dx in (-half, 0.0, half):
        for dy in (-half, 0.0, half):
            for dz in (-half, 0.0, half):
                point = (round(centre[0] + dx, 1), round(centre[1] + dy, 1),
                         round(centre[2] + dz, 1))
                if not is_within_bounds(point):
                    continue
                margin = model.joint_margin(point)
                if margin is None:
                    print(f"  {point[0]:6.1f}  {point[1]:6.1f}  {point[2]:6.1f}"
                          "        --            --       no IK solution")
                    rows.append({"point": point, "margin": None})
                    continue
                tight = min(margin)
                verdict = ("CLAMPED" if tight < 0 else
                           "tight" if tight < args.margin_warn else "ok")
                print(f"  {point[0]:6.1f}  {point[1]:6.1f}  {point[2]:6.1f}"
                      f"    {margin[0]:9.1f}     {margin[1]:9.1f}   {verdict}")
                rows.append({"point": point, "margin": margin})
    return rows


def report_joints(rows: List[dict], warn: float) -> None:
    solved = [r for r in rows if r["margin"] is not None]
    print("\n" + "=" * 74)
    if not solved:
        print("no point in this region has an IK solution at all")
        print("=" * 74)
        return
    clamped = [r for r in solved if min(r["margin"]) < 0]
    tight = [r for r in solved if 0 <= min(r["margin"]) < warn]
    print(f"{len(rows)} points probed, {len(rows) - len(solved)} with no IK solution")
    print(f"  already clamped: {len(clamped)}")
    print(f"  within {warn:.0f} pulses of a clamp: {len(tight)}")
    if clamped or tight:
        print("\n  This region is against a joint limit. Moves that push further into it")
        print("  are accepted and ignored -- which reads as under-travel, not as a limit.")
        print("  Re-run the cube somewhere with more room before trusting its numbers.")
    else:
        worst = min(solved, key=lambda r: min(r["margin"]))
        print(f"\n  Clear of the clamps; tightest margin {min(worst['margin']):.0f} pulses"
              f" at {format_position(worst['point'])}.")
        print("  Under-travel measured here is the servos, not a mechanical limit.")
    print("=" * 74)


def _final_error(link: MaxArmLink, target: Position, reads: int = 5) -> Optional[float]:
    """Median of several reads, to stop 1 mm quantisation dominating the answer."""
    samples = []
    for _ in range(reads):
        reading = link.get_position()
        if reading is not None:
            samples.append(reading)
    if not samples:
        return None
    median = tuple(sorted(s[axis] for s in samples)[len(samples) // 2] for axis in range(3))
    return math.dist(median, target)


def _settle_trial(link: MaxArmLink, args: argparse.Namespace, target: Position,
                  retreat: Position, is_recommanding: bool) -> Optional[dict]:
    """One trial: approach, then either re-command or merely wait, same duration.

    Both arms burn identical wall-clock time, so the only variable is whether
    fresh commands are issued. The final error is a median of several reads.
    """
    link.probe(retreat, args.step_ms * 3, args.settle_ms * 2)
    verdict, reading = link.probe(target, args.step_ms, args.settle_ms)
    if not verdict or reading is None:
        return None
    first = math.dist(reading, target)

    for _ in range(args.repeats - 1):
        if is_recommanding:
            link.probe(target, args.step_ms, args.settle_ms)
        else:
            time.sleep((args.step_ms + args.settle_ms) / 1000.0)
    final = _final_error(link, target)
    if final is None:
        return None
    return {"first": first, "final": final, "gain": first - final}


def repeat_probe(link: MaxArmLink, args: argparse.Namespace) -> List[dict]:
    """Paired A/B, interleaved and repeated: do extra commands beat plain waiting?"""
    from envelope_model import EnvelopeModel
    model = EnvelopeModel(link)
    results = []

    for radius_text in args.repeat_radii.split(","):
        radius = float(radius_text)
        target = (0.0, -radius, args.repeat_z)
        retreat = (0.0, -(radius - 30.0), args.repeat_z)
        if not is_within_bounds(target) or not model.is_predicted_reachable(target):
            print(f"  {format_position(target)} not reachable -- skipping")
            continue

        print(f"\n  reach {radius:.0f} mm, {args.trials} interleaved trials per arm")
        for trial in range(args.trials):
            # Interleaved, and order alternates, so any slow drift in the arm
            # cannot favour one condition over the other.
            order = [False, True] if trial % 2 == 0 else [True, False]
            outcome = {}
            for is_cmd in order:
                got = _settle_trial(link, args, target, retreat, is_cmd)
                if got is not None:
                    outcome["recommanded" if is_cmd else "waited"] = got
            if len(outcome) == 2:
                outcome.update({"radius": radius, "trial": trial})
                results.append(outcome)
                print(f"    trial {trial + 1}: wait {outcome['waited']['first']:4.1f}"
                      f" -> {outcome['waited']['final']:4.1f}   "
                      f"re-cmd {outcome['recommanded']['first']:4.1f}"
                      f" -> {outcome['recommanded']['final']:4.1f}")
    return results


def report_repeat(results: List[dict]) -> None:
    if not results:
        print("\nrepeat: no results")
        return
    waited = [r["waited"]["final"] for r in results]
    recommanded = [r["recommanded"]["final"] for r in results]
    wait_gain = [r["waited"]["gain"] for r in results]
    cmd_gain = [r["recommanded"]["gain"] for r in results]
    paired = [w - c for w, c in zip(waited, recommanded)]

    print("\n" + "=" * 74)
    print(f"paired A/B over {len(results)} trials -- extra commands vs plain waiting")
    print(f"  settling gain, waiting only : {statistics.fmean(wait_gain):+.2f} mm")
    print(f"  settling gain, re-commanding: {statistics.fmean(cmd_gain):+.2f} mm")
    print(f"\n  final error, waiting only   : {statistics.fmean(waited):.2f} mm")
    print(f"  final error, re-commanding  : {statistics.fmean(recommanded):.2f} mm")

    mean = statistics.fmean(paired)
    if len(paired) > 1:
        stderr = statistics.stdev(paired) / math.sqrt(len(paired))
        print(f"\n  paired difference (wait - recommand): {mean:+.2f} mm"
              f"  standard error {stderr:.2f} mm")
        if abs(mean) > 2 * stderr:
            better = "re-commanding" if mean > 0 else "waiting"
            print(f"  -> {better.upper()} is genuinely better"
                  f" ({abs(mean) / stderr:.1f} standard errors).")
        else:
            print(f"  -> NO detectable difference ({abs(mean) / stderr if stderr else 0:.1f}"
                  " standard errors).")
            print("     Extra commands buy nothing that the elapsed time does not."
                  " Prefer waiting:")
            print("     same result, no extra servo traffic.")
    if statistics.fmean(wait_gain) > 0.5 or statistics.fmean(cmd_gain) > 0.5:
        print("\n  Settling itself clearly helps -- both arms improve on the first reading.")
    print("=" * 74)


def reach_sweep(link: MaxArmLink, args: argparse.Namespace) -> List[Sample]:
    """Extend outward at a constant commanded Z and watch Z give way.

    This is the measurement that separates the two things that look alike: a
    standing Z error the arm has everywhere, and the droop it develops only
    when over-extended. Run it high above the desk -- any sag seen here is the
    arm failing to hold its own weight, with nothing to collide with.
    """
    samples: List[Sample] = []
    start = link.get_position()
    if start is None:
        print("  servo read failed -- cannot sweep")
        return samples

    print(f"  reaching out from x={start[0]:.1f} at y={start[1]:.1f}, z={start[2]:.1f}")
    x = start[0]
    while x < X_MAX:
        x = round(x + args.reach_step, 2)
        commanded = (x, start[1], start[2])
        if not is_within_bounds(commanded):
            print("  reached the enforced X limit")
            break

        verdict, first = link.probe(commanded, args.step_ms, args.settle_ms)
        if not verdict or first is None:
            print(f"  arm refused x={x:.1f} -- that is the IK limit")
            break

        settled = first
        for _ in range(args.extra_reads):
            time.sleep(args.read_gap_ms / 1000.0)
            reading = link.get_position()
            if reading is not None:
                settled = reading

        sample = Sample(2, commanded, first, settled)
        samples.append(sample)
        radius = math.hypot(settled[0], settled[1])
        sag = commanded[2] - settled[2]
        baseline = commanded[2] - samples[0].settled_read[2]
        print(f"  x={x:6.1f}  reach={radius:6.1f}  z cmd={commanded[2]:6.1f}"
              f" got={settled[2]:6.1f}  sag={sag:+5.1f}  over baseline={sag - baseline:+5.1f}")
        if sag > args.max_sag:
            print(f"  sag exceeded {args.max_sag:.0f} mm -- stopping and backing off")
            link.probe((start[0], start[1], start[2]), args.step_ms * 4, args.settle_ms * 2)
            break
    return samples


def report_reach(samples: List[Sample]) -> None:
    if len(samples) < 2:
        print("\nreach sweep: not enough samples")
        return
    baseline = samples[0].commanded[2] - samples[0].settled_read[2]
    print(f"\nreach sweep: {len(samples)} points")
    print(f"  baseline Z error at the near end: {baseline:+.1f} mm"
          "   <- this is 'normal', not a fault")
    for threshold in (2.0, 3.0, 5.0):
        onset = next((s for s in samples
                      if (s.commanded[2] - s.settled_read[2]) - baseline > threshold), None)
        if onset is None:
            print(f"  sag never exceeded baseline by {threshold:.0f} mm")
        else:
            radius = math.hypot(onset.settled_read[0], onset.settled_read[1])
            print(f"  sag passes baseline +{threshold:.0f} mm at x={onset.commanded[0]:.1f}"
                  f" (reach {radius:.1f} mm)")
    worst = max(samples, key=lambda s: s.commanded[2] - s.settled_read[2])
    print(f"  worst sag {worst.commanded[2] - worst.settled_read[2]:+.1f} mm"
          f" at x={worst.commanded[0]:.1f}")
    print("\n  --z-drift should sit above the baseline wobble but below the droop onset,")
    print("  so ordinary error is ignored and over-extension is caught.")


def write_cube_csv(path: str, samples: List[CubeSample]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["requested_mm", "origin_x", "origin_y", "origin_z",
                         "commanded_x", "commanded_y", "commanded_z",
                         "first_x", "first_y", "first_z",
                         "settled_x", "settled_y", "settled_z",
                         "err_x", "err_y", "err_z",
                         "commanded_travel", "actual_travel", "travel_ratio", "settling"])
        for s in samples:
            writer.writerow([s.requested_mm, *s.origin, *s.commanded, *s.first_read,
                             *s.settled_read, round(s.error(0), 2), round(s.error(1), 2),
                             round(s.error(2), 2), round(s.commanded_travel, 2),
                             round(s.actual_travel, 2), round(s.travel_ratio, 3),
                             round(s.settling, 2)])
    print(f"\nwrote {len(samples)} moves to {path}")


def write_csv(path: str, by_axis: Dict[int, List[Sample]]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["axis", "commanded_x", "commanded_y", "commanded_z",
                         "first_x", "first_y", "first_z",
                         "settled_x", "settled_y", "settled_z",
                         "offset", "settling", "cross_talk"])
        for axis, samples in by_axis.items():
            for s in samples:
                writer.writerow(["xyz"[axis], *s.commanded, *s.first_read, *s.settled_read,
                                 round(s.offset, 2), round(s.settling, 2),
                                 round(s.cross_talk, 2)])
    print(f"\nwrote samples to {path}")
