#!/usr/bin/env python3
"""Stage-by-stage motion test, smallest risk first. You drive it.

Every stage prints what it proves, waits for you, then runs. Ctrl-C at any
point stops the arm -- a few millimetres, not a lurch -- and returns you to the
prompt, where Ctrl-C again quits.

    ./try-moves.py --dry-run     # print every stage's route, move nothing
    ./try-moves.py               # walk the stages, prompting before each
    ./try-moves.py --verbose     # also print every increment and its readback
    ./try-moves.py --desk        # include the descent to the z floor
    ./try-moves.py --nozzle      # include suction and cup rotation
    ./try-moves.py --both-sides  # allow x < 0, so the base crossing can run

Run `validate-on-board.py` first. It commands no motion and proves the host
model matches this board; if it fails, nothing here is worth trying.

Stages are ordered so each one only risks what the previous one proved. This
script is the agent's tool: it exists to be interrupted. `../main.py` is the
worked example -- a straight-through guided tour with nothing to choose.
"""

import argparse
import math
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maxarm import geometry, tuning                                   # noqa: E402
from maxarm.config import ArmConfig, Limits                           # noqa: E402
from maxarm.maxarm import MaxArm                                      # noqa: E402
from maxarm.route import NoRoute, Router                              # noqa: E402
from maxarm.protocol import ProtocolError                             # noqa: E402
from maxarm.status import MoveStatus                                  # noqa: E402
from maxarm.transport import TransportError                            # noqa: E402
from maxarm.zones import ZoneSet                                      # noqa: E402

# Mid-envelope, well clear of the desk, the base square and the edge: the
# safest pose the arm has, and where every stage starts and ends.
SAFE_POSE = (160.0, -160.0, 120.0)
NEAR_POSE = (200.0, -120.0, 90.0)
EDGE_TARGET = (0.0, -292.0, 90.0)       # ~7 mm past the envelope on purpose
BASE_ZONE_TARGET = (30.0, -40.0, 120.0)
LEFT_POSE = (-150.0, -60.0, 120.0)

# Nothing later in the run can succeed after these, so stop asking.
FATAL = (MoveStatus.STOPPED, MoveStatus.NOT_RESPONDING, MoveStatus.NOT_CONNECTED)


class Quit(Exception):
    """The operator asked to stop, or the arm gave us a reason to."""


def main() -> int:
    options = parse_arguments()
    # The same knob as MAXARM_TRAVEL_SPEED; the flag is here because this is
    # the script somebody runs with one hand on the power switch.
    tuning.TRAVEL_SPEED_MM_S = options.speed
    config = ArmConfig(
        device=options.device,
        limits=Limits(x=(0.0, None) if options.is_right_only else (None, None),
                      z=(options.z_floor, None)))

    if options.is_dry_run:
        return show_routes(config, options)

    arm = MaxArm(config)
    print(f"connecting to {config.resolved().device}")
    print("  NOTE: opening the port resets the board, which homes the arm. Stand clear.")
    try:
        state = arm.connect()
    except TransportError as error:
        print(f"\nno link to the arm:\n  {error}")
        return 1
    except ProtocolError as error:          # BoardNotReadyError is one of these
        print(f"\nthe board is not fit to drive the arm:\n  {error}")
        return 1
    install_stop_handler(arm)
    print(f"  at {fmt(state.position)}, joints {fmt(state.joints, '{:.1f}')}, "
          f"z floor {options.z_floor:.0f} mm, {options.speed:.0f} mm/s\n")

    try:
        for stage in stages(options):
            run_stage(arm, stage, options)
    except Quit:
        print("\nstopping here.")
    finally:
        print("\nreturning to the safe pose")
        try:
            report(arm.move_to(*SAFE_POSE))
        except KeyboardInterrupt:
            print("  left where it stands")
        arm.disconnect()
        print("disconnected")
    return 0


def stages(options):
    """Each stage is (label, what it proves, a target or a function of the arm)."""
    yield ("settle at a safe pose",
           "a move runs end to end in open space",
           SAFE_POSE)

    yield ("jog 10 mm out and back",
           "the arm tracks small commands, and the readback follows",
           jog_out_and_back)

    yield ("short move to a nearer pose",
           "a second move, still well above the desk",
           NEAR_POSE)

    yield ("reach past the envelope edge",
           "an impossible target is approximated, not driven into the stop",
           EDGE_TARGET)

    yield ("target inside the base square",
           "it is refused without a byte going to the board",
           BASE_ZONE_TARGET)

    if not options.is_right_only:
        yield ("reach across to the left of centre",
               "there is room on the left, and the arm can hold a pose there",
               LEFT_POSE)

        yield ("cross the base coming back",
               "the return route keeps clear of the robot -- read the dry run "
               "first to see whether it arcs around or is proven to miss it",
               NEAR_POSE)

    if options.is_nozzle_included:
        yield ("rotate the cup, pulse the suction",
               "the nozzle servo and the pump, with the arm standing still",
               exercise_nozzle)

    if options.is_desk_included:
        yield (f"guarded descent to z={options.z_floor:.0f}",
               "the last 15 mm step 2 mm at a time, watching for touchdown",
               (NEAR_POSE[0], NEAR_POSE[1], options.z_floor))


def jog_out_and_back(arm: MaxArm):
    before = arm.get_position()
    out = arm.move_relative(dx=10.0)
    moved = math.dist(before, arm.get_position()) if before else float("nan")
    print(f"    moved {moved:.1f} mm for a 10 mm command")
    report(out)
    return arm.move_relative(dx=-10.0)


def exercise_nozzle(arm: MaxArm):
    """Rotate the cup and run the pump, with dwells you can actually see.

    Both need time. `set_nozzle_angle()` waits out the rotation itself, and the
    pump only makes itself obvious after it has been running for a moment.
    """
    for angle in (45.0, -45.0, 0.0):
        print(f"    cup to {angle:+.0f} degrees")
        arm.set_nozzle_angle(angle, duration_ms=700)
        time.sleep(0.3)
    print("    pump on for 2 s -- you should hear it and feel suction at the cup")
    arm.grip()
    time.sleep(2.0)
    print("    releasing (the board holds the valve open for 1 s)")
    arm.release()
    return None


def run_stage(arm: MaxArm, stage, options) -> None:
    label, proves, action = stage
    print(f"-- {label}")
    print(f"   proves: {proves}")
    if ask("   [Enter] run, [s] skip, [q] quit: ") == "s":
        print("   skipped\n")
        return

    on_step = trace if options.is_verbose else None
    result = (action(arm) if callable(action)
              else arm.move_to(*action, on_step=on_step))
    if result is not None:
        report(result)
        if result.status in FATAL:
            if result.status is MoveStatus.NOT_RESPONDING:
                print("   the board accepted every command and the arm did not move.")
                print("   Servo torque, servo mode or servo power -- power-cycle the arm.")
            raise Quit
    print()


def show_routes(config: ArmConfig, options) -> int:
    """Read the route before trusting the arm with a new part of the desk."""
    resolved = config.resolved()
    router = Router(resolved.limits, ZoneSet(resolved.zones))
    cursor = geometry.HOME_COMMAND
    print("stage routes (dry run -- nothing is connected):\n")
    for label, proves, action in stages(options):
        print(f"-- {label}\n   proves: {proves}")
        if callable(action):
            print("   (no single move to plan)\n")
            continue
        target, status, detail = router.resolve(action)
        print(f"   {fmt(action)} -> {status.value}"
              f"{'  (' + detail + ')' if detail else ''}")
        if status in (MoveStatus.REACHED, MoveStatus.APPROXIMATED):
            try:
                hops = router.route(cursor, target)
            except NoRoute as error:
                print(f"     no route: {error}")
                print()
                continue
            for index, hop in enumerate(hops, 1):
                how = "stepped 2 mm" if hop.is_stepped else "one command"
                print(f"     {index}. {hop.purpose:<8} {fmt(hop.position)}  {how}")
            cursor = target
        print()
    return 0


def install_stop_handler(arm: MaxArm) -> None:
    def handler(signum, frame):
        print("\n  stopping; Ctrl-C again to quit")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        arm.stop()
    signal.signal(signal.SIGINT, handler)


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        raise Quit


def trace(step) -> None:
    measured = fmt(step.measured) if step.measured else "   <no read>   "
    print(f"     {fmt(step.commanded)} -> {measured}  {step.status.value}")


def report(result) -> None:
    print(f"   {'ok  ' if result.is_ok else 'note'} {result}")


def fmt(values, form: str = "{:.0f}") -> str:
    if values is None:
        return "(none)"
    return "(" + ", ".join(form.format(value) for value in values) + ")"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default=None, help="serial port (default /dev/ttyUSB0)")
    parser.add_argument("--dry-run", dest="is_dry_run", action="store_true",
                        help="print every stage's route, move nothing")
    parser.add_argument("--verbose", dest="is_verbose", action="store_true",
                        help="print every increment and its readback")
    parser.add_argument("--speed", type=float, default=tuning.TRAVEL_SPEED_MM_S,
                        help="mm per second (default %(default).0f; try 60 for a first run)")
    parser.add_argument("--z-floor", type=float, default=52.0,
                        help="operator z floor (default 52, above the survey's 48)")
    parser.add_argument("--desk", dest="is_desk_included", action="store_true",
                        help="include the descent to the z floor")
    parser.add_argument("--nozzle", dest="is_nozzle_included", action="store_true",
                        help="include cup rotation and a suction pulse")
    parser.add_argument("--both-sides", dest="is_right_only", action="store_false",
                        help="allow x < 0 (needs desk space left of centre)")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
