#!/usr/bin/env python3
"""Guided tour of the maxarm library, a step at a time. No options to choose.

Run it with the arm plugged in and the barrel jack on. Opening the port resets
the board, which homes the arm before the first step -- stand clear. Ctrl-C
stops the arm where it stands; a second one quits.

`agenttools/try-moves.py` is the prompted version, with a dry run.
"""

import signal
import sys

from maxarm import (ExclusionZone, Limits, MaxArm, MoveResult, MoveStatus,
                    ProtocolError, TransportError)

# The desk, not the arm: no room left of the centre line. Y is left open and Z
# keeps its default 48 mm floor, which is what stops the cup scraping.
DESK = Limits(x=(0.0, None))

# Where the mug sits. The robot's own base square is always excluded and is not
# declared here -- it cannot be switched off.
MUG = ExclusionZone("mug", x_min=60.0, x_max=110.0, y_min=-160.0, y_max=-110.0)


def main() -> int:
    arm = MaxArm(limits=DESK, zones=(MUG,))
    try:
        state = arm.connect()
    except TransportError as error:
        print(f"no link to the arm: {error}")
        return 1
    except ProtocolError as error:          # BoardNotReadyError is one of these
        print(f"the board is not fit to drive the arm: {error}")
        return 1
    install_stop_handler(arm)
    print(f"connected, arm at {point(state.position)}")

    try:
        check_targets(arm)
        sweep_across_the_table(arm)
        stretch_to_the_limits(arm)
        aim_into_the_no_fly_zones(arm)
        pick_and_place(arm)
    finally:
        step("send the arm home, using the board's own routine")
        arm.home()
        arm.disconnect()
    return 0


def check_targets(arm: MaxArm) -> None:
    """Pure arithmetic -- no motion, no round trip, works before connect()."""
    step("ask what the arm makes of four targets, without moving")
    for target in ((160.0, -160.0, 120.0), (85.0, -135.0, 90.0),
                   (30.0, -40.0, 120.0), (0.0, -292.0, 90.0)):
        _, status, why = arm.check_target(target)
        print(f"   {point(target):18} {status.value:13} {why}".rstrip())


def sweep_across_the_table(arm: MaxArm) -> None:
    """One command each, about 200 mm/s. The library picks the duration."""
    step("two fast sweeps across the table")
    report(arm.move_to(70.0, -250.0, 110.0))
    report(arm.move_to(250.0, -110.0, 110.0))


def stretch_to_the_limits(arm: MaxArm) -> None:
    """A near miss is approximated and reported; a far one is refused."""
    step("stretch past the arm's own limits, then past yours")
    report(arm.move_to(0.0, -292.0, 90.0))       # 1 mm past the 291 mm reach
    report(arm.move_to(150.0, -150.0, 230.0))    # above the ceiling entirely
    report(arm.move_to(-10.0, -200.0, 90.0))     # 10 mm left of DESK: clipped to it
    report(arm.move_to(-80.0, -200.0, 90.0))     # 80 mm left of it: refused instead


def aim_into_the_no_fly_zones(arm: MaxArm) -> None:
    """Refused before a byte reaches the board, and never nudged to the edge."""
    step("aim inside the two no-fly zones")
    report(arm.move_to(30.0, -40.0, 120.0))      # the robot's own base square
    report(arm.move_to(85.0, -135.0, 90.0))      # the mug declared above


def pick_and_place(arm: MaxArm) -> None:
    """Travels high, drops to 15 mm above the floor, then steps down measuring.

    DESK_CONTACT here means the cup touched down early. For a pick that is a
    success, not a fault.
    """
    step("pick something up off the desk and put it down 35 mm to the right")
    report(arm.pick_at(200.0, -150.0, 52.0))
    report(arm.place_at(235.0, -150.0, 52.0))


def step(title: str) -> None:
    print(f"\n-- {title}")


def report(result: MoveResult) -> None:
    """A MoveResult is falsy unless the arm arrived exactly where it was asked."""
    print(f"   {'ok  ' if result.is_ok else 'note'} {result}")
    if result.status is MoveStatus.NOT_RESPONDING:
        raise SystemExit("the arm accepted every command and did not move -- "
                         "servo torque, mode or power. Power-cycle it.")


def install_stop_handler(arm: MaxArm) -> None:
    def handler(signum, frame):
        print("\n  stopping the arm; Ctrl-C again to quit")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        arm.stop()
    signal.signal(signal.SIGINT, handler)


def point(position) -> str:
    return "({:.0f}, {:.0f}, {:.0f})".format(*position)


if __name__ == "__main__":
    sys.exit(main())
