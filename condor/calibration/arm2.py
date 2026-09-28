#!/usr/bin/env python3
"""Set the calibration scene: three cubes, from a stack to three known places.

The arm goes to its reset pose, drops to the stacking height and waits. Build
the stack directly under the nozzle -- red on the desk, blue on top of it,
green on top of that -- press Enter, and the arm takes them off one at a time
and puts each one down where `CUBE_POSITIONS` says it goes.

That is the whole exercise: afterwards there are three coloured cubes on the
desk at coordinates the *arm* chose, which is the one thing `calibrate.py`
cannot get from a photograph.

    ./calibration/arm.py --dry-run     # check every target, board off
    ./calibration/arm.py               # run it, with the arm powered

Ctrl-C stops the arm where it stands; a second one quits. Importing this module
runs nothing and is how `calibrate.py` gets its input:

    from arm import CUBE_POSITIONS, CUBE_SIZE_MM

Everything here is the owner's, measured on his desk with his arm
(`work/PILOT-condor.md`). If the placement or the arm changes, edit
`CUBE_POSITIONS` and the `SCENE` steps that put the cubes there.
"""

import contextlib
import signal
import sys
import time
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maxarm import (MaxArm, MoveResult, MoveStatus, ProtocolError,          # noqa: E402
                    TransportError, tuning)

Position = Tuple[float, float, float]

CUBE_SIZE_MM = 40.0

# Where the three cubes end up: the centre of each one's *base*, in arm
# millimetres. This is what `calibrate.py` imports -- the arm side of the three
# point pairs the camera mapping is solved from.
#
# The base, because that is the only part of a 40 mm cube that touches the
# table plane the camera is being mapped onto; its blob centre floats 20 mm up
# and projects about 40 mm sideways at this camera's elevation.
#
# These heights are measured, not the heights the cube was released at -- the
# arm droops at reach, which is why green is released at z=78 and its 40 mm
# body still stands on the desk rather than being pushed through it. The two
# far cubes read 48 and the near one 45; that 3 mm is the arm's own z error
# across the envelope, on one flat desk.
CUBE_POSITIONS: Dict[str, Position] = {
    "green": (100.0, -254.0, 48.0),
    "blue": (-100.0, -254.0, 48.0),
    "red": (0.0, -83.0, 45.0),
}

class Move(NamedTuple):
    """One line of the playback: where to go, and what to do on arrival."""

    x: float
    y: float
    z: float
    nozzle_deg: Optional[float]         # None: leave the cup where it is
    note: str
    suction: Optional[bool] = None      # switched after arriving, if set
    is_slow: bool = False               # flown slowly: coming down onto a cube
    ask: str = ""                       # hold here until the owner is ready

    @property
    def target(self) -> Position:
        return (float(self.x), float(self.y), float(self.z))


# The steps that get the cubes to `CUBE_POSITIONS`, straight out of the PILOT.
# Nothing imports this -- it is a script, not data.
#
# Stacked, the three cubes reach 120 mm above the desk and the nozzle meets the
# top one at z=168 -- so the desk sits at z≈48 in the arm's frame, and each
# pickup height is a few millimetres into the cube's top face to seat the cup.
#
# The cup is turned *before* the move on the row it appears on, so a placement
# angle is reached in free space on the way over rather than by twisting a cube
# that is already down. The lift away from a placed cube leaves the angle alone
# for the same reason -- `None` means "wherever the cup already is" -- and it
# goes back to zero on the next approach, well clear of anything.
SCENE = (
    Move(0, -163, 168, 0, "stacking height",
         ask="stack the cubes under the nozzle -- red on the desk, then blue, then green"),

    Move(0, -163, 164, 0, "down onto the green cube", is_slow=True, suction=True),
    Move(0, -163, 184, None, "lift the green cube clear"),
    Move(100, -254, 78, -22, "set the green cube down", is_slow=True, suction=False),
    Move(100, -254, 78, -22, "set the green cube down", is_slow=True, suction=False),
    Move(100, -254, 100, None, "lift away from the green cube"),

    Move(0, -163, 140, 0, "above the blue cube"),
    Move(0, -163, 120, None, "down onto the blue cube", is_slow=True, suction=True),
    Move(0, -163, 150, None, "lift the blue cube clear"),
    Move(-100, -254, 78, 20, "set the blue cube down", is_slow=True, suction=False),
    Move(-100, -254, 78, 20, "set the blue cube down", is_slow=True, suction=False),
    Move(-100, -254, 100, None, "lift away from the blue cube"),

    Move(0, -163, 100, 0, "above the red cube"),
    Move(0, -163, 80, None, "down onto the red cube", is_slow=True, suction=True),
    Move(0, -83, 120, None, "lift the red cube clear"),
    Move(0, -83, 85, 0, "set the red cube down", is_slow=True, suction=False),
    Move(0, -83, 85, 0, "set the red cube down", is_slow=True, suction=False),
    Move(0, -83, 100, None, "lift away from the red cube"),
)


def main() -> int:
    arm = MaxArm()
    is_legal = print_plan(arm)
    if "--dry-run" in sys.argv[1:]:
        return 0 if is_legal else 1
    if not is_legal:
        print("\nsome targets are not legal -- fix the coordinates before moving the arm")
        return 1

    # Opening the port resets the ESP32, so this waits out the board's own
    # boot -- measured at 4.7 s, and about 15 s when its BLE init times out.
    # The figure is printed because that wait is the bulk of every run and the
    # only way to tell a slow board from a slow library is to look at it.
    print("\nconnecting; this resets the board, which homes the arm -- stand clear")
    started = time.monotonic()
    try:
        state = arm.connect()
    except TransportError as error:
        print(f"\nno link to the arm: {error}")
        return 1
    except ProtocolError as error:          # BoardNotReadyError is one of these
        print(f"\nthe board is not fit to drive the arm: {error}")
        return 1
    install_stop_handler(arm)
    print(f"connected in {time.monotonic() - started:.1f} s, "
          f"arm at {point(state.position)}")

    try:
        # `work/PILOT-condor.md` brackets the scene with "0,-163,210 go to
        # reset position", and that is `home()` rather than a coordinate: those
        # numbers are the owner reading off where the arm parks itself, and the
        # board's own routine is one command. Commanding them instead lands
        # 2 mm short -- 210 is above the highest pose the arm can *hold* -- and
        # the route down from a pose outside the envelope splits into sixteen
        # three-millimetre hops before the arm is back inside it.
        print("\n-- reset position")
        arm.home()
        is_done = play(arm, SCENE)
        if is_done:
            print("\n-- back to the reset position")
            arm.home()
    finally:
        arm.disconnect()
    if is_done:
        print("\nscene set. cube base centres, in arm coordinates:")
        for colour, position in CUBE_POSITIONS.items():
            print(f"   {colour:6} {point(position)}")
    return 0 if is_done else 1


def play(arm: MaxArm, scene: Sequence[Move], stop: "Stop") -> bool:
    """Every move in order, stopping at the first one that does not arrive.

    Stopping matters more here than in a demo: a move that ends somewhere
    unexpected while the cup is holding a cube leaves the cube somewhere
    unexpected too, and every later coordinate assumes it did not.

    `stop` is checked between steps as well as inside them. `arm.stop()` on its
    own only ends the move that is running -- the next `move_to()` re-arms it,
    which is right for a UI and wrong here, where Ctrl-C means "stop the whole
    sequence". A Ctrl-C landing in a suction wait rather than in a move would
    otherwise be swallowed entirely.
    """
    nozzle_deg: Optional[float] = None
    for index, move in enumerate(scene, 1):
        if stop.is_requested:
            return abandon(arm, "stopped between steps")
        print(f"\n-- {index}/{len(scene)}  {move.note}")
        if move.nozzle_deg is not None and move.nozzle_deg != nozzle_deg:
            arm.set_nozzle_angle(move.nozzle_deg)    # waits out the board's interpolation
            nozzle_deg = move.nozzle_deg
        # `work/PILOT-condor.md` asks for the arm to move slower on the way
        # down to a pickup. The whole of that is the duration the board is
        # given -- the descent is still one command, straight from where the
        # arm is to the cube. The speed is the library's own measured figure
        # for coming down onto the desk, read now rather than at import so it
        # is whatever the library currently says.
        descent = slowed_to(tuning.DESCENT_SPEED_MM_S)
        with descent if move.is_slow else contextlib.nullcontext():
            result = arm.move_to(*move.target)
        print(f"   {'ok  ' if is_arrived(result) else 'note'} {result}")
        if not is_arrived(result):
            return abandon(arm, result)
        if move.suction is not None:
            switch_suction(arm, move.suction)
        if move.ask:
            input(f"   {move.ask}, then press Enter: ")
    return True


@contextlib.contextmanager
def slowed_to(speed_mm_s: float):
    """Fly the next move at this speed instead of the travel speed.

    `tuning.TRAVEL_SPEED_MM_S` is the library's own cautious-run lever, the
    same number `MAXARM_TRAVEL_SPEED` sets from outside the process. Holding it
    down for one move is the whole of "come down slowly": the descent is still
    a single command from where the arm is to where the cube is, with no
    waypoints invented on the way.
    """
    original, tuning.TRAVEL_SPEED_MM_S = tuning.TRAVEL_SPEED_MM_S, speed_mm_s
    try:
        yield
    finally:
        tuning.TRAVEL_SPEED_MM_S = original


def print_plan(arm: MaxArm) -> bool:
    """Check every target against the envelope and the no-fly zones. No motion.

    Pure arithmetic, so this is worth reading with the board switched off --
    the base exclusion square (|x| ≤ 70, |y| ≤ 70) is hardcoded and is the one
    most likely to reject an edited coordinate.
    """
    print(f"{len(SCENE)} moves, bracketed by home() at the reset position\n")
    is_legal = True
    for index, move in enumerate(SCENE, 1):
        _, status, why = arm.check_target(move.target)
        is_legal = is_legal and status in (MoveStatus.REACHED, MoveStatus.APPROXIMATED)
        marks = "".join((" slow" if move.is_slow else "",
                         " suction-on" if move.suction else "",
                         " suction-off" if move.suction is False else "",
                         " wait" if move.ask else ""))
        cup = f"turn to {move.nozzle_deg:+4.0f}" if move.nozzle_deg is not None else "  as it is"
        print(f"{index:3}. {point(move.target):20} cup {cup}  "
              f"{status.value:13} {move.note}{marks}")
        if why:
            print(f"     {why}")
    return is_legal


def abandon(arm: MaxArm, result: MoveResult) -> bool:
    """Say what state the arm was left in. Deliberately does not send it home.

    Homing from an unknown pose with a cube on the cup throws the cube across
    the desk, and the pose itself is the evidence for whatever went wrong.
    """
    held = " and is still holding a cube" if arm.get_state().is_suction_on else ""
    print(f"   stopping here; the arm is at {point(result.position)}{held}")
    if result.status is MoveStatus.NOT_RESPONDING:
        print("   the arm accepted every command and did not move -- servo torque, "
              "mode or power. Power-cycle it at the barrel jack.")
    return False


def switch_suction(arm: MaxArm, is_on: bool) -> None:
    if is_on:
        arm.grip()
    else:
        arm.release()      # holds the valve open for a second, on the board
    print(f"   suction {'on' if is_on else 'off'}")


def is_arrived(result: MoveResult) -> bool:
    """Approximated counts as arrived, and `--dry-run` says which moves can be.

    None of the scene's coordinates approximate as written; the check is here
    so that an edited one that lands a couple of millimetres inside the
    envelope still runs, instead of stopping the playback with a cube on the cup.
    """
    return result.status in (MoveStatus.REACHED, MoveStatus.APPROXIMATED)


def install_stop_handler(arm: MaxArm) -> None:
    def handler(signum, frame):
        print("\n  stopping the arm; Ctrl-C again to quit")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        arm.stop()
    signal.signal(signal.SIGINT, handler)


def point(position: Position) -> str:
    return "({:.0f}, {:.0f}, {:.0f})".format(*position)


if __name__ == "__main__":
    sys.exit(main())
