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

Afterwards, one cube can be sent somewhere else, which is how the camera side
gets a fourth point pair and stops having to make do with an affine fit:

    ./calibration/arm.py --move red=120,-180

Ctrl-C stops the arm where it stands; a second one quits. Importing this module
runs nothing and is how `calibrate.py` gets its input:

    from arm import CUBE_POSITIONS, CUBE_SIZE_MM

Everything here is the owner's, measured on his desk with his arm
(`work/PILOT-condor.md`). If the placement or the arm changes, edit
`CUBE_POSITIONS` and the `SCENE` steps that put the cubes there.
"""

import contextlib
import math
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
# **The height is AACS zero at that x and y: the board z at which the cup is
# snug on the bare desk there**, measured by the owner with a cube out of the
# way. It is not where the cup goes to grip the cube, which is 40 mm higher --
# green is released with the cup at 78 and its 40 mm body stands on a surface
# that reads 38. The far pair read 36-38 and the near one 48, which is about
# 11 mm of difference on one flat desk: the arm's own z reading high as it
# extends (`work/STATUS-condor.md` §8.2). That difference is the reason AACS
# exists -- in it, all three of these are zero.
CUBE_POSITIONS: Dict[str, Position] = {
    "green": (100.0, -254.0, 38.0),
    "blue": (-100.0, -254.0, 36.0),
    "red": (0.0, -90.0, 48.0),
}

# The cup grips a cube's top face, so a cube standing on the desk is gripped at
# AACS z = its own height -- board z = CUBE_POSITIONS height + 40. That is where
# the scene's own placement heights come from: green's surface at 38 is set down
# with the cup at 78.
GRAPPLE_ABOVE_BASE_MM = CUBE_SIZE_MM

# Carried this far above the cup's own gripping height, a cube clears the top of
# any other cube already on the desk with room to spare.
CARRY_LIFT_MM = 30.0


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

    Move(0, -163, 162, 0, "down onto the green cube", is_slow=True, suction=True),
    Move(0, -163, 187, None, "lift the green cube clear"),
    Move(100, -254, 78, -21.49, "set the green cube down", is_slow=True, suction=False),
    Move(100, -254, 100, None, "lift away from the green cube"),

    Move(0, -160, 130, 0, "above the blue cube"),
    Move(0, -160, 120, None, "down onto the blue cube", is_slow=True, suction=True),
    Move(0, -160, 140, None, "lift the blue cube clear"),
    Move(-100, -254, 80, 21.49, "set the blue cube down", is_slow=True, suction=False),
    Move(-100, -254, 101, None, "lift away from the blue cube"),

# the 2's and 1's below instead of 0 seem to have something to do with a bug in board's kinematics
    Move(2, -159, 88, 0, "above the red cube"),
    Move(2, -159, 78, None, "down onto the red cube", is_slow=True, suction=True),
    Move(2, -159, 118, None, "lift the red cube clear"),
    Move(0, -90, 98, None, "approach from above"),
    Move(0, -90, 88, None, "set the red cube down", is_slow=True, suction=False),
    Move(1, -94, 125, None, "lift away from the red cube", is_slow=True),
)


def relocation(colour: str, target: Position) -> Tuple[Move, ...]:
    """Take one cube that is already on the desk and put it somewhere else.

    Not part of setting the scene. This exists so the camera side can have a
    fourth point pair: three cubes give three, which is exactly enough to fit an
    affine mapping and one short of the perspective one this camera really needs
    (`calibration/mapping.py`). One cube moved to a fourth known coordinate --
    with the camera untouched -- is the difference.

    The cup heights are derived rather than measured: the cube is picked off its
    known base and set down on a desk that reads the same height, so a target at
    a very different reach may land a millimetre or two high. That is harmless
    for a point pair, which only uses x and y -- but if the height matters, give
    `calibrate.py --at` the desk height you measured instead of this one.
    """
    source = CUBE_POSITIONS[colour]
    grip_z = source[2] + GRAPPLE_ABOVE_BASE_MM
    release_z = target[2] + GRAPPLE_ABOVE_BASE_MM
    carry_z = max(grip_z, release_z) + CARRY_LIFT_MM
    return (
        Move(source[0], source[1], carry_z, cup_angle(source), f"over the {colour} cube"),
        Move(source[0], source[1], grip_z, None, "down onto it", is_slow=True, suction=True),
        Move(source[0], source[1], carry_z, None, "lift it clear"),
        Move(target[0], target[1], carry_z, cup_angle(target), "over the new spot"),
        Move(target[0], target[1], release_z, None, "set it down", is_slow=True, suction=False),
        Move(target[0], target[1], carry_z, None, "lift away"),
    )


def cup_angle(position: Position) -> float:
    """The cup turned to cancel the base rotation, so the cube lands square.

    The end effector has no yaw of its own -- the whole arm swings to reach an x
    -- so a cube keeps the orientation it was picked up with only if the cup
    turns back by however far the base turned. This is where the scene's own
    -+21.49 degrees comes from, and it is 0 straight out in front.
    """
    return -math.degrees(math.atan2(position[0], -position[1])) + 0.0   # never -0.0


def main() -> int:
    try:
        moved = parse_move(sys.argv[1:])
    except ValueError as error:
        print(error)
        return 2
    steps = SCENE if moved is None else relocation(*moved)

    arm = MaxArm()
    is_legal = print_plan(arm, steps)
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
    stop = Stop(arm)
    print(f"connected in {time.monotonic() - started:.1f} s, "
          f"arm at {point(state.position)}")

    is_done = False
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
        is_done = not stop.is_requested and play(arm, steps, stop)
        if is_done:
            print("\n-- back to the reset position")
            arm.home()
    except KeyboardInterrupt:
        # The second Ctrl-C, from Python's own handler. The arm keeps whatever
        # the board was last told to do; nothing here can call that off.
        print("\ngave up waiting for the arm to stop")
    finally:
        arm.disconnect()
    if is_done and moved is None:
        print("\nscene set. cube base centres, in arm coordinates:")
        for colour, position in CUBE_POSITIONS.items():
            print(f"   {colour:6} {point(position)}")
    elif is_done:
        colour, target = moved
        print(f"\nthe {colour} cube is now at {point(target)}. `CUBE_POSITIONS` still says "
              f"it is at\n{point(CUBE_POSITIONS[colour]):>28} -- that is not edited here, "
              f"because the next run of the\nscene will put it back. Tell the camera side "
              f"where it is instead:\n\n"
              f"   ./calibration/calibrate.py --keep "
              f"--at {colour}={target[0]:.0f},{target[1]:.0f},{target[2]:.0f}")
    return 0 if is_done else 1


def parse_move(argv: Sequence[str]) -> Optional[Tuple[str, Position]]:
    """`--move <colour>=x,y[,z]`, or None for the scene as usual.

    The z is the desk under the new spot, the same thing `CUBE_POSITIONS` holds.
    Left off, it is the height that cube came from.
    """
    if "--move" not in argv:
        return None
    following = argv[argv.index("--move") + 1:]
    colour, _, coordinates = (following[0] if following else "").partition("=")
    if colour not in CUBE_POSITIONS:
        raise ValueError(f"--move wants one of {', '.join(CUBE_POSITIONS)}, "
                         f"as <colour>=x,y[,z]")
    try:
        numbers = [float(value) for value in coordinates.split(",")]
    except ValueError:
        raise ValueError(f"--move {colour}= wants numbers, got {coordinates!r}") from None
    if not 2 <= len(numbers) <= 3:
        raise ValueError(f"--move {colour}= wants x,y or x,y,z, got {coordinates!r}")
    return colour, (numbers[0], numbers[1],
                    numbers[2] if len(numbers) == 3 else CUBE_POSITIONS[colour][2])


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
            return abandon(arm, str(result))
        if move.suction is not None:
            switch_suction(arm, move.suction)
        if stop.is_requested:
            return abandon(arm, "stopped after the step completed")
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


def print_plan(arm: MaxArm, scene: Sequence[Move]) -> bool:
    """Check every target against the envelope and the no-fly zones. No motion.

    Pure arithmetic, so this is worth reading with the board switched off --
    the base exclusion square (|x| ≤ 70, |y| ≤ 70) is hardcoded and is the one
    most likely to reject an edited coordinate, and `--move` takes coordinates
    straight from the command line.
    """
    print(f"{len(scene)} moves, bracketed by home() at the reset position\n")
    is_legal = True
    for index, move in enumerate(scene, 1):
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


def abandon(arm: MaxArm, why: str) -> bool:
    """Say what state the arm was left in. Deliberately does not send it home.

    Homing from an unknown pose with a cube on the cup throws the cube across
    the desk, and the pose itself is the evidence for whatever went wrong.
    """
    state = arm.get_state()
    held = " and is still holding a cube" if state.is_suction_on else ""
    print(f"   {why}")
    print(f"   the arm is at {point(state.position or (0, 0, 0))}{held}")
    if state.last_status is MoveStatus.NOT_RESPONDING:
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


class Stop:
    """Ctrl-C: the first one halts the arm, a second one gives up on it.

    The first press asks the library to stop, which halts the arm at its next
    poll -- about a tenth of a second on a flying hop -- and abandons whatever
    wait was running. It also latches `is_requested`, which is what ends the
    playback rather than just the step: `arm.stop()` alone is cleared by the
    next `move_to()`, so a Ctrl-C between two steps would be forgotten.

    It then puts Python's own handler back, so a *second* press raises
    KeyboardInterrupt wherever it lands. That is the escape hatch for a board
    that has stopped answering, where nothing cooperative can help.
    """

    def __init__(self, arm: MaxArm) -> None:
        self.arm = arm
        self.is_requested = False
        signal.signal(signal.SIGINT, self._handle)

    def _handle(self, signum, frame) -> None:
        self.is_requested = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n  stopping the arm; Ctrl-C again to quit")
        self.arm.stop()


def point(position: Position) -> str:
    return "({:.0f}, {:.0f}, {:.0f})".format(*position)


if __name__ == "__main__":
    sys.exit(main())
