#!/usr/bin/env python3
"""Run `calibration/arm.py` against a fake board, so the real one meets it second.

The scene setup drives a loaded suction cup over a desk with cubes already on
it, which is an awkward thing to debug by watching. Everything here is offline:

    ../../.venv/bin/python agenttools/test-scene.py

What is worth checking is not that the arm moved -- the fake board always moves
-- but that the playback and the coordinates agree with each other, that the
suction is on exactly while a cube is being carried, and that nothing commanded
along the way lands somewhere the nozzle should not be.

**The scene is played twice here and that is deliberate.** Replaying fifteen
moves once per question is most of what a file like this costs, so one run
answers every question about the playback and a second covers `main()` as the
owner runs it.
"""

import builtins
import contextlib
import importlib.util
import io
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from maxarm import tuning, zones                             # noqa: E402
from maxarm.maxarm import MaxArm                             # noqa: E402
from maxarm.protocol import ReplProtocol                     # noqa: E402
from maxarm.status import MoveStatus                         # noqa: E402
from maxarm.transport import SerialTransport                 # noqa: E402
from fake_arm import FakeSerial, brisk_tuning, fake_arm      # noqa: E402
from harness import check, run                               # noqa: E402

SCENE_PATH = Path(__file__).resolve().parents[1] / "calibration" / "arm.py"

# Which cubes, and where in XY -- both structural, both worth pinning. The
# heights deliberately are not: they are the owner's measurements off his own
# desk and get re-measured, so a test that pinned them would only ever be
# edited to agree. What is checked instead is that they are somewhere a desk
# could plausibly be.
EXPECTED_CUBES = {"green": (100.0, -254.0),
                  "blue": (-100.0, -254.0),
                  "red": (0.0, -90.0)}
PLAUSIBLE_DESK_MM = (20.0, 70.0)


def load_scene():
    """Import `calibration/arm.py` the way `calibrate.py` will: as a module."""
    spec = importlib.util.spec_from_file_location("arm_scene", SCENE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scene = load_scene()


def test_import_is_inert() -> bool:
    """Importing it must not open a port -- `calibrate.py` only wants the data."""
    fresh = load_scene()
    return check("importing the scene neither connects nor moves",
                 fresh.CUBE_SIZE_MM == 40.0 and len(fresh.SCENE) == len(scene.SCENE))


def test_cube_positions() -> bool:
    """The exported constants against the spec, and against the scene table."""
    is_ok = check("three cubes are exported", set(scene.CUBE_POSITIONS) == set(EXPECTED_CUBES))
    for colour, expected in EXPECTED_CUBES.items():
        got = scene.CUBE_POSITIONS.get(colour)
        is_ok &= check(f"{colour} sits at {expected} in XY", got is not None and got[:2] == expected,
                       f"got {got}")
        is_ok &= check(f"{colour} base height is somewhere a desk could be",
                       got is not None and PLAUSIBLE_DESK_MM[0] <= got[2] <= PLAUSIBLE_DESK_MM[1],
                       f"z={got[2] if got else '?'}")

    # The constants and the steps are separate on purpose -- the steps are a
    # script, not data -- so nothing but this keeps them agreeing.
    # Sets, not lists: the scene may repeat a placement row deliberately, and
    # what matters is that the two agree on *where*, not how many commands it
    # takes to get there.
    releases = {(float(move.x), float(move.y)) for move in scene.SCENE
                if move.suction is False}
    is_ok &= check("each exported position is where the scene releases a cube",
                   releases == {position[:2] for position
                                in scene.CUBE_POSITIONS.values()},
                   str(sorted(releases)))
    return is_ok


def test_plan_is_legal() -> bool:
    """Every target, checked the way `--dry-run` checks it. No board involved.

    REACHED and not merely APPROXIMATED: a scene coordinate that has to be
    pulled to the nearest legal pose puts the cube somewhere other than where
    `CUBE_POSITIONS` claims it is, which is the one error the camera side
    cannot detect.
    """
    arm = MaxArm()
    is_ok = True
    for index, move in enumerate(scene.SCENE, 1):
        _, status, why = arm.check_target(move.target)
        if status is not MoveStatus.REACHED:
            is_ok &= check(f"move {index} {move.note}", False, f"{status.value}: {why}")
    return check(f"all {len(scene.SCENE)} targets are legal, none approximated", is_ok) and is_ok


def test_the_scene_plays() -> bool:
    """One run of the playback, and every question worth asking about it.

    The fake arm tracks its commands exactly, so it needs no correction nudges
    and the scene comes out as its own coordinates and nothing else. A real arm
    may add a nudge at the end of a move it landed short of; it still never
    adds a waypoint, which is what the command list really asserts.
    """
    print("the scene, once:")
    with fake_arm() as (arm, wire), prompt_answered():
        arm.home()
        wire.arm.commands.clear()
        wire.nozzle.events.clear()
        is_done = quietly(scene.play, arm, scene.SCENE, _never_stops())
        commands, events = list(wire.arm.commands), list(wire.nozzle.events)
        end_z, is_cup_on = arm.get_state().position[2], wire.nozzle.is_on

    wanted = [move.target for move in scene.SCENE]
    # Consecutive repeats collapsed: the scene may repeat a placement row, and
    # a second `off` while the cup is already off is a no-op on the board --
    # `SuctionNozzle._off()` is guarded by `nozzle_st`. What is being asserted
    # is that the cup is on exactly while a cube is being carried.
    switches = _without_repeats(event for event in events if event in ("on", "off"))
    turns = [event for event in events if event.startswith("angle:")]
    router = MaxArm().router
    in_base = [command for command in commands if zones.BASE_ZONE.contains(command)]

    return all([
        check("the playback runs to the end", is_done),
        check(f"all {len(wanted)} steps are {len(wanted)} commands, as written",
              commands == wanted, f"{len(commands)} commands"),
        check("suction switches on/off once per cube",
              switches == ["on", "off"] * 3, str(switches)),
        check(f"the cup is turned {len(angles_asked_for())} times, not once per move",
              turns == angles_asked_for(), str(turns)),
        check("nothing is commanded below the desk floor",
              all(command[2] >= 48.0 for command in commands),
              f"lowest z={min(command[2] for command in commands):.1f}"),
        check("nothing is commanded inside the robot's base square",
              not in_base, str(in_base[:3])),
        check("every command is one the arm can hold",
              all(router.is_in_envelope(command) for command in commands)),
        check("it ends clear of the red cube it just put down", end_z > 90, f"z={end_z:.0f}"),
        check("the cup is empty at the end", not is_cup_on),
    ])


def test_a_descent_is_one_slow_command() -> bool:
    """From where the arm is to the cube, with nothing invented on the way.

    `work/PILOT-condor.md` asks for the pickup descents to be slower, and
    slower is the duration the command is given, not a sequence of smaller
    commands.
    """
    print("a descent onto a cube:")
    with fake_arm() as (arm, wire), prompt_answered():
        arm.move_to(0.0, -163.0, 100.0)
        wire.arm.commands.clear()
        with scene.slowed_to(tuning.DESCENT_SPEED_MM_S):
            quietly(arm.move_to, 0.0, -163.0, 80.0)
        descent = [command for command in wire.arm.commands if command[2] < 100.0]
    return check("a 20 mm descent onto a cube is a single command",
                 descent == [(0.0, -163.0, 80.0)], str(descent))


def test_slowing_a_move_is_temporary() -> bool:
    """`slowed_to` must not leave the whole run crawling at pickup speed."""
    original = tuning.TRAVEL_SPEED_MM_S
    with scene.slowed_to(12.0):
        inside = tuning.TRAVEL_SPEED_MM_S
    is_ok = check("a pickup descent is slower than free travel",
                  tuning.DESCENT_SPEED_MM_S < original,
                  f"{tuning.DESCENT_SPEED_MM_S} mm/s vs {original} in free space")
    is_ok &= check("it applies inside the block and is put back after",
                   inside == 12.0 and tuning.TRAVEL_SPEED_MM_S == original)
    return is_ok


def test_script_runs_end_to_end() -> bool:
    """`main()` as the owner runs it, with the board swapped underneath."""
    wires = []

    def make_arm(config=None, **fields):
        def make_port():
            port = FakeSerial()
            wires.append(port)
            return port

        transport = SerialTransport("/dev/fake", 115200, port_factory=make_port)
        return MaxArm(config, transport=transport,
                      protocol=ReplProtocol(transport), **fields)

    original, scene.MaxArm = scene.MaxArm, make_arm
    original_argv, sys.argv = sys.argv, ["arm.py"]
    try:
        with brisk_tuning(), prompt_answered():
            exit_code = quietly(scene.main)
    finally:
        scene.MaxArm, sys.argv = original, original_argv
    return check("the script exits 0 and leaves the cup empty",
                 exit_code == 0 and not wires[-1].nozzle.is_on, f"exit {exit_code}")


def test_dry_run_moves_nothing() -> bool:
    """`--dry-run` must be safe to run with the arm plugged in and powered."""
    original, scene.MaxArm = scene.MaxArm, _refuse_to_connect
    original_argv, sys.argv = sys.argv, ["arm.py", "--dry-run"]
    try:
        exit_code = quietly(scene.main)
    finally:
        scene.MaxArm, sys.argv = original, original_argv
    return check("--dry-run prints the plan and exits without connecting",
                 exit_code == 0, f"exit {exit_code}")


def test_a_cube_can_be_moved_for_a_fourth_point() -> bool:
    """`--move`: one cube off its known place and onto another, on a fake board.

    This is what turns the camera side's affine fit into a perspective one, so
    what matters is that the cube is picked up from where `CUBE_POSITIONS` says
    it is and put down where the command line said -- and that the cup is holding
    it for exactly the middle of the trip.
    """
    colour, target = scene.parse_move(["--move", "red=120,-180"])
    steps = scene.relocation(colour, target)
    print("a cube moved to a fourth position:")
    with fake_arm() as (arm, wire), prompt_answered():
        arm.home()
        wire.arm.commands.clear()
        wire.nozzle.events.clear()
        is_done = quietly(scene.play, arm, steps, _never_stops())
        commands, events = list(wire.arm.commands), list(wire.nozzle.events)

    source = scene.CUBE_POSITIONS[colour]
    grip = source[2] + scene.CUBE_SIZE_MM
    switches = [event for event in events if event in ("on", "off")]
    return all([
        check("the move runs to the end", is_done),
        check("it is six commands, as written", commands == [move.target for move in steps],
              f"{len(commands)} commands"),
        check("the cup grips one cube-height above the base arm.py recorded",
              commands[1] == (source[0], source[1], grip), str(commands[1])),
        check("and lets go one cube-height above the desk at the new place",
              commands[4] == (target[0], target[1], target[2] + scene.CUBE_SIZE_MM),
              str(commands[4])),
        check("the cup is on for the carry and off after it", switches == ["on", "off"],
              str(switches)),
        check("nothing is commanded below the desk floor",
              all(command[2] >= 48.0 for command in commands)),
        check("the cup cancels the base rotation at both ends",
              [move.nozzle_deg for move in steps if move.nozzle_deg is not None]
              == [scene.cup_angle(source), scene.cup_angle(target)]),
        check("a cube straight out in front needs no turn at all",
              scene.cup_angle((0.0, -200.0, 40.0)) == 0.0),
    ])


def test_move_refuses_nonsense() -> bool:
    """The coordinates come off the command line, so they get read carefully."""
    is_ok = check("no --move means the scene", scene.parse_move(["--dry-run"]) is None)
    is_ok &= check("without a z it keeps the height that cube came from",
                   scene.parse_move(["--move", "blue=0,-200"])[1][2]
                   == scene.CUBE_POSITIONS["blue"][2])
    for argv in (["--move"], ["--move", "purple=1,2"], ["--move", "red=1"],
                 ["--move", "red=1,2,3,4"], ["--move", "red=over,there"]):
        try:
            scene.parse_move(argv)
        except ValueError:
            continue
        is_ok &= check(f"{argv} is refused", False, "it was accepted")
    return is_ok and check("bad --move arguments are all refused with a sentence", True)


def _without_repeats(events) -> List[str]:
    collapsed: List[str] = []
    for event in events:
        if not collapsed or collapsed[-1] != event:
            collapsed.append(event)
    return collapsed


def angles_asked_for() -> List[str]:
    """The turns the scene calls for, in order, skipping repeats.

    Derived rather than written out: the placement angles are the owner's and
    get re-measured, and a test that had to be edited alongside them would only
    ever be edited to agree.
    """
    wanted, current = [], None
    for move in scene.SCENE:
        if move.nozzle_deg is not None and move.nozzle_deg != current:
            wanted.append(f"angle:{float(move.nozzle_deg)}")
            current = move.nozzle_deg
    return wanted


class _never_stops:
    """A `Stop` that never fires -- `play()` takes one and the tests do not."""

    is_requested = False


def _refuse_to_connect(config=None, **fields):
    arm = MaxArm(config, **fields)
    arm.connect = _fail_if_called
    return arm


def _fail_if_called(*args, **kwargs):
    raise AssertionError("--dry-run opened the port")


@contextlib.contextmanager
def prompt_answered(answer: str = ""):
    """Stand in for the owner pressing Enter at the stacking pause."""
    original, builtins.input = builtins.input, lambda *args: answer
    try:
        yield
    finally:
        builtins.input = original


def quietly(function, *args):
    """Run it with its own narration swallowed -- the checks are the output."""
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            return function(*args)
        except SystemExit as error:
            return error.code


TESTS = (test_import_is_inert, test_cube_positions, test_plan_is_legal,
         test_the_scene_plays, test_a_descent_is_one_slow_command,
         test_slowing_a_move_is_temporary, test_a_cube_can_be_moved_for_a_fourth_point,
         test_move_refuses_nonsense, test_script_runs_end_to_end,
         test_dry_run_moves_nothing)

if __name__ == "__main__":
    sys.exit(run(TESTS, "calibration/arm.py, on a fake board"))
