#!/usr/bin/env python3
"""Run `main.py` and `try-moves.py` against the fake board.

The example is the first thing anyone runs on real hardware, so it is worth
knowing that it works before it is pointed at a real nozzle. `main.py` has no
functions to call -- it is eight lines of script, which is the point -- so it
is executed as written, with `MaxArm` swapped for one wired to a fake board.
"""

import argparse
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import maxarm as package                                     # noqa: E402
from maxarm.config import ArmConfig, Limits                  # noqa: E402
from maxarm.maxarm import MaxArm                             # noqa: E402
from maxarm.protocol import ReplProtocol                     # noqa: E402
from maxarm.status import MoveStatus                         # noqa: E402
from maxarm.transport import SerialTransport                 # noqa: E402
from fake_arm import FakeSerial, brisk_tuning, fake_arm      # noqa: E402
from harness import check                                    # noqa: E402

EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "main.py"


def test_example_runs() -> bool:
    """Execute `main.py` exactly as written, on a board that is not there."""
    print("the example, as written:")
    wires = []

    def make_arm(config=None, **fields):
        # Hold the port itself: disconnect() drops the transport's reference to
        # it, and everything worth asserting lives on the port.
        def make_port():
            port = FakeSerial()
            wires.append(port)
            return port

        transport = SerialTransport("/dev/fake", 115200, port_factory=make_port)
        return MaxArm(config, transport=transport,
                      protocol=ReplProtocol(transport), **fields)

    source = EXAMPLE_PATH.read_text()
    output = io.StringIO()
    original = package.MaxArm
    package.MaxArm = make_arm
    exit_code = None
    try:
        with brisk_tuning(), contextlib.redirect_stdout(output):
            try:
                exec(compile(source, str(EXAMPLE_PATH), "exec"), {"__name__": "__main__"})
            except SystemExit as error:            # it ends in sys.exit(main())
                exit_code = error.code
    finally:
        package.MaxArm = original

    text = output.getvalue()
    wire = wires[0]
    commands = list(wire.arm.commands)
    intruded = [command for command in commands
                if abs(command[0]) <= 70.0 and abs(command[1]) <= 70.0]
    left_of_fence = [command for command in commands if command[0] < -0.01]
    below_floor = [command for command in commands if command[2] < 48.0 - 0.01]
    # The guide narrates a status for each of its steps. If a step stops
    # producing the status its own text claims, the guide has started lying.
    promised = ("no_fly", "approximated", "unreachable", "reached")
    missing = [status for status in promised if status not in text]
    return all([
        check("it ran to the end", exit_code == 0 and len(wires) == 1,
              f"exit {exit_code}, {len(commands)} commands"),
        check("every status it talks the reader through actually happened",
              not missing, f"never printed: {missing}"),
        check("nothing was commanded into the base square", not intruded,
              f"{len(intruded)} of {len(commands)}"),
        check("it honoured the x fence it asked for", not left_of_fence,
              f"{len(left_of_fence)} commands left of centre"),
        check("and the z floor", not below_floor,
              f"{len(below_floor)} commands below z=48"),
        check("the pick used the suction cup", "on" in wire.nozzle.events,
              f"{wire.nozzle.events[:4]}"),
        check("it left the arm at home",
              wire.arm.position == package.geometry.HOME_COMMAND,
              f"{wire.arm.position}"),
    ])


def test_staged_motion_script() -> bool:
    """`try-moves.py` is the script that meets the real arm. Run every stage.

    Including the two that are meant to fail -- the edge approach and the base
    square -- because a stage that raises instead of reporting would strand the
    operator mid-run.
    """
    print("the staged motion script:")
    script = load_script("try-moves")
    options = argparse.Namespace(
        is_verbose=False, is_dry_run=False, is_desk_included=True,
        is_nozzle_included=True, is_right_only=True, z_floor=52.0,
        speed=1500.0, device=None)

    script.ask = lambda prompt: ""            # answer every prompt with Enter
    config = ArmConfig(limits=Limits(x=(0.0, None), z=(options.z_floor, None)))

    statuses = []
    real_report = script.report

    def capture(result):
        statuses.append(result.status)
        real_report(result)

    script.report = capture
    output = io.StringIO()
    try:
        with fake_arm(config) as (arm, wire):
            with contextlib.redirect_stdout(output):
                for stage in script.stages(options):
                    script.run_stage(arm, stage, options)
            commands = list(wire.arm.commands)
            suction = list(wire.nozzle.events)
    finally:
        script.report = real_report

    below_floor = [command for command in commands if command[2] < options.z_floor - 0.01]
    return all([
        check("every stage ran without raising", len(statuses) >= 6,
              f"{len(statuses)} results"),
        check("the edge approach was approximated", MoveStatus.APPROXIMATED in statuses,
              f"{[status.value for status in statuses]}"),
        check("the base square was refused", MoveStatus.NO_FLY in statuses),
        check("nothing was commanded below the operator floor", not below_floor,
              f"{len(below_floor)} of {len(commands)}"),
        check("the nozzle stage exercised cup and pump",
              "on" in suction and any(event.startswith("angle") for event in suction),
              f"{suction[:5]}"),
    ])


def test_dry_run_prints_routes() -> bool:
    """The dry run is how the owner reads a route before risking the arm."""
    print("the dry run:")
    script = load_script("try-moves")
    options = argparse.Namespace(
        is_verbose=False, is_dry_run=True, is_desk_included=True,
        is_nozzle_included=False, is_right_only=False, z_floor=52.0,
        speed=200.0, device=None)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = script.show_routes(ArmConfig(limits=Limits(z=(options.z_floor, None))),
                                  options)
    text = output.getvalue()
    return all([
        check("it succeeds with no board attached", code == 0),
        check("it names the refused stage", "no_fly" in text),
        check("it shows the retargeted one", "approximated" in text),
        check("it shows the approach and the stepped descent",
              "approach" in text and "stepped" in text),
        check("every stage is accounted for",
              text.count("proves:") == len(list(script.stages(options))),
              f"{text.count('proves:')} stages printed"),
    ])


def load_script(name: str):
    """Kebab-case scripts are not importable by name, so load them by path."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TESTS = [test_example_runs, test_staged_motion_script, test_dry_run_prints_routes]
