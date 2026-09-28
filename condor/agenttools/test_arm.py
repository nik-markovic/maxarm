#!/usr/bin/env python3
"""End-to-end checks against the fake board.

These go through the real transport, the real raw-REPL framing and the real
routing -- only the ESP32 is imaginary. What they are really testing is the
library's response to the board *misbehaving*: refusing, lying about a clamped
servo, drooping onto the desk, and being interrupted halfway.
"""

import math
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maxarm import geometry, tuning                                  # noqa: E402
from maxarm.config import ArmConfig, ConnectionMethod, Limits        # noqa: E402
from maxarm.maxarm import MaxArm                                     # noqa: E402
from maxarm.protocol import BoardNotReadyError, HOME_DURATION_MS      # noqa: E402
from maxarm.status import MoveStatus                                 # noqa: E402
from maxarm.transport import create_transport                        # noqa: E402
from maxarm.zones import BASE_HALF_MM                                # noqa: E402
from fake_arm import DESK_CONTACT_Z, fake_arm                        # noqa: E402
from harness import check                                            # noqa: E402

# Mid-envelope and clear of everything: where the motion checks start from.
SETTLE_POSE = (150.0, -150.0, 100.0)


def test_connect_and_state() -> bool:
    print("connect and read state:")
    with fake_arm() as (arm, wire):
        state = arm.get_state(is_fresh=True)
        frame = arm.get_joint_frame(is_fresh=True)
        near_home = math.dist(state.position, geometry.HOME_COMMAND) < 2.0
        tip_matches = frame is not None and math.dist(frame.tip, state.position) < 2.0
        names = {name for name in wire.namespace if not name.startswith("__")}
    return all([
        check("connected and reported a pose", state.is_connected and near_home,
              f"{_format(state.position)}"),
        check("joint angles came from the servos", state.joints is not None,
              f"{_round(state.joints)}" if state.joints else "none"),
        check("the joint frame agrees with the tip pose", tip_matches),
        check("nothing was defined in the board's namespace",
              names == {"arm", "time", "nozzle", "bus_servo", "gc"}, f"{sorted(names)}"),
        # Ctrl-D on an empty buffer soft-reboots the board and re-homes the arm.
        # The connect handshake sends `pass` for exactly this reason.
        check("connecting did not soft-reboot the board", wire.soft_reboots == 0,
              f"{wire.soft_reboots} reboots"),
    ])


def test_move_and_refuse() -> bool:
    print("moving, and refusing to move:")
    with fake_arm() as (arm, wire):
        good = arm.move_to(150.0, -150.0, 60.0)
        landed = math.dist(good.position, (150.0, -150.0, 60.0))

        before = len(wire.arm.commands)
        inside = arm.move_to(30.0, -40.0, 120.0)
        no_fly_commands = len(wire.arm.commands) - before

        before = len(wire.arm.commands)
        far = arm.move_to(0.0, -400.0, 60.0)
        far_commands = len(wire.arm.commands) - before

        edge = arm.move_to(0.0, -292.0, 60.0)
    return all([
        check("a good move reports reached", good.status is MoveStatus.REACHED, str(good)),
        check("and lands within a few mm", landed < 5.0, f"{landed:.1f} mm"),
        check("a target in the base square sends nothing at all",
              inside.status is MoveStatus.NO_FLY and no_fly_commands == 0,
              f"{inside.status.value}, {no_fly_commands} commands"),
        check("a target far outside sends nothing either",
              far.status is MoveStatus.UNREACHABLE and far_commands == 0,
              f"{far.status.value}, {far_commands} commands"),
        check("a target just past the edge is approximated, not refused",
              edge.status is MoveStatus.APPROXIMATED, str(edge)),
        check("and the residual is reported honestly",
              0.0 < edge.residual_mm < tuning.RETARGET_LIMIT_MM,
              f"{edge.residual_mm:.1f} mm from the request"),
    ])


def test_safe_space_is_one_command() -> bool:
    """The headline behaviour: a safe move is one command, not a stream of chunks.

    What makes that safe is knowing the swing in advance --
    `geometry.pulse_path()` says where the arm will actually go, and it is
    nowhere near a straight line. The library watches while it flies.
    """
    print("a safe move is one command:")
    with fake_arm() as (arm, wire):
        arm.move_to(*SETTLE_POSE)
        wire.arm.commands.clear()
        result = arm.move_to(60.0, -240.0, 120.0)
        commands = len(wire.arm.commands)
        polls = sum(1 for step in result.steps if step.measured is not None)
        hops = arm.router.route(SETTLE_POSE, (60.0, -240.0, 120.0))
    return all([
        check("the move arrives", result.status is MoveStatus.REACHED, str(result)),
        check("the route is a single hop", len(hops) == 1,
              f"{[hop.purpose for hop in hops]}"),
        check("one command, not a stream of chunks", commands == 1,
              f"{commands} commands"),
        check("and the arm is watched while it flies", polls >= 1,
              f"{polls} in-flight readings"),
    ])


def test_routes_around_the_base() -> bool:
    """The move that matters: legal start, legal end, illegal straight line."""
    print("driving around the base square:")
    with fake_arm(ArmConfig(limits=Limits(z=(48.0, None)))) as (arm, wire):
        arm.move_to(-150.0, -30.0, 60.0)
        wire.arm.commands.clear()
        result = arm.move_to(150.0, -30.0, 60.0)
        intruded = [command for command in wire.arm.commands
                    if abs(command[0]) <= BASE_HALF_MM and abs(command[1]) <= BASE_HALF_MM]
        closest = min(max(abs(command[0]), abs(command[1])) for command in wire.arm.commands)
    return all([
        check("the move completes", result.status is MoveStatus.REACHED, str(result)),
        check("no commanded point ever entered the square", not intruded,
              f"{len(intruded)} of {len(wire.arm.commands)}"),
        check("and it kept clear of the boundary", closest > BASE_HALF_MM + 4.0,
              f"closest max-norm {closest:.1f} mm"),
    ])


def test_desk_descent() -> bool:
    """Travel high, drop to the guard band flying, walk the last few millimetres."""
    print("descending to the desk:")
    with fake_arm() as (arm, wire):
        arm.move_to(*SETTLE_POSE)
        wire.arm.commands.clear()
        # Just clear of the surface the fake board models, and inside the guard
        # band, which is now floor..floor+15 -- a fixed 50 used to satisfy both
        # and satisfies neither since the floor came down to meet the far desk.
        desk = DESK_CONTACT_Z + 1.0
        result = arm.move_to(200.0, -120.0, desk)
        commands = wire.arm.commands
        below_band = [command for command in commands
                      if command[2] < Limits().z_floor + tuning.DESK_GUARD_MM - 0.5]
        gaps = [abs(below_band[i + 1][2] - below_band[i][2])
                for i in range(len(below_band) - 1)]
    return all([
        check("the descent completes", result.status is MoveStatus.REACHED, str(result)),
        check("the last stretch is walked in small steps",
              bool(gaps) and max(gaps) <= tuning.DESCENT_STEP_MM + 0.01,
              f"largest step {max(gaps):.1f} mm" if gaps else "no stepping"),
        check("but the whole move is not", len(commands) < 20,
              f"{len(commands)} commands in total"),
        check("and it ends at the floor", abs(result.position[2] - desk) < 3.0,
              f"z={result.position[2]:.1f}"),
    ])


def test_desk_contact() -> bool:
    """The nozzle touching down must stop the descent and back the arm off."""
    print("desk contact during a descent:")
    with fake_arm(ArmConfig(limits=Limits(z=(40.0, None)))) as (arm, wire):
        arm.move_to(200.0, -100.0, 80.0)
        result = arm.move_to(200.0, -100.0, 42.0)
        final_z = arm.get_state().position[2]
    return all([
        check("contact is detected", result.status is MoveStatus.DESK_CONTACT, str(result)),
        check("and the arm is backed off, not left pressing",
              final_z > 42.0 + 1.0, f"ended at z={final_z:.1f}"),
        check("the result says where it stopped",
              result.residual_mm > 0.0, f"{result.residual_mm:.1f} mm short"),
    ])


def test_accepted_but_motionless() -> bool:
    """Two faults that both look like "True, and nothing happened".

    Never moved at all is a dead servo bus -- torque off, motor mode, or no
    servo power -- and it is worth its own status, because the operator's next
    action is to go and look at the arm rather than pick a different target.
    Moved and then stopped is a limit or an obstruction.
    """
    print("accepted commands that produce no motion:")
    with fake_arm() as (arm, wire):
        arm.move_to(150.0, -150.0, 90.0)
        wire.arm.is_frozen = True
        dead = arm.move_to(220.0, -120.0, 90.0)
        is_all_accepted = wire.arm.move_count > 0

    with fake_arm() as (arm, wire):
        arm.move_to(150.0, -150.0, 60.0)
        seen = []

        def freeze_part_way(step):
            seen.append(step)
            if len(seen) == 3:
                wire.arm.is_frozen = True    # a servo hits its silent clamp
        # A stepped descent, so there are increments to freeze in the middle of:
        # inside the guard band, which sits just above the floor.
        stalled = arm.move_to(150.0, -150.0, Limits().z_floor + 2.0,
                              on_step=freeze_part_way)

    return all([
        check("the board claimed success for every command", is_all_accepted),
        check("an arm that never moved is reported as not responding",
              dead.status is MoveStatus.NOT_RESPONDING, str(dead)),
        check("an arm that moved and then stopped is blocked instead",
              stalled.status is MoveStatus.BLOCKED, str(stalled)),
        check("the stall was caught before the whole descent was spent",
              len(stalled.steps) < 20, f"{len(stalled.steps)} steps"),
    ])


def test_connect_refuses_a_bad_board() -> bool:
    """A board that cannot drive the arm must fail at connect, not at move 3.

    This is the lesson from a real session: the arm sat still while every move
    came back with a plausible status, and the run limped on for several stages
    before anyone concluded the hardware was at fault.
    """
    print("connect-time board check:")
    results = []
    faults = {
        "servo power off": lambda wire: setattr(wire.bus_servo, "voltage_mv", 4800),
        "servo bus silent": lambda wire: setattr(wire.bus_servo, "is_answering", False),
        "bus echoing": lambda wire: setattr(wire.bus_servo, "is_bus_echoing", True),
        "main.py half-built": lambda wire: wire.namespace.pop("nozzle"),
    }
    for label, break_it in faults.items():
        message = ""
        try:
            with fake_arm(break_board=break_it):
                pass
        except BoardNotReadyError as error:
            message = str(error)
        results.append(check(f"refused: {label}", bool(message), message[:72]))
    return all(results)


def test_stop_abandons_a_wait() -> bool:
    """A Ctrl-C landing in a wait must bite as hard as one landing in a move.

    The waits are the pump settling, the cup servo swinging, the board's valve
    hold and the homing routine -- half a second to nearly three each. Until
    now `stop()` ended only a *move*, so a Ctrl-C in one of them looked like
    nothing had happened at all, which is what made the scene script feel
    uninterruptible.
    """
    print("stopping during a wait:")
    with fake_arm() as (arm, _):
        tuning.SUCTION_SETTLE_MS = 600
        try:
            arm.stop()
            started = time.monotonic()
            arm.grip()
            cut_short_ms = (time.monotonic() - started) * 1000.0

            arm.move_to(*SETTLE_POSE)          # an explicit move re-arms the stop
            started = time.monotonic()
            arm.grip()
            honoured_ms = (time.monotonic() - started) * 1000.0
        finally:
            tuning.SUCTION_SETTLE_MS = 0

        # home() sends a board routine that cannot be called off; only the
        # waiting for it is ours, and that is what has to give way.
        threading.Timer(0.05, arm.stop).start()
        started = time.monotonic()
        homed = arm.home()
        home_ms = (time.monotonic() - started) * 1000.0

    return all([
        check("a stop cuts a pump wait short", cut_short_ms < 200,
              f"{cut_short_ms:.0f} ms of 600"),
        check("a new move re-arms it, so the next wait is honoured",
              honoured_ms >= 600, f"{honoured_ms:.0f} ms"),
        check("a stop cuts the homing wait short too", home_ms < 500,
              f"{home_ms:.0f} ms of {(HOME_DURATION_MS + 300)}"),
        check("and homing says it was stopped rather than reached",
              homed.status is MoveStatus.STOPPED, homed.status.value),
    ])


def test_connect_survives_a_dropped_read() -> bool:
    """One bad servo-bus answer must not condemn a healthy arm.

    About one connect in twenty was failing with "servo supply reads 0.00 V"
    on an arm that was fine. `get_vin()` gives up after 50 ms and the board has
    only just finished homing, so a single miss means nothing -- the same
    reasoning `read_position()` already applied and this check did not.
    """
    print("a dropped servo read at connect:")
    reads = []

    def flake_on_the_first_read(wire):
        answer_properly = wire.bus_servo.get_vin

        def get_vin(servo_id):
            reads.append(servo_id)
            return 0 if len(reads) == 1 else answer_properly(servo_id)

        wire.bus_servo.get_vin = get_vin

    try:
        with fake_arm(break_board=flake_on_the_first_read) as (arm, _):
            return all([
                check("connect survives one 0 V reading", arm.is_connected),
                check("because it read again rather than believing it",
                      len(reads) > 1, f"{len(reads)} reads"),
            ])
    except BoardNotReadyError as error:
        return check("connect survives one 0 V reading", False, str(error)[:72])


def test_nudges_a_small_miss() -> bool:
    """Landing 5 mm out is corrected by aiming past it, not by re-sending it.

    Re-commanding the same coordinate was measured to gain nothing on the real
    arm, so a correction has to mirror the error. The fake board undershoots by
    construction, which is what gives this something to correct.
    """
    print("nudging the last few millimetres:")
    with fake_arm() as (arm, wire):
        arm.move_to(*SETTLE_POSE)
        wire.arm.sag_per_mm = 0.06        # droop enough to miss by more than the tolerance
        target = (60.0, -240.0, 120.0)
        wire.arm.commands.clear()
        result = arm.move_to(*target)
        first_miss = wire.arm.sag_per_mm * (math.hypot(60.0, 240.0) - 150.0)
        corrections = [command for command in wire.arm.commands[1:]
                       if math.dist(command, target) > 0.1]
    return all([
        check("the first command would have missed",
              first_miss > tuning.ARRIVAL_TOLERANCE_MM, f"{first_miss:.1f} mm of droop"),
        check("so it was nudged", bool(corrections), f"{len(corrections)} corrections"),
        check("and the nudge aimed past the target, not at it",
              all(command[2] > target[2] for command in corrections),
              f"z {corrections[0][2]:.1f} for a target at {target[2]:.0f}"),
        check("which brought it inside tolerance",
              result.error_mm <= tuning.ARRIVAL_TOLERANCE_MM,
              f"{result.error_mm:.1f} mm from target"),
        check("without chasing it forever",
              len(corrections) <= tuning.CORRECTION_ATTEMPTS, f"{len(corrections)}"),
    ])


def test_stop_halts_a_move() -> bool:
    """`stop()` mid-flight must actually stop the arm, not wait it out."""
    print("stopping a move in flight:")
    with fake_arm() as (arm, wire):
        arm.move_to(*SETTLE_POSE)
        seen = []

        def watch(step):
            seen.append(step)
            if len(seen) == 2:
                arm.stop()

        wire.arm.command_delay_s = 0.01      # give the move time to be interrupted
        wire.arm.tracking_gain = 0.1         # and make it a move still under way
        wire.arm.commands.clear()
        result = arm.move_to(60.0, -240.0, 120.0, on_step=watch)
        halt = wire.arm.commands[-1]
    return all([
        check("the move reports stopped", result.status is MoveStatus.STOPPED, str(result)),
        check("a halt was commanded at the pose it had reached",
              math.dist(halt, result.position) < 5.0,
              f"halt at {_format(halt)}, arm at {_format(result.position)}"),
        check("it stopped short of the target", result.error_mm > 10.0,
              f"{result.error_mm:.0f} mm short"),
    ])


def test_nozzle_and_pick() -> bool:
    print("nozzle and pick-and-place:")
    with fake_arm() as (arm, wire):
        arm.set_nozzle_angle(45.0, duration_ms=1)
        arm.set_nozzle_angle(200.0, duration_ms=1)      # must clamp, not fault
        clamped = wire.nozzle.angle
        wire.nozzle.events.clear()
        result = arm.pick_at(180.0, -140.0, 55.0, approach_mm=25.0)
        picked_state = arm.get_state()
        arm.release(is_waited=False)
    return all([
        check("rotation is applied and clamped at 90", abs(clamped - 90.0) < 1e-9,
              f"{clamped}"),
        check("the pick completes", result.status is MoveStatus.REACHED, str(result)),
        check("suction went on before the descent", wire.nozzle.events[0] == "on",
              f"{wire.nozzle.events}"),
        check("and the arm lifted clear afterwards",
              result.position[2] > 70.0, f"z={result.position[2]:.0f}"),
        check("state tracks the suction", picked_state.is_suction_on),
        check("release opens the valve", wire.nozzle.is_on is False),
    ])


def test_the_pump_is_waited_for() -> bool:
    """`nozzle.on()` returns before the cup is holding anything.

    `brisk_tuning` zeroes this for every other check, so a real figure is put
    back here -- otherwise the one place the wait could silently disappear is
    the one place nothing is looking.
    """
    print("waiting for the pump:")
    with fake_arm() as (arm, wire):
        tuning.SUCTION_SETTLE_MS = 120
        try:
            started = time.monotonic()
            arm.grip()
            gripped_ms = (time.monotonic() - started) * 1000.0
            started = time.monotonic()
            arm.release(is_waited=False)
            released_ms = (time.monotonic() - started) * 1000.0
        finally:
            tuning.SUCTION_SETTLE_MS = 0
    return all([
        check("grip() blocks for the pump", gripped_ms >= 120.0, f"{gripped_ms:.0f} ms"),
        check("so does release(), valve hold or not", released_ms >= 120.0,
              f"{released_ms:.0f} ms"),
        check("the pump was switched both ways",
              [e for e in wire.nozzle.events if e in ("on", "off")] == ["on", "off"],
              str(wire.nozzle.events)),
        # A second vent pulse was tried here and measured worse on the arm, so
        # a release must send nothing but off(). protocol.VENT_CUP has the why.
        check("release drives the valve itself, and nothing else",
              not any(e.startswith("valve") for e in wire.nozzle.events),
              str(wire.nozzle.events)),
    ])


def test_state_is_pollable_during_a_move() -> bool:
    """A web or TUI front end has to read the pose while the arm is moving."""
    print("polling state from another thread:")
    with fake_arm() as (arm, wire):
        arm.move_to(80.0, -200.0, 120.0)
        wire.arm.command_delay_s = 0.004      # give the move some duration to poll
        samples = []
        is_running = threading.Event()

        def move():
            is_running.set()
            arm.move_to(220.0, -120.0, 60.0)

        worker = threading.Thread(target=move)
        worker.start()
        is_running.wait(1.0)
        deadline = time.time() + 5.0
        while worker.is_alive() and time.time() < deadline:
            started = time.time()
            samples.append(arm.get_state().position)
            if time.time() - started > 0.05:
                break                        # a blocked read is the failure here
            time.sleep(0.005)
        worker.join(5.0)
        distinct = len({tuple(sample) for sample in samples if sample})
    return all([
        check("the move thread finished", not worker.is_alive()),
        check("polls never blocked on the motion", len(samples) > 5, f"{len(samples)} polls"),
        check("and they saw the arm moving", distinct > 1, f"{distinct} distinct poses"),
    ])


def test_guard_rails() -> bool:
    print("guard rails:")
    detached = MaxArm()
    not_connected = detached.move_to(150.0, -150.0, 60.0)
    errors = {}
    for method in (ConnectionMethod.UART, ConnectionMethod.BLE):
        try:
            create_transport(ArmConfig(connection=method).resolved())
        except NotImplementedError as error:
            errors[method] = str(error)
    both = ""
    try:
        MaxArm(ArmConfig(), limits=Limits())
    except TypeError as error:
        both = str(error)
    return all([
        check("moving without connecting is a status, not a crash",
              not_connected.status is MoveStatus.NOT_CONNECTED, str(not_connected)),
        check("UART explains it needs a firmware swap",
              "firmware" in errors.get(ConnectionMethod.UART, "").lower()),
        check("BLE explains it has no readback",
              "readback" in errors.get(ConnectionMethod.BLE, "").lower()),
        check("a target can be checked with no board attached",
              detached.is_reachable((150.0, -150.0, 60.0))),
        check("a config and loose keywords together is a clear error", bool(both), both),
    ])


def _format(position) -> str:
    return "({:.1f}, {:.1f}, {:.1f})".format(*position)


def _round(values) -> str:
    return "(" + ", ".join(f"{value:.1f}" for value in values) + ")"


TESTS = [test_connect_and_state, test_move_and_refuse, test_safe_space_is_one_command,
         test_routes_around_the_base, test_desk_descent, test_desk_contact,
         test_accepted_but_motionless, test_connect_refuses_a_bad_board,
         test_connect_survives_a_dropped_read, test_stop_abandons_a_wait,
         test_nudges_a_small_miss,
         test_stop_halts_a_move, test_nozzle_and_pick,
         test_the_pump_is_waited_for,
         test_state_is_pollable_during_a_move, test_guard_rails]
