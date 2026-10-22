#!/usr/bin/env python3
"""`MaxArm` end to end against the fake board: what is sent, in what order, and what is waited for.

Most checks use a flat desk -- a desk.json that is the identity, so desk points
are the arm points the board receives and the numbers can be read off the log.
Waits are recorded instead of slept (`fake_arm.instant_waits`), with the
library's real constants, so the durations checked are the real ones.
"""

import contextlib
import json
import math
import signal
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maxarm import (BoardNotReadyError, MaxArm, MoveStatus, MoveType,  # noqa: E402
                    NotConnectedError, Space, Spaces)
from maxarm import arm as arm_module                           # noqa: E402
from maxarm import nozzle                                       # noqa: E402
from maxarm.geometry import HOME, bearing_degrees              # noqa: E402
from maxarm.path import REACH_MIN_MM                           # noqa: E402
from maxarm.protocol import ReplProtocol                       # noqa: E402
from maxarm.transport import SerialTransport                   # noqa: E402
import fake_arm as fake_arm_module                             # noqa: E402
from fake_arm import FakeSerial, fake_arm, instant_waits       # noqa: E402
from harness import BASELINE, check, with_raw_heights          # noqa: E402

FAST, SLOW = arm_module.MOVE_FAST_MM_S, arm_module.MOVE_SLOW_MM_S


@contextlib.contextmanager
def flat_desk(with_desk: bool = True):
    """A config directory whose desk is the arm's own frame -- or, `with_desk=False`, no desk at all."""
    with tempfile.TemporaryDirectory() as directory:
        if with_desk:
            Path(directory, "desk.json").write_text(json.dumps({"desk_to_arm": np.eye(4).tolist()}))
        yield Path(directory)


def ms(distance_mm: float, speed: float) -> int:
    return int(min(max(distance_mm / speed * 1000.0, arm_module.MOVE_MIN_MS), arm_module.MOVE_MAX_MS))


def test_connect_and_ask() -> bool:
    print("connecting, and asking without moving:")
    with fake_arm(connect=False) as (arm, board, _):
        reachable = arm.can_reach(100.0, -200.0, 60.0)
        unreachable = arm.can_reach(0.0, -290.0, 60.0)
        try:
            arm.move(100.0, -200.0, 60.0)
            is_refused = False
        except NotConnectedError:
            is_refused = True
        arm.connect()
        position = arm._position
    bad = FakeSerial()
    bad.bus_servo.voltage_mv = 1000
    try:
        with instant_waits():
            MaxArm(protocol=ReplProtocol(SerialTransport("/dev/fake", port_factory=lambda: bad))).connect()
        reason = ""
    except BoardNotReadyError as error:
        reason = str(error)
    return all([
        check("can_reach answers before connect()", reachable and not unreachable),
        check("a move before connect() raises", is_refused),
        check("after connect() the arm is where the board homed it", math.dist(position, HOME) < 1.5,
              f"{position}"),
        check("a board with no servo supply is refused at connect", "supply" in reason, reason[:60]),
    ])


def test_fast_and_slow() -> bool:
    print("fast and slow moves: one command each, timed by their speed:")
    with flat_desk() as config, fake_arm(config) as (arm, board, waits):
        arm.use_coordinates(Space.DESK)
        fast = arm.move(150.0, -150.0, 100.0)
        fast_log = list(board.log)
        board.log.clear()
        slow = arm.move(150.0, -150.0, 80.0, MoveType.SLOW)
        slow_log = list(board.log)
    fast_ms = ms(math.dist(HOME, (150.0, -150.0, 100.0)), FAST)
    return all([
        check("fast: straight there at 200 mm/s", fast and fast_log == [
            ("move", (150.0, -150.0, 100.0), fast_ms)], f"{fast_log}"),
        check("slow: 20 mm at 10 mm/s is 2 s", slow and slow_log == [
            ("move", (150.0, -150.0, 80.0), 2000)], f"{slow_log}"),
        check("each is waited out, nothing read back", waits == [fast_ms, 2000], f"{waits}"),
    ])


def test_precise() -> bool:
    print("PRECISE: servos with a deadband, corrected from their own reading:")
    with flat_desk() as config, fake_arm(config) as (arm, board, _):
        arm.use_coordinates(Space.DESK)
        arm.move(150.0, -150.0, 60.0)
        gain, fake_arm_module.TRACKING_GAIN = fake_arm_module.TRACKING_GAIN, 1.0
        arm.get_position()                                   # the servos are where they were sent
        board.arm.deadband_mm, board.arm.stop_short_mm = 1.5, 1.0
        try:
            plain = arm.move(150.0, -150.0, 59.0, MoveType.SLOW)
            ignored = board.arm.actual
            board.log.clear()
            precise = arm.move(150.0, -150.0, 58.0, MoveType.PRECISE)
            landed, sent = board.arm.actual, board.moves()
            board.arm.deadband_mm = 3.0
            board.log.clear()
            stuck = arm.move(150.0, -150.0, 57.0, MoveType.PRECISE)
            stuck_sent = board.moves()
        finally:
            fake_arm_module.TRACKING_GAIN = gain
    return all([
        check("a 1 mm slow step inside the deadband: the servos do not move", bool(plain)
              and abs(ignored[2] - 60.0) < 0.01, f"z {ignored[2]:.2f}"),
        check("PRECISE, 2 mm: stops 1 mm short, then sent the target plus that; lands",
              bool(precise) and len(sent) >= 2 and sent[0][2] == 58.0 and 56.5 < sent[1][2] < 57.5
              and abs(landed[2] - 58.0) < 0.4,
              f"sent {[round(point[2], 2) for point in sent]}, lands at z {landed[2]:.2f}"),
        check("a correction that gains nothing ends it, and the result is the target",
              bool(stuck) and len(stuck_sent) == 2 and stuck.position == (150.0, -150.0, 57.0),
              f"sent {[round(point[2], 2) for point in stuck_sent]}"),
    ])


def test_precise_at_the_reach_limit() -> bool:
    print(f"PRECISE at the {REACH_MIN_MM:.0f} mm reach limit: a correction may aim past it:")
    target = (0.0, -REACH_MIN_MM - 0.5, 60.0)
    with flat_desk() as config, fake_arm(config) as (arm, board, _):
        arm.use_coordinates(Space.DESK)
        arm.move(0.0, -150.0, 60.0)
        gain, fake_arm_module.TRACKING_GAIN = fake_arm_module.TRACKING_GAIN, 1.0
        arm.get_position()
        board.arm.deadband_mm, board.arm.stop_short_mm = 1.5, 3.0     # 1 mm in or out is a pulse here
        try:
            board.log.clear()
            result = arm.move(*target, MoveType.PRECISE)
            landed, sent = math.hypot(*board.arm.actual[:2]), board.moves()
        finally:
            fake_arm_module.TRACKING_GAIN = gain
    nearest = min(math.hypot(x, y) for x, y, _ in sent)
    return check("stopped 3 mm short of 95.5, sent 3 mm inside the limit to make up for it; lands",
                 bool(result) and nearest < REACH_MIN_MM and abs(landed - (REACH_MIN_MM + 0.5)) < 0.4,
                 f"sent as near as {nearest:.2f} mm, lands at {landed:.2f}")


def test_refusals_send_nothing() -> bool:
    print("a refused move sends nothing:")
    with flat_desk() as config, fake_arm(config) as (arm, board, _):
        arm.use_coordinates(Space.DESK)
        far = arm.move(0.0, -300.0, 60.0)
        base = arm.move(30.0, -50.0, 60.0)
        sunk = arm.move(100.0, -200.0, -20.0)
        log = list(board.log)
    return all([
        check("too far", far.status is MoveStatus.REFUSED and not far, far.detail),
        check("the arm-base limit", base.status is MoveStatus.REFUSED, base.detail),
        check("20 mm into the desk", sunk.status is MoveStatus.REFUSED, sunk.detail),
        check("and not one command reached the board", log == [], f"{log}"),
        check("the result says where the arm still is", math.dist(far.position, HOME) < 1.5),
    ])


def test_turn_rides_on_the_move() -> bool:
    print("a cup turn rides on the move, in the same round trip, and is waited for:")
    with flat_desk() as config, fake_arm(config) as (arm, board, waits):
        arm.use_coordinates(Space.DESK)
        before = board.executions
        arm.move(150.0, -150.0, 100.0, angle=0.0)
        trips = board.executions - before
        moved = list(board.log)
        board.log.clear()
        waits.clear()
        arm.move(150.0, -150.0, 100.0, angle=90.0)
        turned = list(board.log)
    move_ms = ms(math.dist(HOME, (150.0, -150.0, 100.0)), FAST)
    return all([
        check("angle 0 at bearing 45 is the cup at -45: square to the arm's axes",
              moved[0] == ("turn", -45.0, 300) and moved[1][0] == "move", f"{moved}"),
        check("one round trip for both", trips == 1, f"{trips}"),
        check("a turn in place: 90 degrees at 150 deg/s", turned == [("turn", 45.0, 600)], f"{turned}"),
        check("waited for, with the settle", waits == [600 + nozzle.NOZZLE_SETTLE_MS],
              f"{waits} (the move before took {move_ms})"),
    ])


def test_grab_and_drop() -> bool:
    print("grab and drop, as the PILOT lays them out:")
    with flat_desk() as config, fake_arm(config) as (arm, board, waits):
        arm.use_coordinates(Space.DESK)
        arm.set_clearance(14.0)
        grabbed = arm.grab(100.0, -200.0, 4.0)
        grab_log, grab_waits = list(board.log), list(waits)
        board.log.clear()
        waits.clear()
        dropped = arm.drop(-100.0, -200.0, 5.0)
        drop_log, drop_waits = list(board.log), list(waits)
    across = ms(math.dist(HOME, (100.0, -200.0, 23.0)), FAST)
    return all([
        check("grab: fast to 14 + 4 + 5 = 23 above it, slowly down, suction, slowly up",
              grabbed and grab_log == [
                  ("move", (100.0, -200.0, 23.0), across), ("move", (100.0, -200.0, 4.0), 1900),
                  ("suction", True), ("move", (100.0, -200.0, 23.0), 1900)], f"{grab_log}"),
        check("the grip is waited for", grab_waits == [across, 1900, 1500, 1900], f"{grab_waits}"),
        check("drop: slowly up 1 mm to 14 + 5 + 5, across, down, release, up",
              dropped and [entry[:2] for entry in drop_log] == [
                  ("move", (100.0, -200.0, 24.0)), ("move", (-100.0, -200.0, 24.0)),
                  ("move", (-100.0, -200.0, 5.0)), ("suction", False),
                  ("move", (-100.0, -200.0, 24.0))], f"{drop_log}"),
        check("the release is waited for, longer than the grip",
              drop_waits[3] == arm_module.SUCTION_RELEASE_MS > arm_module.SUCTION_GRAB_MS,
              f"{drop_waits}"),
    ])


def test_turning_a_piece() -> bool:
    print("turning a piece between grab and drop:")
    with flat_desk() as config, fake_arm(config) as (arm, board, _):
        arm.use_coordinates(Space.DESK)
        arm.set_clearance(14.0)
        arm.grab(100.0, -200.0, 4.0, angle=10.0)
        board.log.clear()
        arm.drop(-100.0, -200.0, 5.0, angle=10.0)
        kept = [entry for entry in board.log if entry[0] == "turn"]
        arm.grab(-100.0, -200.0, 5.0, angle=10.0)
        recentred = [entry for entry in board.log if entry[0] == "turn"][-1]
        board.log.clear()
        too_far = arm.drop(100.0, -200.0, 5.0, angle=10.0 + 150.0)    # cup at +97
        refused_log = list(board.log)
        is_still_held = arm._is_suction_on
    swing = 2 * math.degrees(math.atan2(100.0, 200.0))
    return all([
        check("same facing at another bearing: the cup takes up the base's turn",
              len(kept) == 1 and abs(kept[0][1] - swing) < 0.01, f"{kept}, base turned {swing:.2f}"),
        check("the next grab centres the cup on the way", recentred[1] == 0.0, f"{recentred}"),
        check("more than the cup can turn is refused", too_far.status is MoveStatus.REFUSED,
              too_far.detail),
        check("before anything was sent, still holding the piece",
              refused_log == [] and is_still_held, f"{refused_log}"),
    ])


def test_grab_lets_go_first_and_needs_clearance() -> bool:
    print("grab's preconditions:")
    with flat_desk() as config, fake_arm(config) as (arm, board, waits):
        arm.use_coordinates(Space.DESK)
        try:
            arm.grab(100.0, -200.0, 4.0)
            is_refused = False
        except RuntimeError as error:
            is_refused = "set_clearance" in str(error)
        arm.set_clearance(14.0)
        arm.set_suction(True)
        board.log.clear()
        waits.clear()
        arm.grab(100.0, -200.0, 4.0)
    return all([
        check("without a clearance it raises, naming set_clearance()", is_refused),
        check("with the suction on, it lets go first and waits for the release",
              board.log[0] == ("suction", False) and waits[0] == arm_module.SUCTION_RELEASE_MS,
              f"{board.log[0]}, {waits[0]}"),
    ])


def test_clearance_guard() -> bool:
    print("the clearance guard on a plain move:")
    with flat_desk() as config, fake_arm(config) as (arm, board, _):
        arm.use_coordinates(Space.DESK)
        arm.set_clearance(14.0)
        arm.move(150.0, -150.0, 4.0)
        down = board.moves()
        board.log.clear()
        arm.move(-150.0, -150.0, 4.0)
        across = board.moves()
        arm.grab(150.0, -150.0, 4.0)
        board.log.clear()
        arm.move(-150.0, -150.0, 4.0)
        carrying = board.moves()
        board.log.clear()
        arm.home()
        home = board.moves()
    return all([
        check("from high: across at 14 + 5, then down", down == [
            (150.0, -150.0, 19.0), (150.0, -150.0, 4.0)], f"{down}"),
        check("low to low: up, across, down", across == [
            (150.0, -150.0, 19.0), (-150.0, -150.0, 19.0), (-150.0, -150.0, 4.0)], f"{across}"),
        check("carrying a 4 mm piece: across at 14 + 4 + 5, where grab() left it", carrying == [
            (-150.0, -150.0, 23.0), (-150.0, -150.0, 4.0)], f"{carrying}"),
        check("home from low climbs first", home[0] == (-150.0, -150.0, 23.0)
              and home[-1] == HOME, f"{home}"),
    ])


def test_round_the_base() -> bool:
    print("a move across the base goes by its front corners:")
    with flat_desk() as config, fake_arm(config) as (arm, board, _):
        arm.use_coordinates(Space.DESK)
        arm.move(-150.0, 30.0, 60.0)
        board.log.clear()
        arm.move(150.0, 30.0, 60.0)
        legs = board.moves()
    return check("both front corners, then the target",
                 [(round(x), round(y)) for x, y, _ in legs] == [(-80, -80), (80, -80), (150, 30)],
                 f"{legs}")


def test_stop() -> bool:
    print("stop() and Ctrl-C:")
    with flat_desk() as config, fake_arm(config) as (arm, board, _):
        arm.use_coordinates(Space.DESK)
        arm.move(150.0, -150.0, 100.0)
        original = board.arm.set_position

        def stop_once_under_way(position, duration_ms):
            board.arm.set_position = original
            verdict = original(position, duration_ms)
            arm.stop()
            return verdict

        board.arm.set_position = stop_once_under_way
        board.log.clear()
        stopped = arm.move(-150.0, -150.0, 100.0)
        sent = list(board.log)
        halt = sent[-1][1]
        after = arm.move(150.0, -150.0, 100.0)
        previous = signal.getsignal(signal.SIGINT)
        try:
            arm.stop_on_ctrl_c()
            with contextlib.redirect_stderr(None):
                signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            is_second_default = signal.getsignal(signal.SIGINT) is signal.SIG_DFL
            board.log.clear()
            frozen = [arm.move(100.0, -200.0, 60.0), arm.set_suction(True), arm.home()]
            frozen_log = list(board.log)
        finally:
            signal.signal(signal.SIGINT, previous)
    return all([
        check("a stop during a leg holds the arm part-way", stopped.status is MoveStatus.STOPPED
              and -150.0 < halt[0] < 150.0, f"{stopped}"),
        check("by sending it, quickly, to the pose its servos read back",
              len(sent) == 2 and sent[-1][2] == arm_module.MOVE_MIN_MS
              and math.dist(halt, board.arm.actual) < 1.0, f"{sent}"),
        check("the next call goes ahead", bool(after), f"{after}"),
        check("after Ctrl-C every call stops at once and sends nothing",
              all(result.status is MoveStatus.STOPPED for result in frozen) and frozen_log == [],
              f"{frozen_log}"),
        check("and a second Ctrl-C would quit", is_second_default),
    ])


def test_camera_space() -> bool:
    print("camera space, through elephant's calibration:")
    spaces = Spaces(BASELINE)
    with fake_arm(BASELINE) as (arm, board, _):
        arm.use_coordinates(Space.CAMERA)
        arm.set_clearance(14.0)
        px, py, _ = spaces.convert((20.0, -180.0, 4.0), Space.DESK, Space.CAMERA)
        result = arm.grab(px, py, 4.0)
        descent = board.moves()[1]
        expected = spaces.convert((px, py, 4.0), Space.CAMERA, Space.ARM)
        desk = spaces.convert((px, py, 4.0), Space.CAMERA, Space.DESK)
        lifted = spaces.convert((desk[0], desk[1], result.position[2]), Space.DESK, Space.CAMERA)
    return all([
        check("a grab at a pixel and a height", bool(result), f"{result}"),
        check("comes down onto that point of the desk, as the arm has it",
              math.dist(descent, expected) < 0.01, f"{descent} vs {tuple(round(v, 2) for v in expected)}"),
        check("and the result is the pixel it is seen at, at the height it was lifted to",
              math.dist(result.position, lifted) < 1e-6 and math.dist(result.position[:2], (px, py)) > 5.0,
              f"{tuple(round(v, 1) for v in result.position)}, picked at ({px:.1f}, {py:.1f})"),
    ])


def test_without_a_desk_calibration() -> bool:
    print("without desk.json: the default desk, arm z 44:")
    with flat_desk(with_desk=False) as config, fake_arm(config) as (arm, board, _):
        arm.use_coordinates(Space.DESK)
        arm.set_clearance(14.0)
        arm.move(150.0, -150.0, 0.0)
        on_desk = board.moves()[-1]
        board.log.clear()
        arm.move(-150.0, -150.0, 0.0)
        low = board.moves()
    with fake_arm(None) as (arm, board, _):
        arm.set_clearance(14.0)
        arm.move(150.0, -150.0, 44.0)
        board.log.clear()
        arm.move(-150.0, -150.0, 44.0)
        unguarded = board.moves()
    return all([
        check("desk z 0 is sent as arm z 44", on_desk[2] == 44.0, f"{on_desk}"),
        check("a low move climbs over the clearance: 44 + 14 + 5", [z for _, _, z in low] == [63.0, 63.0, 44.0],
              f"{low}"),
        check("with no calibration at all, none: it goes straight", unguarded == [(-150.0, -150.0, 44.0)],
              f"{unguarded}"),
    ])


def test_switching_calibration() -> bool:
    print("use_calibration(): measure the desk with none, then try the one just written:")
    with flat_desk() as config, fake_arm(None) as (arm, board, _):
        below = arm.move(100.0, -200.0, -20.0)
        try:
            arm.use_coordinates(Space.DESK)
            is_desk_refused = False
        except FileNotFoundError:
            is_desk_refused = True
        arm.use_calibration(config)
        arm.use_coordinates(Space.DESK)
        sunk = arm.move(100.0, -200.0, -20.0)
        try:
            arm.use_calibration(None)
            is_kept = False
        except FileNotFoundError:
            is_kept = arm.move(100.0, -200.0, 10.0).position == (100.0, -200.0, 10.0)
    return all([
        check("no calibration: arm z -20 is the firmware's business, not a desk floor", bool(below),
              f"{below}"),
        check("and desk coordinates are refused", is_desk_refused),
        check("with the file: the desk floor is back", sunk.status is MoveStatus.REFUSED, sunk.detail),
        check("dropping it while in desk coordinates raises and keeps it", is_kept),
    ])


def test_cup_to_either_end() -> bool:
    print("the cup turns all the way to -90 and +90, wherever the arm is:")
    refused = []
    with fake_arm(None) as (arm, board, _):
        for x, y in [(77.0, -240.0), (196.0, -140.0), (-206.0, -123.0), (142.0, -202.0), (210.0, 49.0)]:
            arm.move(x, y, 120.0)
            for servo in (-90.0, 90.0):
                result = arm.move(x, y, 120.0, angle=servo + bearing_degrees(x, y))
                if not result:
                    refused.append(f"({x:.0f}, {y:.0f}) {servo:+.0f}: {result.detail}")
    return check("none refused", not refused, "; ".join(refused))


def test_small_moves_are_sent() -> bool:
    print("a move of a tenth of a millimetre is sent, not skipped:")
    with fake_arm(None) as (arm, board, _):
        arm.move(100.0, -200.0, 60.0)
        board.log.clear()
        arm.move(100.0, -200.0, 59.9)
        tenth = list(board.log)
        board.log.clear()
        arm.move(100.0, -200.0, 59.904)
        nothing = list(board.log)
    return all([
        check("0.1 mm: one command, at the shortest duration",
              tenth == [("move", (100.0, -200.0, 59.9), arm_module.MOVE_MIN_MS)], f"{tenth}"),
        check("under the command's two decimals: nothing", nothing == [], f"{nothing}"),
    ])


TESTS = [with_raw_heights(test) for test in [
    test_connect_and_ask, test_fast_and_slow, test_precise, test_precise_at_the_reach_limit,
    test_refusals_send_nothing,
    test_turn_rides_on_the_move, test_grab_and_drop, test_turning_a_piece,
    test_grab_lets_go_first_and_needs_clearance, test_clearance_guard, test_round_the_base,
    test_stop, test_camera_space, test_without_a_desk_calibration, test_switching_calibration,
    test_cup_to_either_end, test_small_moves_are_sent]]
