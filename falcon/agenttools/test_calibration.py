#!/usr/bin/env python3
"""The calibration app against the fake board: what each request sends the arm, SAVE, and the desk tests.

No web server and no browser: the handlers are called as the server calls
them, the library's waits are recorded instead of slept, and config/ is a
scratch directory each time. The board is sent two decimals, hence `CMD_MM`.
"""

import contextlib
import json
import math
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "calibration"), str(HERE)]

from maxarm import MaxArm, Space, Spaces                     # noqa: E402
from maxarm.arm import MOVE_FAST_MM_S, MOVE_SLOW_MM_S        # noqa: E402
from maxarm.geometry import HOME                             # noqa: E402
from common import arm_api                                   # noqa: E402
from common.motion import GLIDE_STEP_MM, HOVER_MM, Motion    # noqa: E402
from common.server import Server                             # noqa: E402
from common.worker import Worker                             # noqa: E402
from desk import store                                       # noqa: E402
from desk.plane import Plane                                  # noqa: E402
from desk.store import Point                                  # noqa: E402
from desk.api import DeskSession                             # noqa: E402
import fake_arm                                              # noqa: E402
from fake_arm import fake_pyserial, instant_waits            # noqa: E402
from harness import BASELINE, check, with_raw_heights        # noqa: E402

CMD_MM = 0.005
# Five touch points round the work area on z = 0.01 x - 0.05 y + 30, the last 1.5 mm above it. They span
# more than SAVE needs: a point is where the servos say the cup is, to the nearest pulse.
TOUCHES = [(-100.0, -250.0, 41.5), (100.0, -250.0, 43.5), (0.0, -95.0, 34.75), (140.0, -120.0, 37.4),
           (-140.0, -140.0, 37.1)]


@contextlib.contextmanager
def app(with_desk: bool = True, prepare: Optional[Callable[[Path], None]] = None):
    """Yields post(path, **body) -> reply, state() -> dict, the fake board, and the scratch config.

    `prepare` writes files into the scratch config before the app starts."""
    gain, fake_arm.TRACKING_GAIN = fake_arm.TRACKING_GAIN, 1.0       # the servos get where they are sent
    with tempfile.TemporaryDirectory() as scratch, fake_pyserial() as boards, instant_waits():
        config = Path(scratch)
        if with_desk:
            shutil.copy(BASELINE / "desk.json", config)
        if prepare:
            prepare(config)
        worker = Worker(Motion(MaxArm(config_dir=None), config))
        server = Server(ROOT / "calibration", home="desk")
        arm_api.register(server, worker)
        DeskSession(worker, config).register(server)
        worker.thread.start()
        worker.connect()
        boards[0].log.clear()

        def post(path: str, **body) -> dict:
            return server.routes[("POST", f"/api/{path}")](body)

        def state() -> dict:
            return json.loads(json.dumps(server.routes[("GET", "/api/state")]({})))

        try:
            yield post, state, boards[0], config
        finally:
            worker.shutdown()
            fake_arm.TRACKING_GAIN = gain


def legs(board) -> list:
    """The moves sent, as (point, ms), and the log cleared."""
    sent = [(entry[1], entry[2]) for entry in board.log if entry[0] == "move"]
    board.log.clear()
    return sent


def speed(start, leg) -> float:
    point, ms = leg
    return math.dist(start, point) / ms * 1000.0


def record_touches(post) -> None:
    for x, y, z in TOUCHES:
        post("jump", x=x, y=y)
        post("move", x=x, y=y, z=z)
        post("desk/record")


def test_arm_space_moves() -> bool:
    print("arm space: fast high up, slow low down, and no desk floor in the way:")
    with app() as (post, state, board, _):
        post("move", x=100.0, y=-200.0, z=150.0)
        legs(board)
        down = post("move", x=100.0, y=-200.0, z=40.0)
        descent = legs(board)
        jump = post("jump", x=-100.0, y=-200.0)
        over = legs(board)
        deep = post("move", x=-100.0, y=-200.0, z=20.0)
        sent = state()["sent"]
    hover = HOVER_MM[Space.ARM]
    return all([
        check("down through the hover height: fast to it, then slow",
              down["status"] == "done" and [point[2] for point, _ in descent] == [hover, 40.0]
              and abs(speed((100, -200, 150), descent[0]) - MOVE_FAST_MM_S) < 1
              and abs(speed(descent[0][0], descent[1]) - MOVE_SLOW_MM_S) < 0.1, f"{descent}"),
        check("a jump from low: slowly straight up, then fast across, and it stays up",
              jump["status"] == "done" and over[0][0] == (100.0, -200.0, hover)
              and over[-1][0] == (-100.0, -200.0, hover)
              and abs(speed((100, -200, 40), over[0]) - MOVE_SLOW_MM_S) < 0.1, f"{over}"),
        check("20 mm under where desk.json has the desk: allowed, the desk is being measured",
              deep["status"] == "done" and sent == [-100.0, -200.0, 20.0], f"{deep['detail']}"),
    ])


def test_record_and_save() -> bool:
    print("touch points, the plane through them, and SAVE:")
    with app() as (post, state, board, config):
        post("jump", x=-100.0, y=-250.0)
        post("move", x=-100.0, y=-250.0, z=41.5)
        post("desk/record")
        early = post("desk/save")
        record_touches(post)
        post("desk/delete", id=1)
        desk = state()["desk"]
        saved = post("desk/save")
        written = json.loads((config / "desk.json").read_text())
        is_kept = (config / "desk-previous.json").exists()
        recorded = [(point["x"], point["y"], point["z"]) for point in desk["points"]]
        heights = [Spaces(config).convert(point, Space.ARM, Space.DESK)[2] for point in recorded]
        after = state()["desk"]
    offsets = [point["off"]["fit"] for point in desk["points"]]
    worst = max(range(len(offsets)), key=lambda index: abs(offsets[index]))
    return all([
        check("one point: SAVE refused, saying why", early["status"] == "refused", early["detail"]),
        check("five points, numbered on from the deleted one", [point["id"] for point in desk["points"]]
              == [2, 3, 4, 5, 6], f"{[point['id'] for point in desk['points']]}"),
        check("the odd point out is the one furthest off the plane", worst == 4,
              f"{[round(offset, 2) for offset in offsets]}"),
        check("SAVE writes desk.json and keeps the one it replaced", saved["status"] == "done" and is_kept,
              saved["detail"]),
        check("x and y as they are; the plane in z", written["desk_to_arm"][0] == [1.0, 0.0, 0.0, 0.0]
              and written["desk_to_arm"][1] == [0.0, 1.0, 0.0, 0.0] and len(written["points"]) == 5),
        check("through the library, each point sits at its own offset from desk height 0",
              all(abs(height - offset) < 1e-9 for height, offset in zip(heights, offsets)),
              f"{[round(height, 2) for height in heights]}"),
        check("and the page hears of it", "5 touch points" in after["saved"]["summary"], after["saved"]["summary"]),
    ])


def test_numbers_and_readings() -> bool:
    print("a point is what the servos read, and a deleted point's number is not used again:")
    with app() as (post, state, board, _):
        record_touches(post)
        post("desk/delete", id=5)
        post("jump", x=60.0, y=-200.0)
        post("move", x=60.0, y=-200.0, z=40.0)
        board.arm.position = board.arm.actual = (60.0, -200.0, 41.3)   # settled short of the 40 it was sent to
        recorded = post("desk/record")
        points = state()["desk"]["points"]
        post("desk/clear")
        post("desk/record")
        after_clear = [point["id"] for point in state()["desk"]["points"]]
    newest = points[-1]
    return all([
        check("after deleting #5, the next is #6", [point["id"] for point in points] == [1, 2, 3, 4, 6],
              f"{[point['id'] for point in points]}"),
        check("recorded where the servos are, with where it was sent beside it",
              abs(newest["z"] - 41.3) < 0.7 and newest["sent"] == [60.0, -200.0, 40.0], recorded["detail"]),
        check("CLEAR starts the numbers again", after_clear == [1], f"{after_clear}"),
    ])


def test_refusals_to_save() -> bool:
    print("points SAVE will not take:")
    reasons = []
    for touches in ([(-110.0, -245.0, 40.0), (0.0, -167.5, 40.0), (110.0, -90.0, 40.0), (55.0, -128.75, 40.0)],
                    [(-100.0, -250.0, 40.0), (100.0, -250.0, 40.0), (0.0, -150.0, 40.0)]):
        with app() as (post, _, _, _):
            for x, y, z in touches:
                post("jump", x=x, y=y)
                post("move", x=x, y=y, z=z)
                post("desk/record")
            reasons.append(post("desk/save"))
    return all([
        check("all in a row, though spanning 220 by 155", reasons[0]["status"] == "refused"
              and "row" in reasons[0]["detail"], reasons[0]["detail"]),
        check("only 100 mm of Y", reasons[1]["status"] == "refused" and "in Y" in reasons[1]["detail"],
              reasons[1]["detail"]),
    ])


def test_point_tests() -> bool:
    print("point tests in desk space: from above first, then gliding while nothing else moves the arm:")
    with app() as (post, state, board, config):
        record_touches(post)
        post("desk/save")
        matrix = json.loads((config / "desk.json").read_text())["desk_to_arm"][2]

        def height(point) -> float:
            return point[2] - (matrix[0] * point[0] + matrix[1] * point[1] + matrix[3])

        legs(board)
        first = post("desk/test-point", id=1, height=4.0)
        approach = legs(board)
        is_gliding = state()["desk"]["is_gliding"]
        glide = post("desk/test-point", id=2, height=4.0)
        level = legs(board)
        lowered = post("desk/test-height", height=2.0)
        down = legs(board)
        post("desk/test-point", id=4, height=2.0)
        low = legs(board)
        post("suction", on=True)
        board.log.clear()
        again = post("desk/test-point", id=1, height=2.0)
        over = list(board.log)
    return all([
        check("first: up to the hover height, over, and slowly down to 4 mm",
              first["status"] == "done" and abs(height(approach[0][0]) - HOVER_MM[Space.DESK]) < CMD_MM
              and abs(height(approach[-1][0]) - 4.0) < CMD_MM
              and math.dist(approach[-1][0][:2], TOUCHES[0][:2]) < 0.5,
              f"{len(approach)} legs, heights {[round(height(point), 1) for point, _ in approach]}"),
        check("the page is told the next one glides", is_gliding),
        check("second: a glide, in legs of 10 mm at most, every one at 4.0 mm, all slow",
              glide["status"] == "done" and len(level) > 1
              and all(abs(height(point) - 4.0) < CMD_MM for point, _ in level)
              and all(math.dist(a[0], b[0]) <= GLIDE_STEP_MM + CMD_MM for a, b in zip(level, level[1:]))
              and all(abs(speed(a[0], b) - MOVE_SLOW_MM_S) < 0.2 for a, b in zip(level, level[1:])),
              f"{len(level)} legs"),
        check("a new test height moves straight down, and keeps the glide",
              lowered["status"] == "done" and len(down) == 1 and abs(height(down[0][0]) - 2.0) < CMD_MM
              and len(low) > 1 and all(abs(height(point) - 2.0) < CMD_MM for point, _ in low)),
        check("suction switched in between: let go first, then from above again",
              again["status"] == "done" and over[0] == ("suction", False)
              and height(over[1][1]) > 15.0, f"{over[:3]}"),
    ])


def test_input_and_reset() -> bool:
    print("TEST INPUT, a desk move, and RESET:")
    with app() as (post, state, board, config):
        record_touches(post)
        post("desk/save")
        matrix = json.loads((config / "desk.json").read_text())["desk_to_arm"][2]
        surface = matrix[0] * 50.0 + matrix[1] * -180.0 + matrix[3]
        board.log.clear()
        tested = post("desk/test-input", x=50.0, y=-180.0, z=0.0, hold=0.0)
        log = list(board.log)
        board.log.clear()
        nudged = post("desk/move", x=50.0, y=-180.0, z=-0.5)
        nudge = legs(board)
        post("cup", angle=30.0)
        post("suction", on=True)
        board.log.clear()
        reset = post("reset")
        home = list(board.log)
        after = state()
    suction = [entry for entry in log if entry[0] == "suction"]
    last_move = [entry for entry in log if entry[0] == "move"][-1]
    return all([
        check("down onto the desk point, grip, let go, and stay",
              tested["status"] == "done" and suction == [("suction", True), ("suction", False)]
              and abs(last_move[1][2] - surface) < 0.01 and log[-1] == ("suction", False),
              f"lands at arm z {last_move[1][2]:.2f}, the plane says {surface:.2f}"),
        check("a desk move under it: straight down, slowly",
              nudged["status"] == "done" and len(nudge) == 1 and nudge[0][1] == 150),
        check("RESET: let go, slowly up to the hover height, then home with the cup centred",
              reset["status"] == "done" and home[0] == ("suction", False)
              and home[1][0] == "move" and home[1][1][2] == HOVER_MM[Space.ARM]
              and any(entry[:2] == ("turn", 0.0) for entry in home) and home[-1][:2] == ("move", HOME),
              f"{home}"),
        check("and the page shows it there", after["sent"] == list(HOME) and after["cup"] == 0.0
              and not after["suction"]),
    ])


def firmware_z(x: float, y: float, true_z: float) -> float:
    """The raw z the board should get for a true z: the PILOT's formula, written out here and not taken
    from the library, so that this checks it."""
    out = math.hypot(x, y) - 207.0
    height = max(true_z - 44.0, 0.0)
    return true_z - 0.05 * out + (height / 120.0) ** 2 * (3.4 + 0.0126 * out)


def test_true_heights() -> bool:
    """What the page sends, against the formula. Nothing here trusts the fake board's readings."""
    print("true heights: the page's arm z 44 and its desk tests, as the firmware is sent them:")
    desk_z = 44.0
    spots = [(-100.0, -245.0), (0.0, -100.0), (130.0, -100.0), (-180.0, -160.0), (100.0, -245.0)]
    points = [Point(index + 1, x, y, desk_z, None) for index, (x, y) in enumerate(spots)]

    def level_desk(config: Path) -> None:
        store.save_points(config, points, len(points) + 1)
        store.save_desk(config, Plane(0.0, 0.0, desk_z), points)

    with app(with_desk=False, prepare=level_desk) as (post, _, board, _):
        misses, jumps = [], []
        for x, y in spots:
            post("jump", x=x, y=y)
            jumps.append(legs(board))
            post("move", x=x, y=y, z=desk_z)
            misses.append(legs(board)[-1][0][2] - firmware_z(x, y, desk_z))
        post("desk/test-point", id=1, height=4.0)
        legs(board)
        post("desk/test-point", id=2, height=4.0)
        glide = legs(board)
    over = [z - firmware_z(x, y, desk_z) for (x, y, z), _ in glide]
    crossing = [z - firmware_z(x, y, 54.0) for jump in jumps[1:] for (x, y, z), _ in jump]
    return all([
        check("arm z 44 sends the firmware's own height for it, at every reach",
              max(abs(miss) for miss in misses) < CMD_MM, f"worst {max(abs(miss) for miss in misses):.3f} mm"),
        check("a grid click from the desk goes up to true z 54 and across at it: raw z for each end",
              len(crossing) >= 2 * (len(spots) - 1) and max(abs(miss) for miss in crossing) < CMD_MM,
              f"{len(crossing)} legs, worst {max(abs(miss) for miss in crossing):.3f} mm"),
        check("a glide over a level desk at true 44 is sent 4 mm over it in raw z, leg by leg",
              len(glide) > 10 and max(abs(height - 4.0) for height in over) <= 2 * CMD_MM,
              f"{min(over):.3f} .. {max(over):.3f} mm over {len(glide)} legs"),
    ])


def test_files_from_before() -> bool:
    print("touch points kept as the servos read them, and files written before:")
    stack = (0.0, -239.52, 167.81)                   # the owner's #77, in true z by reach alone
    with tempfile.TemporaryDirectory() as scratch:
        config = Path(scratch)
        (config / store.POINTS_FILE).write_text(json.dumps({"next_id": 2, "points": [
            {"id": 1, "x": 0.0, "y": -260.0, "z": 41.85, "sent": [0.0, -260.0, 43.0]}]}))
        points, _ = store.load_points(config)
        store.save_points(config, points, 2)
        kept = json.loads((config / store.POINTS_FILE).read_text())
        again, _ = store.load_points(config)
        (config / store.POINTS_FILE).write_text(json.dumps({"heights": "true", "next_id": 3, "points": [
            {"id": 1, "x": 0.0, "y": -260.02, "z": 44.5, "sent": None},
            {"id": 2, "x": stack[0], "y": stack[1], "z": stack[2], "sent": list(stack)}]}))
        first_day, _ = store.load_points(config)
        desk = json.loads((BASELINE / "desk.json").read_text())
        desk["joint_offsets_deg"] = [0.0, 0.0, -4.17]
        (config / store.DESK_FILE).write_text(json.dumps(desk))
        spaces, described = Spaces(config), store.describe_desk(config)
        refusal = spaces.ignored_desk or ""
        default = spaces.convert((0.0, -200.0, 44.0), Space.ARM, Space.DESK)
    by_reach_alone = stack[2] - 0.05 * (math.hypot(*stack[:2]) - 207.0)
    return all([
        check("touch points in raw z are read as true z",
              abs(points[0].z - 44.5) < 0.001 and abs(points[0].sent[2] - 45.65) < 0.001, f"{points[0]}"),
        check("and kept as the servos read them, raw", kept["heights"] == "raw"
              and kept["points"][0]["z"] == 41.85 and kept["points"][0]["sent"][2] == 43.0, f"{kept['points']}"),
        check("read back as they were", again == points),
        check("a file in true z by reach alone: a desk touch reads as it did",
              abs(first_day[0].z - 44.5) <= 0.01, f"{first_day[0]}"),
        check("and a stack touch is the same raw reading, through the formula as it is now",
              abs(firmware_z(*first_day[1].target) - by_reach_alone) < 0.006
              and abs(firmware_z(*first_day[1].sent) - by_reach_alone) < 0.006,
              f"{first_day[1].z} true, raw {by_reach_alone:.2f}"),
        check("a desk.json fitted with the elbow offset is not used: the default desk is",
              not spaces.is_desk_saved and "SAVE" in refusal and abs(default[2]) < 1e-12, refusal),
        check("and the page says to SAVE again", not described["exists"] and "SAVE" in described["summary"],
              described["summary"]),
    ])


def test_without_desk_json() -> bool:
    print("no desk.json yet: the desk tests use the default desk, arm z 44:")
    with app(with_desk=False) as (post, state, board, _):
        record_touches(post)
        legs(board)
        before = post("desk/test-point", id=1, height=4.0)
        landed = legs(board)[-1][0]
        post("desk/save")
        after = post("desk/test-point", id=1, height=4.0)
    return all([
        check("a desk test works, 4 mm over arm z 44", before["status"] == "done" and abs(landed[2] - 48.0) < CMD_MM,
              f"{before['detail']} {landed}"),
        check("and once SAVE has written one", after["status"] == "done", after["detail"]),
    ])


def test_offsets() -> bool:
    print("each point's offset three ways: off the fit of the points, the saved desk.json, the default desk:")
    with app(with_desk=False) as (post, state, _, _):
        record_touches(post)
        before = state()["desk"]
        post("desk/save")
        after = state()["desk"]
    default = [point["z"] - 44.0 for point in before["points"]]       # the default desk, written out here
    fits = [point["off"]["fit"] for point in after["points"]]
    return all([
        check("off the default desk: z - 44, for every point",
              all(abs(point["off"]["default"] - offset) < 1e-9 for point, offset in zip(before["points"], default))
              and abs(before["off"]["default"]["worst"] - max(map(abs, default))) < 1e-9),
        check("off the saved desk: none before SAVE", before["off"]["saved"] is None
              and all(point["off"]["saved"] is None for point in before["points"])),
        check("after SAVE the saved desk is the fit, point by point",
              all(abs(point["off"]["saved"] - fit) < 1e-9 for point, fit in zip(after["points"], fits))
              and abs(after["off"]["saved"]["rms"] - after["off"]["fit"]["rms"]) < 1e-9,
              f"{[round(fit, 2) for fit in fits]}"),
    ])


def test_reading_after_moves() -> bool:
    print("what the arm says is out of date from a move until the servos are read, a SAVE queued behind or not:")
    with tempfile.TemporaryDirectory() as scratch, fake_pyserial(), instant_waits():
        worker = Worker(Motion(MaxArm(config_dir=None), Path(scratch)))
        reads = []
        read = worker.motion.read
        worker.motion.read = lambda: (reads.append(worker.motion.sent), read())
        worker.thread.start()
        worker.connect()
        gate = threading.Event()
        replies = {}

        def queue(name: str, action, uses_arm: bool) -> threading.Thread:
            thread = threading.Thread(target=lambda: replies.update(
                {name: worker.run(name, name, action, uses_arm=uses_arm)}))
            thread.start()
            return thread

        threads = [queue("hold", lambda motion: gate.wait() and None, False)]
        while worker.doing != "hold":
            time.sleep(0.001)
        reads.clear()
        threads.append(queue("move", lambda motion: motion.direct(Space.ARM, (100.0, -200.0, 120.0)), True))
        while len(worker.queue) < 1:
            time.sleep(0.001)
        threads.append(queue("save", lambda motion: str(worker.state()["is_reading"]), False))
        while len(worker.queue) < 2:
            time.sleep(0.001)
        gate.set()
        for thread in threads:
            thread.join(timeout=10.0)
        while worker.doing is not None:
            time.sleep(0.001)
        after = worker.state()
        worker.run("alone", "alone", lambda motion: None, uses_arm=False)
        alone = len(reads)
        worker.shutdown()
    return all([
        check("a SAVE right behind a move: still out of date while it runs", replies["save"]["detail"] == "True",
              f"{replies}"),
        check("then read once, after the move, and up to date", reads == [(100.0, -200.0, 120.0)]
              and not after["is_reading"], f"reads {reads}"),
        check("a command that does not use the arm reads nothing", alone == 1),
    ])


TESTS = [with_raw_heights(test) for test in [
    test_arm_space_moves, test_record_and_save, test_numbers_and_readings, test_refusals_to_save, test_point_tests,
    test_input_and_reset, test_without_desk_json]] + [
    test_true_heights, test_files_from_before, test_offsets, test_reading_after_moves]
