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
from pathlib import Path

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
from desk.api import DeskSession                             # noqa: E402
from fake_arm import fake_pyserial, instant_waits            # noqa: E402
from harness import BASELINE, check                          # noqa: E402

CMD_MM = 0.005
# Five touch points round the work area on z = 0.01 x - 0.05 y + 30, the last 1.5 mm above it.
TOUCHES = [(-100.0, -250.0, 41.5), (100.0, -250.0, 43.5), (0.0, -100.0, 35.0), (140.0, -120.0, 37.4),
           (-140.0, -140.0, 37.1)]


@contextlib.contextmanager
def app(with_desk: bool = True):
    """Yields post(path, **body) -> reply, state() -> dict, the fake board, and the scratch config."""
    with tempfile.TemporaryDirectory() as scratch, fake_pyserial() as boards, instant_waits():
        config = Path(scratch)
        if with_desk:
            shutil.copy(BASELINE / "desk.json", config)
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
        heights = [Spaces(config).convert(point, Space.ARM, Space.DESK)[2] for point in TOUCHES]
        after = state()["desk"]
    offsets = [point["deviation"] for point in desk["points"]]
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


def test_refusals_to_save() -> bool:
    print("points SAVE will not take:")
    reasons = []
    for touches in ([(-100.0, -250.0, 40.0), (0.0, -175.0, 40.0), (100.0, -100.0, 40.0), (50.0, -137.5, 40.0)],
                    [(-100.0, -250.0, 40.0), (100.0, -250.0, 40.0), (0.0, -150.0, 40.0)]):
        with app() as (post, _, _, _):
            for x, y, z in touches:
                post("jump", x=x, y=y)
                post("move", x=x, y=y, z=z)
                post("desk/record")
            reasons.append(post("desk/save"))
    return all([
        check("all in a row, though spanning 200 by 150", reasons[0]["status"] == "refused"
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
              and math.dist(approach[-1][0][:2], TOUCHES[0][:2]) < CMD_MM,
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


def test_without_desk_json() -> bool:
    print("no desk.json yet:")
    with app(with_desk=False) as (post, state, _, _):
        record_touches(post)
        before = post("desk/test-point", id=1, height=4.0)
        post("desk/save")
        after = post("desk/test-point", id=1, height=4.0)
    return all([
        check("a desk test is refused, saying what to do", before["status"] == "refused"
              and "SAVE" in before["detail"], before["detail"]),
        check("and works once SAVE has written one", after["status"] == "done", after["detail"]),
    ])


TESTS = [test_arm_space_moves, test_record_and_save, test_refusals_to_save, test_point_tests,
         test_input_and_reset, test_without_desk_json]
