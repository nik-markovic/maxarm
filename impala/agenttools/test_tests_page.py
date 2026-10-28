#!/usr/bin/env python3
"""The tests page against the fake board and a fake camera: FIND RED CUBE, ARM TO DESK, the calibration the page
draws by, the red cube's points that refine it, and the red cube finder on its own.

The scenes are drawn through a known camera (`cube_scene.py`): the page's CALIBRATE is fitted to one, then the red
cube is moved and found again in the live view, 800 x 600, through that calibration. What is checked is what is
sent to the arm and whether a cube that is there to find is found where it was drawn, never anything about a real
camera or arm. Heights are the firmware's own (`with_raw_heights`), the desk level at arm z 44, as in
`test_camera.py`, whose app this is.
"""

import contextlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "calibration"), str(HERE)]

from maxarm.arm import MOVE_FAST_MM_S, MOVE_SLOW_MM_S         # noqa: E402
from maxarm.geometry import HOME                              # noqa: E402
from camera import store as camera_store                      # noqa: E402
from camera import feed as feed_module                        # noqa: E402
from camera.cubes import CubesNotFound                        # noqa: E402
from camera.feed import VIEW_SIZE, Feed                       # noqa: E402
from camera.check import difference_mm                       # noqa: E402
from camera.fit import HANDEDNESS, CameraFit, CameraNotFound, fit_camera  # noqa: E402
from camera.locate import find_red_cube                       # noqa: E402
from camera.model import BASE_DIAGONALS, project              # noqa: E402
from camera.refine import Sighting, refine                    # noqa: E402
from tests.api import FIND_OVER_MM, TOUCH_MM, TOUCH_OVER_MM   # noqa: E402
import cube_scene                                             # noqa: E402
from fake_camera import fake_camera                           # noqa: E402
from harness import check, with_raw_heights                   # noqa: E402
from test_calibration import legs, speed                      # noqa: E402
from test_camera import DESK_Z, FRAME, app, over, patched, refusal, wait_until   # noqa: E402

NOTCH = 44.0
MOVED_RED = (110.0, -150.0)          # beside the green cube in the picture, as a red cube may well stand
# The cubes CALIBRATE is fitted to stand this far from where the arm recorded them, desk mm: centred by eye.
CENTRED_BY_EYE = {"blue": (2.0, -1.0), "green": (-1.5, 1.5), "red": (1.5, 1.0)}
# Where the red cube is set down for the points that refine the calibration.
REFINE_SPOTS = [(-100.0, -200.0), (0.0, -150.0), (120.0, -120.0), (-170.0, -90.0), (60.0, -230.0),
                (-40.0, -240.0), (170.0, -60.0)]


@contextlib.contextmanager
def quick_app():
    """test_camera's app, the live view's still not waiting for a camera's exposure to settle: there is none."""
    with patched(feed_module, "STARTING_FRAMES", 2), app() as parts:
        yield parts


def calibrate(post, camera, config) -> cube_scene.Scene:
    """The three cubes recorded where the page suggests, drawn there, and CALIBRATE."""
    for colour in cube_scene.SPOTS:
        over(post, colour)
        post("camera/record", colour=colour)
    recorded = {colour: (point.x, point.y) for colour, point in camera_store.load_points(config).items()}
    scene = cube_scene.render(spots=recorded)
    camera.frame = scene.frame
    calibrated = post("camera/calibrate")
    assert calibrated["status"] == "done", calibrated
    return scene


def calibrate_by_eye(post, camera, config) -> cube_scene.Scene:
    """CALIBRATE on cubes that stand a little off where the arm recorded them, as when centred by eye."""
    for colour in cube_scene.SPOTS:
        over(post, colour)
        post("camera/record", colour=colour)
    points = camera_store.load_points(config)
    scene = cube_scene.render(spots={colour: (point.x + CENTRED_BY_EYE[colour][0], point.y + CENTRED_BY_EYE[colour][1])
                                     for colour, point in points.items()})
    camera.frame = scene.frame
    calibrated = post("camera/calibrate")
    assert calibrated["status"] == "done", calibrated
    return scene


def record_points(post, state, camera, spots) -> list:
    """For each spot: the red cube alone set down there, found, centred under the cup, and RECORD POINT.

    Gives each reply, and where the cube was drawn: where the cup was sent."""
    recorded = []
    for spot in spots:
        camera.frame = cube_scene.render(spots={"red": spot}, decoy=False).frame
        post("tests/find-red")
        x, y, _ = state()["camera"]["position"]
        camera.frame = cube_scene.render(spots={"red": (x, y)}, decoy=False).frame
        recorded.append((post("tests/record"), (x, y)))
    return recorded


def view_pixel(scene: cube_scene.Scene, x: float, y: float) -> np.ndarray:
    """Where the drawing puts a desk point in the live view, 800 x 600: the same picture scaled, pixel centres kept."""
    pixel = project(scene.desk_to_pixel, np.array([[x, y, 0.0]]))[0]
    return (pixel + 0.5) * VIEW_SIZE[0] / cube_scene.SIZE[0] - 0.5


def drawn_truth(scene: cube_scene.Scene) -> CameraFit:
    """The camera that drew the scenes, the arm's x and y true but for the board's mirror."""
    return CameraFit(scene.true_to_pixel, HANDEDNESS.copy(), {}, cube_scene.SIZE, (0.0, 0.0), 0)


def shares(scene: cube_scene.Scene, x: float, y: float):
    """Where the drawing puts a desk point, as shares of the frame: what a click there sends."""
    px, py = project(scene.desk_to_pixel, np.array([[x, y, 0.0]]))[0]
    width, height = cube_scene.SIZE
    return (px + 0.5) / width, (py + 0.5) / height


def heights(moves) -> list:
    return [round(point[2] - DESK_Z, 2) for point, _ in moves]


def test_find_red_cube() -> bool:
    print("FIND RED CUBE: the arm out of the picture, the red cube found in the live view, the cup down over it:")
    with quick_app() as (post, state, board, camera, config, _):
        before = state()
        scene = calibrate(post, camera, config)
        camera.frame = cube_scene.render(spots=dict(scene.points, red=MOVED_RED)).frame
        post("camera/jump", x=-100.0, y=-200.0)
        board.log.clear()
        found_it = post("tests/find-red")
        moves = legs(board)
        after = state()
    found = after["tests"]["found"] or {}
    point = (found.get("x", math.inf), found.get("y", math.inf))
    end = [(move[0][0], move[0][1]) for move in moves[-2:]]
    return all([
        check("without a camera.json the page is not there, and says why",
              "no camera calibration" in (before["tests"]["problem"] or "")
              and before["unavailable"].get("tests") == before["tests"]["problem"], before["tests"]["problem"]),
        check("after CALIBRATE it is", after["tests"]["problem"] is None and after["unavailable"] == {}),
        check("home first, out of the picture", moves and moves[0][0] == HOME, f"{moves[:2]}"),
        check("the red cube found in the live view, moved since the calibration: within 1 mm of where it was drawn",
              found.get("frame") == list(VIEW_SIZE) and math.dist(point, MOVED_RED) < 1.0,
              f"{np.round(point, 2)} for {MOVED_RED}, {found.get('frame')}"),
        check("over it at 54 mm, fast, then slowly down to 44: 4 mm over its top",
              found_it["status"] == "done" and heights(moves[-2:]) == [FIND_OVER_MM, NOTCH]
              and all(math.dist(xy, point) < 0.01 for xy in end)
              and abs(speed(moves[-2][0], moves[-1]) - MOVE_SLOW_MM_S) < 0.1, f"{moves[-2:]}"),
        check("the page is told where it stands, and what to draw", found_it["detail"].startswith("the red cube stands")
              and len(found["corners"]) == 8 and len(found["seen"]) + len(found["hidden"]) == 12,
              found_it["detail"]),
    ])


def test_find_red_refusals() -> bool:
    print("FIND RED CUBE refuses, saying why: no calibration, no red cube, something red that is not the cube:")
    with quick_app() as (post, state, board, camera, config, _):
        uncalibrated = post("tests/find-red")
        unmoved = not legs(board)
        scene = calibrate(post, camera, config)
        camera.frame = cube_scene.render(spots={colour: spot for colour, spot in scene.points.items()
                                                if colour != "red"}, decoy=False).frame
        missing = post("tests/find-red")
        cleared = state()["tests"]["found"]
        camera.frame = flat_red_square(scene, 40.0)
        flat = post("tests/find-red")
    return all([
        check("no camera.json: refused, nothing moved",
              uncalibrated["status"] == "refused" and "no camera calibration" in uncalibrated["detail"]
              and unmoved, uncalibrated["detail"]),
        check("no red cube in view: refused, saying what the camera does see",
              missing["status"] == "refused" and "no red cube in view" in missing["detail"] and cleared is None,
              missing["detail"]),
        check("a flat red square where the cube stood: not a 40 mm cube standing on the desk",
              flat["status"] == "refused" and "not a 40 mm cube" in flat["detail"], flat["detail"]),
    ])


def flat_red_square(scene: cube_scene.Scene, size: float) -> np.ndarray:
    """The scene without the red cube, a red square of paper lying where it stood."""
    frame = cube_scene.render(spots={colour: spot for colour, spot in scene.points.items() if colour != "red"},
                              decoy=False).frame.astype(np.float32)
    centre, half = HANDEDNESS @ np.array(scene.points["red"]), size / 2
    square = np.array([(centre[0] + dx * half, centre[1] + dy * half, 0.0)
                       for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    cube_scene._fill(frame, project(scene.true_to_pixel, square), cube_scene.COLOURS_BGR["red"])
    return np.clip(frame, 0, 255).astype(np.uint8)


def test_touch() -> bool:
    print("ARM TO DESK: the desk at the point clicked in the live view, over it at 14 mm, down to 4, corrected:")
    with quick_app() as (post, state, board, camera, config, _):
        scene = calibrate(post, camera, config)
        first, second = (100.0, -200.0), (-120.0, -180.0)
        board.log.clear()
        replies, moves = [], []
        for point in (first, second):
            u, v = shares(scene, *point)
            replies.append(post("tests/touch", u=u, v=v))
            moves.append(legs(board))
        far = post("tests/touch", u=0.02, v=0.02)
        far_moves = legs(board)
        outside = refusal(lambda: post("tests/touch", u=1.2, v=0.5))
    sent = [(path[-1][0][0], path[-1][0][1]) for path in moves]
    return all([
        check("each click names the desk point under it, within 1 mm of where it is drawn, and the arm goes there",
              all(math.dist(reply["point"], point) < 1.0 and math.dist(xy, reply["point"]) < 0.01
                  for reply, point, xy in zip(replies, (first, second), sent)),
              f"{[reply['point'] for reply in replies]}"),
        check("from high up: down fast to 14 mm over the point, then slowly to 4",
              replies[0]["status"] == "done" and heights(moves[0][-2:]) == [TOUCH_OVER_MM, TOUCH_MM]
              and abs(speed(moves[0][-2][0], moves[0][-1]) - MOVE_SLOW_MM_S) < 0.1, f"{moves[0]}"),
        check("from 4 mm over the last: slowly up to 14, fast across at 14, slowly down to 4",
              heights(moves[1]) == [TOUCH_OVER_MM, TOUCH_OVER_MM, TOUCH_MM]
              and abs(speed(moves[1][0][0], moves[1][1]) - MOVE_FAST_MM_S) < 1, f"{moves[1]}"),
        check("a click on the desk out of reach: refused, nothing moved",
              far["status"] == "refused" and "out of the arm's reach" in far["detail"] and not far_moves,
              far["detail"]),
        check("a click off the view: refused", "goes from 0 to 1" in str(outside), str(outside)),
    ])


def test_the_calibration_the_page_draws_by() -> bool:
    """desk_to_view, and the page's own arithmetic (shares.js) on it, in node: the same as the app's."""
    print("the page draws by the calibration, in shares of the view; its own arithmetic agrees with the app's:")
    with quick_app() as (post, state, board, camera, config, _):
        scene = calibrate(post, camera, config)
        matrix = np.array(state()["tests"]["desk_to_view"])
        to_true = state()["tests"]["desk_to_true"]
        desk = np.array([(100.0, -200.0, 0.0), (-150.0, -120.0, 0.0), (160.0, 0.0, 40.0)])
        from_page = project(matrix, desk)
        fit = camera_store.load_camera(config)
        width, height = fit.frame_size
        from_fit = (project(fit.desk_to_pixel, desk) + 0.5) / (width, height)
        clicks = [shares(scene, x, y) for x, y, _ in desk[:2]]
        replies = [post("tests/touch", u=u, v=v)["point"] for u, v in clicks]
    is_same = np.abs(from_page - from_fit).max() < 1e-9
    node = shutil.which("node")
    if node is None:
        print("  SKIP  node is not on PATH: shares.js not run")
        return check("desk_to_view is the calibration, in shares of the frame", is_same)
    turns = np.linspace(0, 2 * math.pi, 49)
    rim = np.column_stack((10 * np.cos(turns), 10 * np.sin(turns))) @ np.linalg.inv(to_true).T
    true_rim = project(matrix, np.column_stack((100.0 + rim[:, 0], -200.0 + rim[:, 1], np.full(len(turns), 4.0))))
    driver = (f"import {{ circle, toDesk, toView }} from '{ROOT / 'calibration' / 'common' / 'shares.js'}';\n"
              f"const matrix = {json.dumps(matrix.tolist())};\n"
              f"console.log(JSON.stringify({{ view: {json.dumps(desk.tolist())}.map((p) => toView(matrix, p)), "
              f"desk: {json.dumps(clicks)}.map((c) => toDesk(matrix, c)), "
              f"rim: circle(matrix, [100, -200], 10, 4, {json.dumps(to_true)}) }}));\n")
    with tempfile.TemporaryDirectory() as scratch:
        (Path(scratch) / "driver.mjs").write_text(driver)
        ran = subprocess.run([node, str(Path(scratch) / "driver.mjs")], capture_output=True, text=True, timeout=60)
    page = json.loads(ran.stdout) if ran.returncode == 0 else {"view": [[math.inf] * 2] * 3, "desk": [[0, 0]] * 2,
                                                                "rim": [[math.inf] * 2] * 49}
    return all([
        check("desk_to_view is the calibration, in shares of the frame", is_same),
        check("the page puts desk points where the app does", np.abs(np.array(page["view"]) - from_page).max() < 1e-9,
              ran.stderr.strip()[:200]),
        check("and a click on the same desk point as the app sends the arm to",
              np.abs(np.array(page["desk"]) - np.array(replies)).max() < 0.01, f"{page['desk']} for {replies}"),
        check("the cup's rim drawn its true size: 20 mm through the arm's x/y error, as the camera has it",
              np.abs(np.array(page["rim"]) - true_rim).max() < 1e-9),
    ])


def test_refine() -> bool:
    print("the red cube's points: the arm over each, the camera's view of it, and camera.json refined by them:")
    with quick_app() as (post, state, board, camera, config, _):
        scene = calibrate_by_eye(post, camera, config)
        cubes_alone = camera_store.load_camera(config)
        truth = drawn_truth(scene)
        board.log.clear()
        recorded = record_points(post, state, camera, REFINE_SPOTS[:1])
        moves = legs(board)
        recorded += record_points(post, state, camera, REFINE_SPOTS[1:])
        shown = state()["tests"]["refine"]
        on_file = camera_store.load_refine_points(config)
        check_stamp = (config / camera_store.CHECK_FILE).stat().st_mtime_ns
        saved = post("tests/save")
        refined = camera_store.load_camera(config)
        is_check_drawn = (config / camera_store.CHECK_FILE).stat().st_mtime_ns != check_stamp
        camera.frame = scene.frame
        validated = post("camera/validate")
        forgot = post("tests/forget", id=2)
        after_forget = [point["id"] for point in state()["tests"]["refine"]["points"]]
        cleared = post("tests/clear")
        put_back = post("tests/save")
        alone_again = camera_store.load_camera(config)
        record_points(post, state, camera, REFINE_SPOTS[:1])
        camera.frame = scene.frame
        post("camera/calibrate")
        set_aside = state()["tests"]["refine"]["points"]
        previous = json.loads((config / camera_store.REFINE_PREVIOUS_FILE).read_text())["points"]
    before_mm, after_mm = difference_mm(cubes_alone, truth), difference_mm(refined, truth)
    apart = [math.dist(point.pixel, view_pixel(scene, *drawn)) for point, (_, drawn) in zip(on_file, recorded)]
    return all([
        check("each point recorded: the arm read over the cube, parked, the cube seen where it stands",
              all(reply["status"] == "done" for reply, _ in recorded) and len(on_file) == len(REFINE_SPOTS)
              and moves[-1][0] == HOME, f"{[reply['detail'] for reply, _ in recorded][:2]}"),
        check("the camera's view of each, its centre in the live view: within 0.4 px of the drawing's",
              max(apart) < 0.4, f"{max(apart):.2f} px"),
        check("the page is told each point and cube, how far off through the saved and the refined, and the change",
              len(shown["points"]) == len(REFINE_SPOTS) and len(shown["cubes"]) == 3 and shown["fit"] is not None
              and all(row["off"]["after"] is not None for row in shown["points"] + shown["cubes"]),
              f"{shown['fit']}"),
        check("SAVE CALIBRATION: camera.json refined, the cubes' own fit kept beside it",
              saved["status"] == "done" and refined.refined["points"] == len(REFINE_SPOTS)
              and np.allclose(refined.cube_fit().desk_to_pixel, cubes_alone.desk_to_pixel), saved["detail"]),
        check("and less than half as far from the drawing's desk as the cubes alone, which were centred by eye",
              after_mm[0] < 0.5 * before_mm[0] and after_mm[1] < 0.5 * before_mm[1],
              f"cubes alone {before_mm[0]:.2f} mm, {before_mm[1]:.2f} at worst; refined {after_mm[0]:.2f}, "
              f"{after_mm[1]:.2f}"),
        check("the check image redrawn with it", is_check_drawn),
        check("VALIDATE against the cubes' own fit: a refinement is not a camera moved",
              validated["status"] == "done" and "has the camera moved" not in validated["detail"],
              validated["detail"]),
        check("a point forgotten", forgot["status"] == "done" and 2 not in after_forget
              and len(after_forget) == len(REFINE_SPOTS) - 1),
        check("all cleared, SAVE puts the cubes' own calibration back",
              cleared["status"] == "done" and put_back["status"] == "done" and alone_again.refined is None
              and np.allclose(alone_again.desk_to_pixel, cubes_alone.desk_to_pixel)
              and alone_again.cubes == cubes_alone.cubes, put_back["detail"]),
        check("a new CALIBRATE sets the points aside", set_aside == [] and len(previous) == 1),
    ])


def test_refine_refusals() -> bool:
    print("RECORD POINT and SAVE CALIBRATION refuse, saying why:")
    with quick_app() as (post, state, board, camera, config, _):
        calibrate(post, camera, config)
        board.log.clear()
        high = post("tests/record")
        unmoved = not legs(board)
        camera.frame = cube_scene.render(spots={"red": (0.0, -160.0)}, decoy=False).frame
        post("tests/find-red")
        camera.frame = cube_scene.render(spots={"red": (0.0, -200.0)}, decoy=False).frame
        board.log.clear()
        away = post("tests/record")
        parked = legs(board)
        too_few = post("tests/save")
        unknown = refusal(lambda: post("tests/forget", id=7))
        points = state()["tests"]["refine"]["points"]
    return all([
        check("the cup not over a cube's top: refused, nothing moved",
              high["status"] == "refused" and "not over a cube's top" in high["detail"] and unmoved, high["detail"]),
        check("the cube 40 mm from the cup: the arm parked, nothing recorded",
              away["status"] == "refused" and "not under the cup" in away["detail"] and parked[-1][0] == HOME
              and points == [], away["detail"]),
        check("SAVE with too few points", too_few["status"] == "refused" and "3 more points" in too_few["detail"],
              too_few["detail"]),
        check("a point that is not there", "no point 7" in str(unknown), str(unknown)),
    ])


def test_refine_arithmetic() -> bool:
    """camera/refine.py alone: the points are where the drawing's camera puts them, no frames, no arm."""
    print("the refinement's arithmetic: the arm's error found again, the camera kept, the form of camera.json kept:")
    error = np.array([[0.97, 0.02], [0.02, 1.03]])            # the arm's x/y error, as the drawing has it
    scene = cube_scene.render(spots={colour: (x + CENTRED_BY_EYE[colour][0], y + CENTRED_BY_EYE[colour][1])
                                     for colour, (x, y) in cube_scene.SPOTS.items()}, stretch=error)
    fit = fit_camera(scene.frame, cube_scene.SPOTS)
    spots = [(x, y) for x in range(-200, 201, 50) for y in range(-250, 1, 50) if 110 < math.hypot(x, y) < 260]
    points = [Sighting(float(x), float(y), tuple(view_pixel(scene, x, y)), VIEW_SIZE) for x, y in spots]
    refined = refine(fit, points)
    again = refine(refined, points)
    truth = CameraFit(scene.true_to_pixel, error @ HANDEDNESS, {}, cube_scene.SIZE, (0.0, 0.0), 0)
    before_mm, after_mm = difference_mm(fit, truth), difference_mm(refined, truth)
    stretch = refined.desk_to_true @ HANDEDNESS
    return all([
        check(f"{len(points)} points spread over the desk outvote the three cubes centred by eye: the drawing's desk "
              f"within 0.3 mm (median), 1.5 at worst", after_mm[0] < 0.3 and after_mm[1] < 1.5,
              f"cubes alone {before_mm[0]:.2f} mm, {before_mm[1]:.2f} at worst; refined {after_mm[0]:.2f}, "
              f"{after_mm[1]:.2f}"),
        check("the camera itself is the cubes' fit, turned and shifted on the desk only",
              np.allclose(np.linalg.svd(refined.true_to_pixel[:, :3])[1], np.linalg.svd(fit.true_to_pixel[:, :3])[1])),
        check("the arm's error still a symmetric stretch and the mirror", np.allclose(stretch, stretch.T)),
        check("refined again from the same points: the same", np.allclose(again.desk_to_pixel, refined.desk_to_pixel)),
        check("no points: the cubes' own fit", np.allclose(refine(fit, []).desk_to_pixel, fit.desk_to_pixel, atol=1e-6)),
    ])


def test_page_with_points() -> bool:
    """tests.js run in node on the app's own states (drive-page.mjs): no calibration, no points, one, three,
    saved. Needs node."""
    print("the page's script on the app's states, in node: nothing thrown, the points and SAVE shown as they are:")
    node = shutil.which("node")
    if node is None:
        print("  SKIP  node is not on PATH")
        return True
    with quick_app() as (post, state, board, camera, config, server):
        limits = server.routes[("GET", "/api/limits")]({})
        states = [state()]
        calibrate_by_eye(post, camera, config)
        states.append(state())
        record_points(post, state, camera, REFINE_SPOTS[:1])
        states.append(state())
        record_points(post, state, camera, REFINE_SPOTS[1:3])
        states.append(state())
        post("tests/save")
        states.append(state())
    watch = ["blocked-why", "record", "refine-count", "refine-ask", "points", "refine-fit", "refine-saved", "save"]
    replies = {"limits": limits, "states": states, "clicks": ["find", "record", "record", "save"], "watch": watch}
    with tempfile.TemporaryDirectory() as scratch:
        (Path(scratch) / "replies.json").write_text(json.dumps(replies))
        ran = subprocess.run([node, "--experimental-default-type=module", str(HERE / "drive-page.mjs"),
                              str(ROOT / "calibration" / "tests" / "tests.js"), str(Path(scratch) / "replies.json")],
                             capture_output=True, text=True, timeout=60)
    shown = json.loads(ran.stdout) if ran.returncode == 0 and ran.stdout else {"errors": [ran.stderr], "steps": []}
    steps = shown["steps"]
    if len(steps) != len(states):
        return check("every state shown", False, f"{shown['errors'][:1]}")
    blocked, calibrated, one, enough, saved = steps
    return all([
        check("nothing thrown, from the page with no calibration to camera.json refined", not shown["errors"],
              "\n".join(shown["errors"])[:600]),
        check("no calibration: the page says why", "no camera calibration" in blocked["blocked-why"]["text"].lower()),
        check("calibrated: the three cubes in the table, no point yet, SAVE not to be pressed",
              calibrated["points"]["rows"].count('class="cube"') == 3 and "No red cube points yet"
              in calibrated["points"]["rows"] and calibrated["save"]["disabled"], calibrated["refine-fit"]["html"]),
        check("one point: numbered in the table; SAVE says how many more", one["points"]["rows"].count('class="badge')
              == 1 and one["save"]["disabled"] and "2 more points" in one["save"]["title"], one["save"]["title"]),
        check("enough: SAVE can be pressed, and what it would do is said",
              enough["points"]["rows"].count('class="badge') == 3 and not enough["save"]["disabled"]
              and "After SAVE" in enough["refine-fit"]["html"] and "3 of" in enough["refine-count"]["html"],
              enough["refine-fit"]["html"]),
        check("a place for the next point is suggested", "dashed circle" in calibrated["refine-ask"]["html"],
              calibrated["refine-ask"]["html"][-120:]),
        check("saved: camera.json says it is refined", "refined by 3 points" in saved["refine-saved"]["text"],
              saved["refine-saved"]["text"]),
    ])


def test_still_frame() -> bool:
    print("a still from the live view: taken after it is asked for, the camera started for it and let go after:")
    with patched(feed_module, "STARTING_FRAMES", 2), fake_camera(FRAME) as camera:
        feed = Feed()
        feed.start()
        try:
            still = feed.still()
            is_let_go = wait_until(lambda: camera.released == camera.opened)
            size = feed.state()["size"]
        finally:
            feed.close()
    return all([
        check("the view's frame, as the camera gave it", still.shape[:2] == (VIEW_SIZE[1], VIEW_SIZE[0])
              and size == list(VIEW_SIZE), f"{still.shape}, {size}"),
        check("no page watching: the camera opened for it, and let go after", camera.opened == 1 and is_let_go),
    ])


def test_check_image_cubes() -> bool:
    print("the check image's cubes: every edge, those the faces hide among them, and an X under each centre:")
    scene = cube_scene.render()
    fit = fit_camera(scene.frame, scene.points)
    results = []
    for colour in cube_scene.SPOTS:
        cube = fit.wireframe(colour)
        x, y = fit.cubes[colour]["x"], fit.cubes[colour]["y"]
        under = project(fit.desk_to_pixel, np.array([[x, y, 0.0]]))[0]
        (a, b), (c, d) = (cube.corners[list(diagonal)] for diagonal in BASE_DIAGONALS)
        crossing = a + _cross(c - a, d - c) / _cross(b - a, d - c) * (b - a)
        results.append((len(cube.seen), len(cube.hidden), float(np.linalg.norm(crossing - under))))
    return all([
        check("12 edges a cube, seen or hidden; 3 hidden for a cube seen corner-on",
              all(seen + hidden == 12 and hidden in (3, 5) for seen, hidden, _ in results), f"{results}"),
        check("the X crosses where the cube's centre stands on the desk", all(apart < 1e-6 for *_, apart in results)),
    ])


def _cross(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def test_on_the_owners_frame() -> bool:
    """The red cube in the owner's own calibration frame, at the live view's size, through the three cubes' own
    camera: a refined one moves the cubes by design. Skipped without one."""
    print("the owner's frame at the live view's size: the red cube where the three cubes' calibration has it:")
    config = ROOT / "config"
    saved = camera_store.load_camera(config) if (config / camera_store.CAMERA_FILE).exists() else None
    fit = saved.cube_fit() if saved else None
    frame = cv2.imread(str(config / camera_store.FRAME_FILE))
    if fit is None or frame is None:
        print("  SKIP  no config/camera.json and camera-frame.png")
        return True
    try:
        found = find_red_cube(cv2.resize(frame, VIEW_SIZE, interpolation=cv2.INTER_AREA), fit)
    except (CubesNotFound, CameraNotFound) as reason:
        return check("found", False, str(reason))
    red = (fit.cubes["red"]["x"], fit.cubes["red"]["y"])
    return check("within 0.5 mm of the calibration's red cube", math.dist((found.x, found.y), red) < 0.5,
                 f"({found.x:.2f}, {found.y:.2f}) for {red}; edges {found.edge_px:.2f} px")


TESTS = [with_raw_heights(test) for test in [
    test_find_red_cube, test_find_red_refusals, test_touch, test_the_calibration_the_page_draws_by, test_refine,
    test_refine_refusals, test_page_with_points]] + [
    test_refine_arithmetic, test_still_frame, test_check_image_cubes, test_on_the_owners_frame]
