#!/usr/bin/env python3
"""The camera page against the fake board and a fake camera: what each request sends the arm, the cubes'
points, CALIBRATE and CHECK, the live view, which camera is picked, and the page's own scripts.

The handlers are called as the server calls them, but for the live view, which is fetched over a real socket:
how the stream is framed is the point. The library's waits are recorded instead of slept; config/ is a scratch
directory each time, with a level desk at arm z 44 saved in it, so desk height is arm z less 44. Heights are
checked in the firmware's own (`with_raw_heights`): what is checked here does not depend on true z.
"""

import contextlib
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "calibration"), str(HERE)]

from maxarm import MaxArm, Space                              # noqa: E402
from maxarm.arm import MOVE_FAST_MM_S, MOVE_SLOW_MM_S         # noqa: E402
from maxarm.geometry import HOME                              # noqa: E402
from camera import devices                                    # noqa: E402
from camera import feed as feed_module                        # noqa: E402
from camera import store as camera_store                      # noqa: E402
from camera.api import NOTCH_MM, CameraSession                # noqa: E402
from camera.devices import CameraNode                         # noqa: E402
from camera.feed import Feed                                  # noqa: E402
from common import arm_api                                    # noqa: E402
from common import server as server_module                    # noqa: E402
from common.motion import HOVER_MM, OVER_CUBES_MM, RESET_LIFT_MM, Motion  # noqa: E402
from common.server import BadRequest, Server                  # noqa: E402
from common.worker import Worker                              # noqa: E402
from desk import store as desk_store                          # noqa: E402
from desk.api import DeskSession                              # noqa: E402
from desk.plane import Plane                                  # noqa: E402
import fake_arm                                               # noqa: E402
from fake_arm import fake_pyserial, instant_waits             # noqa: E402
from fake_camera import fake_camera                           # noqa: E402
from harness import check, with_raw_heights                   # noqa: E402
from test_calibration import CMD_MM, legs, speed              # noqa: E402

DESK_Z = 44.0
SPOTS = {"blue": (-160.0, -220.0), "green": (160.0, -220.0), "red": (160.0, 0.0)}
GUIDE = ROOT / "calibration" / "camera" / "guide.md"


def still_frame() -> np.ndarray:
    """A frame to stand in for the camera's: smooth, so that it compresses as a real one does."""
    rows, columns = np.mgrid[0:972, 0:1296]
    return np.dstack([(columns // 6) % 256, (rows // 4) % 256, (rows + columns) // 9 % 256]).astype(np.uint8)


FRAME = still_frame()


@contextlib.contextmanager
def app():
    """The app as calibration.py builds it, on the fake board and camera.

    Yields post(path, **body) -> reply, state() -> dict, the fake board, the fake camera, the scratch config and
    the server."""
    gain, fake_arm.TRACKING_GAIN = fake_arm.TRACKING_GAIN, 1.0       # the servos get where they are sent
    with tempfile.TemporaryDirectory() as scratch, fake_pyserial() as boards, instant_waits(), \
            fake_camera(FRAME) as camera:
        config = Path(scratch)
        desk_store.save_desk(config, Plane(0.0, 0.0, DESK_Z), [])
        worker = Worker(Motion(MaxArm(config_dir=None), config))
        feed = Feed()
        server = Server(ROOT / "calibration", home="desk", pages=["camera"])
        arm_api.register(server, worker)
        DeskSession(worker, config).register(server)
        CameraSession(worker, feed, config).register(server)
        worker.thread.start()
        feed.start()
        worker.connect()
        boards[0].log.clear()

        def post(path: str, **body) -> dict:
            return server.routes[("POST", f"/api/{path}")](body)

        def state() -> dict:
            return json.loads(json.dumps(server.routes[("GET", "/api/state")]({})))

        try:
            yield post, state, boards[0], camera, config, server
        finally:
            feed.close()
            worker.shutdown()
            fake_arm.TRACKING_GAIN = gain


def over(post, colour: str) -> None:
    """The arm over a cube's place, at the notch."""
    x, y = SPOTS[colour]
    post("camera/jump", x=x, y=y)
    post("camera/move", x=x, y=y, z=NOTCH_MM)


def height(point) -> float:
    return point[2] - DESK_Z


def test_moves_over_the_cubes() -> bool:
    print("the arm over the cubes: across at 60 mm over the desk, down onto a cube slowly, corrected there:")
    with app() as (post, state, board, _, _, _):
        jumped = post("camera/jump", x=-160.0, y=-220.0)
        first = legs(board)
        lowered = post("camera/move", x=-160.0, y=-220.0, z=NOTCH_MM)
        down = legs(board)
        post("camera/jump", x=160.0, y=-220.0)
        across = legs(board)
        position = state()["camera"]["position"]
    return all([
        check("from home, over the blue cube's place, high above it", jumped["status"] == "done"
              and len(first) == 1 and first[0][0][:2] == (-160.0, -220.0)
              and height(first[0][0]) >= OVER_CUBES_MM, f"{first}"),
        check("down to the notch: fast to 60 over the desk, then slowly",
              lowered["status"] == "done"
              and [round(height(point), 2) for point, _ in down] == [OVER_CUBES_MM, NOTCH_MM]
              and abs(speed(down[0][0], down[1]) - MOVE_SLOW_MM_S) < 0.1, f"{down}"),
        check("to the next place: slowly up to 60, fast across at 60, and it stays there",
              [round(height(point), 2) for point, _ in across] == [OVER_CUBES_MM, OVER_CUBES_MM]
              and abs(speed((-160.0, -220.0, DESK_Z + NOTCH_MM), across[0]) - MOVE_SLOW_MM_S) < 0.1
              and abs(speed(across[0][0], across[1]) - MOVE_FAST_MM_S) < 1, f"{across}"),
        check("the page is told where it was sent, in desk space",
              max(abs(a - b) for a, b in zip(position, (160.0, -220.0, OVER_CUBES_MM))) < CMD_MM, f"{position}"),
    ])


def test_points() -> bool:
    print("a point per cube: where the servos say the cup is, in desk space, kept in config/camera-points.json:")
    with app() as (post, state, board, _, config, _):
        over(post, "blue")
        board.arm.position = board.arm.actual = (-160.6, -219.2, DESK_Z + 44.7)   # settled off where it was sent
        recorded = post("camera/record", colour="blue")
        cubes = state()["camera"]["cubes"]
        on_file = json.loads((config / camera_store.POINTS_FILE).read_text())["cubes"]
        read_back = camera_store.load_points(config)
        forgot = post("camera/forget", colour="blue")
        after = state()["camera"]["cubes"]["blue"]
        refusals = [refusal(lambda: post("camera/record", colour="purple")),
                    refusal(lambda: post("camera/visit", colour="blue"))]
    blue = cubes["blue"]
    return all([
        check("recorded where the servos are, with where it was sent beside it",
              abs(blue["x"] + 160.6) < 0.7 and abs(blue["y"] + 219.2) < 0.7 and abs(blue["z"] - 44.7) < 0.7
              and blue["sent"] == [-160.0, -220.0, NOTCH_MM], recorded["detail"]),
        check("kept on file, the cubes not recorded as null", on_file == cubes, f"{on_file}"),
        check("and read back as written", read_back["blue"] == camera_store.CubePoint(**blue)
              and read_back["green"] is None),
        check("forgotten", forgot["status"] == "done" and after is None),
        check("a colour that is not a cube, or a cube not recorded, is refused",
              "purple" in refusals[0] and "not recorded" in refusals[1], f"{refusals}"),
    ])


def test_calibrate_and_check() -> bool:
    print("CALIBRATE and CHECK: the arm out of the picture first, then the snapshot:")
    with app() as (post, state, board, camera, config, server):
        early = post("camera/calibrate")
        for colour in SPOTS:
            over(post, colour)
            post("camera/record", colour=colour)
        board.log.clear()
        calibrated = post("camera/calibrate")
        parked = [entry for entry in board.log if entry[0] == "move"]
        snapshots = camera.snapshots
        frame = cv2.imread(str(config / camera_store.FRAME_FILE))
        frame_written = (config / camera_store.FRAME_FILE).stat().st_mtime_ns
        described = state()["camera"]["check"]
        image = server.byte_routes["/api/camera/check.jpg"]()
        checked = post("camera/check")
        after = state()["camera"]
        is_frame_kept = (config / camera_store.FRAME_FILE).stat().st_mtime_ns == frame_written
    red_x, red_y = SPOTS["red"]
    return all([
        check("refused until all three cubes are recorded", early["status"] == "refused"
              and "blue, green and red cubes" in early["detail"], early["detail"]),
        check("parked: straight up from the red cube, fast, then home",
              [entry[1] for entry in parked] == [(red_x, red_y, RESET_LIFT_MM), HOME], f"{parked}"),
        check("then one snapshot, kept as config/camera-frame.png as it was taken",
              calibrated["status"] == "done" and snapshots == 1 and frame is not None
              and np.array_equal(frame, FRAME), calibrated["detail"]),
        check("which says the fit is still to come", "not built yet" in calibrated["detail"]),
        check("the check image is the frame for now, and the page can fetch it",
              described["size"] == [FRAME.shape[1], FRAME.shape[0]] and image[1] == "image/jpeg"
              and image[0][:2] == b"\xff\xd8", f"{described}"),
        check("CHECK: a new snapshot for the check image, the calibration's frame left alone",
              checked["status"] == "done" and camera.snapshots == 2 and is_frame_kept
              and after["check"]["stamp"] != described["stamp"], checked["detail"]),
    ])


def test_reset_clears_the_cubes() -> bool:
    print("RESET from low down beside a cube: slowly up off the desk, fast on up over the cubes, then home:")
    with app() as (post, _, board, _, _, _):
        post("camera/jump", x=-180.0, y=-200.0)
        post("camera/move", x=-180.0, y=-200.0, z=5.0)
        board.log.clear()
        reset = post("reset")
        moves = legs(board)
    hover = HOVER_MM[Space.ARM]
    return check("up to the arm's hover height slowly, to the lift fast, then home",
                 reset["status"] == "done"
                 and [point for point, _ in moves] == [(-180.0, -200.0, hover), (-180.0, -200.0, RESET_LIFT_MM), HOME]
                 and abs(speed((-180.0, -200.0, DESK_Z + 5.0), moves[0]) - MOVE_SLOW_MM_S) < 0.1
                 and abs(speed(moves[0][0], moves[1]) - MOVE_FAST_MM_S) < 1, f"{moves}")


def test_live_view() -> bool:
    print("the live view: JPEGs over a socket while a page watches, paused for a snapshot, the camera let go after:")
    with fake_camera(FRAME) as camera:
        feed = Feed()
        server = Server(ROOT / "calibration", home="desk", pages=["camera"])
        server.get_stream("/api/camera/feed", feed.frames)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server_module._RequestHandler)
        httpd.daemon_threads = True
        httpd.app = server
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        feed.start()
        try:
            url = f"http://127.0.0.1:{httpd.server_address[1]}/api/camera/feed"
            with urllib.request.urlopen(url, timeout=10) as response:
                kind = response.headers["Content-Type"]
                first = read_part(response)
                live = feed.state()
                still, _ = feed.snapshot()
                second = read_part(response)
                reopened = camera.opened
            is_let_go = wait_until(lambda: feed.state()["status"] == "idle" and camera.released == camera.opened)
        finally:
            feed.close()
            httpd.shutdown()
            httpd.server_close()
    image = cv2.imdecode(np.frombuffer(first, np.uint8), cv2.IMREAD_COLOR)
    width, height = feed_module.VIEW_SIZE
    return all([
        check("an MJPEG stream", kind.startswith("multipart/x-mixed-replace"), kind),
        check("of JPEGs at the view's size, and the page is told it is live",
              image is not None and image.shape[:2] == (height, width) and live["status"] == "live", f"{live}"),
        check("a snapshot meanwhile: the full frame, the view's camera let go for it",
              np.array_equal(still, FRAME) and camera.open_at_snapshots == [0], f"{camera.open_at_snapshots}"),
        check("and the view picks up again after it", second is not None and reopened == 2),
        check("the page gone, the camera is let go", is_let_go),
    ])


def test_camera_failing() -> bool:
    print("a camera that will not open: the page is told why, and it is tried again:")
    retry, feed_module.RETRY_S = feed_module.RETRY_S, 0.05
    with fake_camera(FRAME) as camera:
        camera.is_broken = True
        feed = Feed()
        feed.start()
        frames = []
        viewer = threading.Thread(target=lambda: frames.append(next(feed.frames())), daemon=True)
        viewer.start()
        try:
            is_failing = wait_until(lambda: feed.state()["status"] == "failed")
            reason = feed.state()["detail"]
            camera.is_broken = False
            viewer.join(timeout=5.0)
        finally:
            feed.close()
            feed_module.RETRY_S = retry
    return all([
        check("the page is told why", is_failing and "unplugged" in reason, reason),
        check("and once it opens, frames come", len(frames) == 1),
    ])


def test_which_camera() -> bool:
    print("which camera: the owner's if plugged in, else the first real one, never a virtual one:")
    nodes = [CameraNode("/dev/video0", "v4l2 loopback", "OBS Virtual Camera", "platform:v4l2loopback-000", True),
             CameraNode("/dev/video1", "uvcvideo", "Brio 101", "usb-0000:0b:00.0-4", True),
             CameraNode("/dev/video2", "uvcvideo", "Brio 101", "usb-0000:0b:00.0-4", False),
             CameraNode("/dev/video3", "uvcvideo", "USB Camera", devices.PREFERRED_BUS, True),
             CameraNode("/dev/video4", "uvcvideo", "USB Camera", devices.PREFERRED_BUS, False)]
    picked = []
    for plugged in (nodes, nodes[:3], [nodes[0], nodes[2]]):
        with patched(devices, "list_nodes", lambda: plugged):
            picked.append(refusal(devices.find_camera, RuntimeError) or "")
    return all([
        check("the owner's, though another comes first", picked[0] == "/dev/video3", picked[0]),
        check("without it, the first that is not virtual", picked[1] == "/dev/video1", picked[1]),
        check("none but a virtual camera and a metadata node: refused, saying what is there",
              "no camera found" in picked[2] and "OBS Virtual Camera" in picked[2], picked[2]),
    ])


def test_page_scripts() -> bool:
    """The pages' scripts parse, and guide.md comes out of the page's own renderer as it should. Needs node."""
    print("the pages' scripts, and the GUIDE through the page's renderer, in node:")
    node = shutil.which("node")
    if node is None:
        print("  SKIP  node is not on PATH")
        return True
    scripts = sorted((ROOT / "calibration").glob("*/*.js"))
    broken = [script.name for script in scripts if subprocess.run(
        [node, "--experimental-default-type=module", "--check", str(script)], capture_output=True).returncode]
    source = (ROOT / "calibration" / "camera" / "guide.js").read_text()
    renderer = "\n".join(line for line in source.splitlines() if not line.startswith("import "))
    driver = renderer + ("\nconst fs = await import('node:fs');\n"
                         "for (const path of process.argv.slice(2)) "
                         "console.log(JSON.stringify(renderGuide(fs.readFileSync(path, 'utf8'))));\n")
    sample = "Text with <tags> & *stars*,\ncarried on.\n- a bullet\n  carried on\n* another\nafter\n"
    with tempfile.TemporaryDirectory() as scratch:
        (Path(scratch) / "guide.mjs").write_text(driver)
        (Path(scratch) / "sample.md").write_text(sample)
        ran = subprocess.run([node, str(Path(scratch) / "guide.mjs"), str(GUIDE), str(Path(scratch) / "sample.md")],
                             capture_output=True, text=True, timeout=60)
    rendered = [json.loads(line) for line in ran.stdout.splitlines()] if ran.returncode == 0 else ["", ""]
    guide, tricky = rendered
    bullets = sum(1 for line in GUIDE.read_text().splitlines() if line.lstrip().startswith(("- ", "* ")))
    return all([
        check(f"all {len(scripts)} scripts parse", not broken, f"{broken}"),
        check("every bullet of guide.md is a list item, in two lists", guide.count("<li>") == bullets
              and guide.count("<ul>") == 2, f"{guide.count('<li>')} of {bullets}; {ran.stderr.strip()[:200]}"),
        check("bold where the stars are, and no star left over", "<b>Camera placement</b>" in guide
              and "*" not in guide),
        check("markup in the text is shown, not obeyed; lines run on", tricky == (
            "<p>Text with &lt;tags&gt; &amp; <i>stars</i>, carried on.</p>\n"
            "<ul><li>a bullet carried on</li><li>another</li></ul>\n<p>after</p>"), tricky),
    ])


def read_part(response) -> Optional[bytes]:
    """The next JPEG of a multipart stream."""
    length = None
    while True:
        line = response.readline()
        if not line:
            return None
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1])
        elif line == b"\r\n" and length is not None:
            return response.read(length)


def refusal(action: Callable, kind=BadRequest) -> Optional[str]:
    """What `action` was refused with, or what it returned."""
    try:
        return action()
    except kind as error:
        return str(error)


@contextlib.contextmanager
def patched(module, name: str, value):
    saved = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, saved)


def wait_until(condition: Callable[[], bool], timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


TESTS = [with_raw_heights(test) for test in [
    test_moves_over_the_cubes, test_points, test_calibrate_and_check, test_reset_clears_the_cubes]] + [
    test_live_view, test_camera_failing, test_which_camera, test_page_scripts]
