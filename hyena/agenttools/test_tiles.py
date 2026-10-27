#!/usr/bin/env python3
"""The 25 mm tile finder, the work areas, and the Tiles page against the fake board and a fake camera.

The finder is tried on tiles drawn through a known camera (`tile_scene.py`) -- on wood, plain grey and white paper,
two of them touching and two a millimetre apart -- and on the owner's two stills through the calibration of the day:
what is checked is whether tiles that are there are found, where they were found when the owner's arm landed on them,
and nothing else; never anything about a real camera. The page's requests run as the server calls them; heights are
the firmware's own (`with_raw_heights`), the desk level at arm z 44, as in `test_camera.py`, whose app this is.
"""

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

from maxarm import Camera                                     # noqa: E402
from maxarm.arm import MOVE_SLOW_MM_S                         # noqa: E402
from maxarm.geometry import HOME                              # noqa: E402
from applib import areas                                      # noqa: E402
from applib.tiles import Tile, find_tiles                     # noqa: E402
from applib.tiles.pose import square_corners                  # noqa: E402
from camera import feed as feed_module                        # noqa: E402
from camera import store as camera_store                      # noqa: E402
from tiles.api import VISIT_MM, VISIT_OVER_MM                 # noqa: E402
import tile_scene                                             # noqa: E402
from harness import check, with_raw_heights                   # noqa: E402
from test_calibration import legs, speed                      # noqa: E402
from test_camera import DESK_Z, app, patched                  # noqa: E402

SHOTS = HERE / "tile-shots"
STILL_CAMERA = SHOTS / "still-camera.json"
# Where the finder puts each still's tiles, desk mm: on the first, all eleven, the touching pair as two; on the second
# the four the arm, left down over the others, does not hide or cross with its hose.
STILL_TILES = {
    "still-1.png": [(-152.9, -109.0), (-183.9, -173.9), (-150.3, -148.1), (-107.5, -115.3), (-140.4, -211.8),
                    (-113.3, -176.6), (-102.2, -200.7), (-79.3, -159.2), (-61.4, -199.7), (-30.1, -166.2),
                    (-33.1, -237.0)],
    "still-2.png": [(-152.7, -109.0), (-140.1, -211.5), (-33.0, -236.8), (-19.0, -175.8)],
}


def drawn_layout():
    """Ten tiles in view of the drawn camera: two touching, two a millimetre apart, one right of the arm's centre
    line, two past the reach, at any turn."""
    pair, gap = math.radians(10.0), math.radians(-12.0)
    return [(-150.0, -150.0, 10.0), (-150.0 + 25 * math.cos(pair), -150.0 + 25 * math.sin(pair), 10.0),
            (-100.0, -200.0, -30.0), (-120.0, -100.0, 40.0), (-60.0, -230.0, 0.0), (-200.0, -60.0, 25.0),
            (-180.0, -210.0, -12.0), (-180.0 - 26 * math.sin(gap), -210.0 + 26 * math.cos(gap), -12.0),
            (60.0, -200.0, 5.0), (0.0, -280.0, 15.0)]


def matched(tiles, truth):
    """For each drawn tile, how far the nearest found one is, mm and degrees; and how many found match none."""
    offs = []
    for x, y, turn in truth:
        best = min(tiles, key=lambda tile: math.hypot(tile.x - x, tile.y - y)) if tiles else None
        offs.append((math.inf, math.inf) if best is None else
                    (math.hypot(best.x - x, best.y - y), abs((best.turn - turn + 45.0) % 90.0 - 45.0)))
    strays = [tile for tile in tiles if min(math.hypot(tile.x - x, tile.y - y) for x, y, _ in truth) > 3.0]
    return offs, strays


def test_drawn_tiles() -> bool:
    print("drawn tiles on wood and on plain grey: every one found where it was drawn, nothing else:")
    results = []
    for desk in ("wood", "grey"):
        scene = tile_scene.render(drawn_layout(), desk=desk)
        offs, strays = matched(find_tiles(scene.frame, Camera(scene.desk_to_pixel, tile_scene.SIZE)), scene.tiles)
        worst = max(off for off, _ in offs)
        results.append(check(f"{desk}: all {len(offs)} within 1 mm and 1 degree, none found that is not there",
                             worst < 1.0 and max(turn for _, turn in offs) < 1.0 and not strays,
                             f"mm {[round(off, 2) for off, _ in offs]}, strays {len(strays)}"))
    return all(results)


def test_drawn_tiles_on_paper() -> bool:
    print("drawn tiles on white paper, their far sides faint: those in reach found, within 2 mm, nothing else:")
    scene = tile_scene.render(drawn_layout(), desk="paper")
    in_reach = [tile for tile in scene.tiles if math.hypot(tile[0], tile[1]) <= areas.TILE_REACH_MM[1]]
    found = find_tiles(scene.frame, Camera(scene.desk_to_pixel, tile_scene.SIZE))
    offs, _ = matched(found, in_reach)
    _, strays = matched(found, scene.tiles)
    return check(f"all {len(in_reach)} in reach within 2 mm, none found that is not there",
                 max(off for off, _ in offs) < 2.0 and not strays,
                 f"mm {[round(off, 2) for off, _ in offs]}, strays {len(strays)}")


def test_the_owners_tiles() -> bool:
    print("the owner's stills, through the calibration of the day: each tile where it was found, nothing else:")
    results = []
    for name, expected in STILL_TILES.items():
        if not (SHOTS / name).exists() or not STILL_CAMERA.exists():
            print(f"  SKIP  agenttools/tile-shots/ has no {name} or its camera")
            continue
        tiles = find_tiles(cv2.imread(str(SHOTS / name)), Camera.load(STILL_CAMERA))
        offs = [min(math.hypot(tile.x - x, tile.y - y) for tile in tiles) for x, y in expected]
        results.append(check(f"{name}: {len(expected)} tiles, each within 0.5 mm of where it was found",
                             len(tiles) == len(expected) and max(offs) < 0.5,
                             f"{len(tiles)} found; mm {np.round(offs, 2)}"))
    return all(results)


def test_frame_size() -> bool:
    print("a frame of another size than the calibration's is refused: every pixel there means another place:")
    scene = tile_scene.render(drawn_layout()[:2])
    smaller = cv2.resize(scene.frame, (1296, 972))
    try:
        find_tiles(smaller, Camera(scene.desk_to_pixel, tile_scene.SIZE))
        refused = ""
    except ValueError as error:
        refused = str(error)
    return check("refused, saying both sizes", "1296 x 972" in refused and "2592 x 1944" in refused, refused)


def tile(x: float, y: float, turn: float = 0.0) -> Tile:
    corners = square_corners((x, y), math.radians(turn), 25.0)
    return Tile(x, y, turn, tuple(map(tuple, corners)), (0.0, 0.0), 9.0)


def test_areas() -> bool:
    print("the work areas: pickup the outer two thirds on the arm's left, dropoff the inner third; each tile placed:")
    always = lambda x, y, z: True                          # noqa: E731
    layout = [tile(-150.0, -150.0), tile(-150.0, -123.0), tile(-80.0, -200.0), tile(-50.0, -110.0),
              tile(150.0, -150.0), tile(-190.0, -190.0), tile(-60.0, -70.0), tile(-200.0, 150.0),
              tile(-120.0, -200.0), tile(-120.0, -226.0)]
    places = areas.place_tiles(layout, always)
    statuses = [place.status.value for place in places]
    unreachable = areas.place_tiles([tile(-150.0, -150.0)], lambda x, y, z: False)[0]
    corner = areas.gap_mm(tile(0.0, 0.0), tile(30.0, 30.0))
    return all([
        check("the line between them a third of the way out, the reach 5 mm inside the arm's both ways",
              areas.TILE_REACH_MM == (100.0, 265.0) and areas.DIVIDE_MM == 155.0, f"{areas.TILE_REACH_MM}, "
              f"{areas.DIVIDE_MM}"),
        check("a tile 2 mm clear of the next: ready; two 2 mm apart: both ready",
              statuses[0] == "ready" and statuses[1] == "ready" and abs(places[0].gap_mm - 2.0) < 1e-9,
              f"{statuses[:2]}, gap {places[0].gap_mm}"),
        check("in the dropoff area: elsewhere, saying so; right of the centre line: elsewhere",
              statuses[3] == "elsewhere" and places[3].area == "dropoff" and "dropoff" in places[3].detail
              and statuses[4] == "elsewhere" and places[4].area == "", f"{statuses[3:5]}, {places[3].detail}"),
        check("past 265 mm, inside 100, behind the arm: out of reach, saying why",
              statuses[5] == "out of reach" and "269 mm" in places[5].detail and statuses[6] == "out of reach"
              and statuses[7] == "out of reach" and "behind" in places[7].detail,
              f"{statuses[5:8]}: {places[5].detail}; {places[7].detail}"),
        check("1 mm apart in the pickup area: both too close, each naming the other",
              statuses[8:] == ["too close", "too close"] and places[8].nearest == 9 and places[9].nearest == 8
              and abs(places[8].gap_mm - 1.0) < 1e-9, f"{statuses[8:]}, {places[8].detail}"),
        check("a pose the library's can_reach() refuses: out of reach", unreachable.status.value == "out of reach"
              and "cannot hold" in unreachable.detail, unreachable.detail),
        check("the gap corner to corner, as if sharp: 5 x root 2 for two squares 30 mm apart both ways",
              abs(corner - 5 * math.sqrt(2)) < 1e-9, f"{corner:.4f}"),
    ])


def calibrated(config: Path, camera, desk: str = "wood") -> tile_scene.Scene:
    """camera.json the drawing's camera, as CALIBRATE would have written it, and the camera seeing the drawing."""
    scene = tile_scene.render(drawn_layout(), desk=desk)
    camera_store.save_camera(config, scene.fit)
    camera.frame = scene.frame
    return scene


@with_raw_heights
def test_locate_and_visit() -> bool:
    print("LOCATE TILES: the arm out of the picture, the snapshot, the tiles placed; a visit to one, down over it:")
    with patched(feed_module, "STARTING_FRAMES", 2), app() as (post, state, board, camera, config, server):
        uncalibrated = post("tiles/locate")
        before = state()
        scene = calibrated(config, camera)
        post("camera/jump", x=-100.0, y=-200.0)
        board.log.clear()
        located = post("tiles/locate")
        parked, snapshots = legs(board), camera.snapshots
        found = state()["tiles"]["found"]
        snapshot = server.byte_routes["/api/tiles/snapshot.jpg"]()
        ready = next(tile for tile in found["tiles"] if tile["status"] == "ready")
        visited = post("tiles/visit", n=ready["n"])
        down = legs(board)
        far = next(tile for tile in found["tiles"] if tile["status"] == "out of reach")
        refused = post("tiles/visit", n=far["n"])
        unmoved = not legs(board)
        camera.frame = cv2.resize(scene.frame, (1296, 972))
        resized = post("tiles/locate")
    drawn = [(x, y) for x, y, _ in scene.tiles]
    offs = [min(math.dist((tile["x"], tile["y"]), point) for point in drawn) for tile in found["tiles"]]
    heights = [round(point[2] - DESK_Z, 2) for point, _ in down]
    return all([
        check("without a camera.json: refused, and the page is not there", uncalibrated["status"] == "refused"
              and before["unavailable"].get("tiles") == before["tiles"]["problem"], uncalibrated["detail"]),
        check("home first, out of the picture, then one snapshot", parked and parked[-1][0] == HOME
              and snapshots == 1, f"{parked[-1:]}, {snapshots} snapshot(s)"),
        check("every drawn tile in view found, within 1 mm, and the counts said",
              len(found["tiles"]) == len(drawn) and max(offs) < 1.0 and located["detail"].startswith("10 tiles"),
              f"{located['detail']}; mm {np.round(offs, 2)}"),
        check("each placed: the touching pair too close, the one past 265 mm out of reach",
              sum(tile["status"] == "too close" for tile in found["tiles"]) >= 2 and far["reach"] > 265,
              f"{[tile['status'] for tile in found['tiles']]}"),
        check("the snapshot is there for the page, as a JPEG", snapshot and snapshot[1] == "image/jpeg"
              and cv2.imdecode(np.frombuffer(snapshot[0], np.uint8), cv2.IMREAD_COLOR).shape[:2] == (1944, 2592)),
        check("a visit: over the tile at 14 mm, slowly down to 6, 4 mm over its top, on its centre",
              visited["status"] == "done" and heights[-2:] == [VISIT_OVER_MM, VISIT_MM]
              and all(math.dist(point[:2], (ready["x"], ready["y"])) < 0.01 for point, _ in down[-2:])
              and abs(speed(down[-2][0], down[-1]) - MOVE_SLOW_MM_S) < 0.1, f"{down[-2:]}"),
        check("a tile out of reach: refused, nothing moved", refused["status"] == "refused" and unmoved,
              refused["detail"]),
        check("a snapshot of another size than the calibration's: refused", resized["status"] == "refused"
              and "1296 x 972" in resized["detail"], resized["detail"]),
    ])


def test_page_script() -> bool:
    """tiles.js run in node on the app's own states (drive-page.mjs): not calibrated, calibrated, tiles found."""
    print("the page's script, in node, on the app's own states: nothing thrown, and each shown as it should be:")
    node = shutil.which("node")
    if node is None:
        print("  SKIP  node is not on PATH")
        return True
    with patched(feed_module, "STARTING_FRAMES", 2), app() as (post, state, board, camera, config, server):
        limits = server.routes[("GET", "/api/limits")]({})
        states = [state()]
        calibrated(config, camera)
        states.append(state())
        post("tiles/locate")
        states.append(state())
    with tempfile.TemporaryDirectory() as scratch:
        replies = Path(scratch) / "replies.json"
        replies.write_text(json.dumps({"limits": json.loads(json.dumps(limits)), "states": states[1:],
                                       "clicks": ["locate"], "watch": ["counts", "locate", "tiles", "locate-line"]}))
        blocked = Path(scratch) / "blocked.json"
        blocked.write_text(json.dumps({"limits": json.loads(json.dumps(limits)), "states": states[:1],
                                       "watch": ["blocked-why", "locate"]}))
        runs = [subprocess.run([node, "--experimental-default-type=module", str(HERE / "drive-page.mjs"),
                                str(ROOT / "calibration" / "tiles" / "tiles.js"), str(path)],
                               capture_output=True, text=True, timeout=60) for path in (blocked, replies)]
    outputs = [json.loads(run.stdout) if run.returncode == 0 and run.stdout else {"errors": [run.stderr], "steps": []}
               for run in runs]
    errors = [error for output in outputs for error in output["errors"]]
    blocked_steps, steps = outputs[0]["steps"], outputs[1]["steps"]
    return all([
        check("nothing thrown", not errors, "; ".join(error.splitlines()[0] for error in errors)[:300]),
        check("not calibrated: the page says why, LOCATE TILES cannot be pressed", blocked_steps
              and "no camera calibration" in blocked_steps[0]["blocked-why"]["text"].lower()
              and blocked_steps[0]["locate"]["disabled"], f"{blocked_steps[:1]}"),
        check("calibrated, nothing found yet: the table says LOCATE TILES", steps
              and "LOCATE TILES to find them" in steps[0]["tiles"]["rows"] and not steps[0]["locate"]["disabled"]),
        check("found: a row per tile, a go for each in reach, the counts by status", len(steps) > 1
              and steps[1]["tiles"]["rows"].count("<tr ") == 10 and "too close" in steps[1]["counts"]["html"]
              and "out of reach" in steps[1]["counts"]["html"] and "10</b> tiles" in steps[1]["locate-line"]["html"],
              steps[1]["counts"]["html"][:200] if len(steps) > 1 else ""),
    ])


TESTS = [test_drawn_tiles, test_drawn_tiles_on_paper, test_the_owners_tiles, test_frame_size, test_areas,
         test_locate_and_visit, test_page_script]


if __name__ == "__main__":
    from harness import run
    sys.exit(run(TESTS, "tiles"))
