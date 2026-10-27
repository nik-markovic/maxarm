"""The Tiles page's requests: where the 25 mm tiles are, and which of them the arm can pick up.

Desk space, through the saved `desk.json` or the default desk; the camera through
`camera.json`, as on the Fine Tune page. The tiles are found in the snapshot, the
camera's full-size frame -- the one the calibration was fitted at -- with the arm
parked out of the picture first, and placed in the work areas (`applib/areas.py`).

| request                       | body | does                                                            |
| ----------------------------- | ---- | --------------------------------------------------------------- |
| `POST /api/tiles/locate`      |      | park the arm, take the snapshot, find the tiles and place them  |
| `POST /api/tiles/visit`       | n    | the cup over tile n's centre, down to 4 mm over its top face    |
| `GET /api/tiles/snapshot.jpg` |      | the snapshot the tiles were found in                            |

A visit tries the tile's centre: the cup comes down over it, and the person at the
arm sees whether it is centred on the tile. A tile the arm cannot reach is refused,
nothing moved.
"""

import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from maxarm import Camera, MaxArm, MoveResult, Space

from applib import areas
from applib.tiles import TILE_HEIGHT_MM, TILE_MM, Tile, find_tiles
from camera import store
from camera.feed import Feed
from camera.fit import CameraFit
from camera.locate import SAME_SHAPE
from common.motion import Motion
from common.server import BadRequest, Server, number
from common.worker import Refused, Worker

# A visit comes over the tile at the height the areas allow for, then down to 4 mm over its top face.
VISIT_OVER_MM = areas.OVER_TILE_MM
VISIT_MM = TILE_HEIGHT_MM + 4.0
# The snapshot as the page is sent it.
SNAPSHOT_QUALITY = 85
COUNTED = (areas.Status.READY, areas.Status.TOO_CLOSE, areas.Status.ELSEWHERE, areas.Status.OUT_OF_REACH)


class TilesSession:
    def __init__(self, worker: Worker, feed: Feed, config_dir: Path) -> None:
        self.worker = worker
        self.feed = feed
        self.config_dir = Path(config_dir)
        self.lock = threading.Lock()
        self.found: Optional[dict] = None          # the last LOCATE TILES: the tiles, placed, and when
        self.tiles: List[Tuple[Tile, areas.Place]] = []
        self.snapshot: Optional[bytes] = None      # its frame, as a JPEG
        self._loaded: Tuple[Optional[int], Optional[CameraFit]] = (None, None)

    def register(self, server: Server) -> None:
        server.add_state(self.state)
        server.add_limits(lambda: {"tiles": {
            "tile": TILE_MM, "height": TILE_HEIGHT_MM, "visit": {"over": VISIT_OVER_MM, "down": VISIT_MM},
            "gap": areas.MIN_GAP_MM, "margin": areas.REACH_MARGIN_MM,
            "areas": [area.to_json() for area in (areas.PICKUP, areas.DROPOFF)]}})
        server.post("/api/tiles/locate", self.locate)
        server.post("/api/tiles/visit", self.visit)
        server.get_bytes("/api/tiles/snapshot.jpg", self._snapshot)

    # --- the requests -----------------------------------------------------

    def locate(self, body: dict) -> dict:
        return self.worker.run("tiles-locate", "LOCATE TILES: parking the arm, the snapshot, finding the tiles",
                               self._locate)

    def _locate(self, motion: Motion):
        fit = self._usable()
        parked = motion.reset()                     # out of the picture
        if not parked:
            return parked
        frame, described = self.feed.snapshot()
        camera = Camera(fit.desk_to_pixel, fit.frame_size)
        try:
            camera.check_frame(frame)
        except ValueError as reason:
            raise Refused(f"{reason}: the calibration holds for its own frame size only") from None
        started = time.monotonic()
        tiles = find_tiles(frame, camera)
        seconds = time.monotonic() - started
        places = areas.place_tiles(tiles, _arithmetic_arm(self.config_dir).can_reach)
        jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, SNAPSHOT_QUALITY])[1].tobytes()
        counts = {status.value: sum(place.status is status for place in places) for status in COUNTED}
        with self.lock:
            self.tiles = list(zip(tiles, places))
            self.snapshot = jpeg
            self.found = {
                "tiles": [{"n": index + 1, **tile.to_json(), **place.to_json(),
                           "nearest": None if place.nearest is None else place.nearest + 1}
                          for index, (tile, place) in enumerate(self.tiles)],
                "counts": counts, "taken": time.strftime("%H:%M:%S"), "stamp": time.time_ns(),
                "frame": [frame.shape[1], frame.shape[0]], "seconds": round(seconds, 2), "described": described,
            }
        return describe_counts(len(tiles), counts)

    def visit(self, body: dict) -> dict:
        n = int(number(body, "n", 1))
        with self.lock:
            if not 1 <= n <= len(self.tiles):
                raise BadRequest(f"there is no tile {n}: LOCATE TILES found {len(self.tiles)}")
            tile, place = self.tiles[n - 1]
        if place.status is areas.Status.OUT_OF_REACH:
            return {"status": "refused", "detail": f"tile {n} is out of the arm's reach: {place.detail}"}
        return self.worker.run(
            "tiles-visit", f"over tile {n}, desk ({tile.x:.1f}, {tile.y:.1f}), down to {VISIT_MM:.0f} mm",
            lambda motion: self._visit(motion, tile), latest_only=True)

    @staticmethod
    def _visit(motion: Motion, tile: Tile) -> MoveResult:
        result = motion.approach(Space.DESK, (tile.x, tile.y, VISIT_MM), precise=True, hover=VISIT_OVER_MM)
        if not result:
            return result
        return MoveResult(result.status, result.position,
                          f"the cup is {VISIT_MM - TILE_HEIGHT_MM:.0f} mm over the tile's centre, as the camera has it")

    # --- for the page -----------------------------------------------------

    def state(self) -> dict:
        fit, problem = self._calibration()
        with self.lock:
            found = self.found
        return {
            "tiles": {
                "problem": problem,
                "view_problem": self._view_problem(fit) if fit else None,
                "frame_size": list(fit.frame_size) if fit else None,
                "desk_to_view": fit.desk_to_shares.tolist() if fit else None,
                "desk_to_true": fit.desk_to_true.tolist() if fit else None,
                "found": found,
            },
            "unavailable": {"tiles": problem} if problem else {},
        }

    def _snapshot(self) -> Optional[Tuple[bytes, str]]:
        with self.lock:
            return (self.snapshot, "image/jpeg") if self.snapshot else None

    def _calibration(self) -> Tuple[Optional[CameraFit], Optional[str]]:
        """The saved camera calibration, read again only when the file changes; or why there is none."""
        path = self.config_dir / store.CAMERA_FILE
        if not path.exists():
            return None, "there is no camera calibration yet: CALIBRATE on the camera page"
        stamp = path.stat().st_mtime_ns
        if stamp != self._loaded[0]:
            self._loaded = (stamp, store.load_camera(self.config_dir))
        fit = self._loaded[1]
        return (fit, None) if fit else (None, "camera.json was not written by the camera page: CALIBRATE again")

    def _usable(self) -> CameraFit:
        fit, problem = self._calibration()
        if problem:
            raise Refused(problem)
        return fit

    def _view_problem(self, fit: CameraFit) -> Optional[str]:
        """Why the live view cannot show the tiles in place, if it cannot: it must be the calibration's picture."""
        size = self.feed.state()["size"]
        width, height = fit.frame_size
        if size is None or abs(size[0] * height / (size[1] * width) - 1) <= SAME_SHAPE:
            return None
        return (f"the live view is {size[0]} x {size[1]}, the calibration {width} x {height}: not the same "
                f"picture, so the tiles are drawn on the snapshot only")


def describe_counts(found: int, counts: dict) -> str:
    """'11 tiles: 7 ready to pick up, 2 too close, 1 outside the pickup area, 1 out of reach'."""
    if not found:
        return "no tiles found: are they in view, and the arm out of the picture?"
    words = {"ready": "ready to pick up", "too close": "too close", "elsewhere": "outside the pickup area",
             "out of reach": "out of reach"}
    parts = [f"{counts[status]} {words[status]}" for status in words if counts.get(status)]
    return f"{found} tile{'' if found == 1 else 's'}: {', '.join(parts)}"


def _arithmetic_arm(config_dir: Path) -> MaxArm:
    """A MaxArm in desk space that is never connected: `can_reach()` is arithmetic, and the port is not opened."""
    arm = MaxArm(config_dir=config_dir)
    arm.use_coordinates(Space.DESK)
    return arm
