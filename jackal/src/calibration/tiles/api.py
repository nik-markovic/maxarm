"""The Tiles page's requests: where the 25 mm tiles are, their letters, which the arm can pick up, and a word of them.

What finds, reads, places and carries the tiles is `applib/tilework.py`, which the
/IOTCONNECT app runs too; this is the page's way in to it. The tiles and the word
are drawn over the live view by `applib/overlay`, as a transparent layer the page
lays over the picture: the stream's overlay, the same code.

| request                       | body          | does                                                            |
| ----------------------------- | ------------- | --------------------------------------------------------------- |
| `POST /api/tiles/locate`      |               | park the arm, take the snapshot, find the tiles, read and place |
| `POST /api/tiles/visit`       | n             | the cup over tile n's centre, down to 4 mm over its top face    |
| `POST /api/tiles/place`       | letters, grip | the tiles that spell them carried to the dropoff arc, in order, |
|                               |               | each gripped with the cup at desk height `grip`, mm, now kept   |
| `POST /api/settings`          | name: value   | the app's settings, `config/appcfg.json`: `grip_mm`, `reader`   |
| `GET /api/tiles/scene.png`    | w, s, areas   | the tiles drawn for a picture `w` px wide at pixel ratio `s`;   |
|                               |               | the work areas unless `areas` is 0                              |

A visit tries the tile's centre: the cup comes down over it, and the person at the
arm sees whether it is centred on the tile. A tile the arm cannot reach is refused,
nothing moved. Until LOCATE TILES, the tiles a word moved cannot be visited.
"""

from pathlib import Path

from maxarm import MoveResult, Space

from applib import areas, word
from applib.camera import store
from applib.camera.feed import Feed
from applib.camera.fit import CameraFit
from applib.motion import Motion
from applib.overlay import draw_tiles, inks, picture_layer
from applib.settings import GRIP_RANGE_MM, Settings
from applib.tiles import TILE_HEIGHT_MM, TILE_MM, Tile
from applib.tilework import DROP_MM, TileWork
from common.server import BadRequest, Server, number
from common.worker import Refused, Worker

# A visit comes over the tile at the height the areas allow for, then down to 4 mm over its top face.
VISIT_OVER_MM = areas.OVER_TILE_MM
VISIT_MM = TILE_HEIGHT_MM + 4.0


class TilesSession:
    def __init__(self, worker: Worker, feed: Feed, config_dir: Path) -> None:
        self.worker = worker
        self.config_dir = Path(config_dir)
        self.work = TileWork(feed, config_dir)
        self.settings = Settings(config_dir)
        self.camera_file = store.CameraFile(config_dir)

    def register(self, server: Server) -> None:
        server.add_state(self.state)
        server.add_limits(lambda: {"tiles": {
            "tile": TILE_MM, "height": TILE_HEIGHT_MM, "visit": {"over": VISIT_OVER_MM, "down": VISIT_MM},
            "gap": areas.MIN_GAP_MM, "margin": areas.REACH_MARGIN_MM,
            "areas": [area.to_json() for area in (areas.PICKUP, areas.DROPOFF)],
            "word": {"arc": areas.WORD_ARC_MM, "gap": areas.WORD_GAP_MM, "most": areas.WORD_MAX_LETTERS,
                     "readers": list(areas.READERS), "drop": DROP_MM, "grip_range": list(GRIP_RANGE_MM)},
            "ink": inks.to_json()}})
        server.post("/api/tiles/locate", self.locate)
        server.post("/api/tiles/visit", self.visit)
        server.post("/api/tiles/place", self.place)
        server.post("/api/settings", self.set_settings)
        server.get_bytes("/api/tiles/scene.png", self.scene_image)

    # --- the requests -----------------------------------------------------

    def locate(self, body: dict) -> dict:
        return self.worker.run("tiles-locate", "LOCATE TILES: parking the arm, the snapshot, finding the tiles",
                               lambda motion: self.work.detect(motion, self._usable()))

    def visit(self, body: dict) -> dict:
        n = int(number(body, "n", 1))
        with self.work.lock:
            if not 1 <= n <= len(self.work.tiles):
                raise BadRequest(f"there is no tile {n}: LOCATE TILES found {len(self.work.tiles)}")
            tile, place = self.work.tiles[n - 1]
            is_moved = n in self.work.moved()
        if is_moved:
            return {"status": "refused", "detail": f"PLACE has carried tile {n} off: LOCATE TILES again"}
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

    def place(self, body: dict) -> dict:
        try:
            self.settings.set(grip_mm=number(body, "grip"))
        except ValueError as reason:
            raise BadRequest(str(reason)) from None
        grip, reader = self.settings.get("grip_mm"), self.settings.get("reader")
        try:
            fit = self._usable()
            steps, stamp = self.work.plan(str(body.get("letters", "")), grip, reader)
        except (Refused, word.WordRefused) as reason:
            return {"status": "refused", "detail": str(reason)}
        spelled = "".join(step.letter for step in steps)
        return self.worker.run("tiles-place", f"PLACE {spelled}: each tile to the dropoff arc in turn, gripped at "
                               f"{grip:g} mm, for a reader at the {reader}",
                               lambda motion: self.work.place(motion, steps, stamp, fit))

    def set_settings(self, body: dict) -> dict:
        try:
            values = self.settings.set(**body)
        except (TypeError, ValueError) as reason:
            raise BadRequest(str(reason)) from None
        return {"status": "done", "detail": ", ".join(f"{name} {values[name]}" for name in body)}

    def scene_image(self, query: dict):
        fit, _ = self.camera_file.get()
        if fit is None:
            return None
        try:
            width, scale = int(query.get("w", fit.frame_size[0])), float(query.get("s", 1.0))
        except ValueError:
            return None
        scene = self.work.scene()
        layer = picture_layer(fit, width, max(0.25, min(scale, 8.0)),
                              lambda canvas, view: draw_tiles(canvas, view, scene, query.get("areas") != "0"))
        return layer.png(), "image/png"

    # --- for the page -----------------------------------------------------

    def state(self) -> dict:
        fit, problem = self.camera_file.get()
        scene = self.work.scene()
        return {
            "tiles": {
                "problem": problem,
                "view_problem": self.work.view_problem(fit) if fit else None,
                "frame_size": list(fit.frame_size) if fit else None,
                "desk_to_view": fit.desk_to_shares.tolist() if fit else None,
                "desk_to_true": fit.desk_to_true.tolist() if fit else None,
                "found": scene["found"],
                "word": scene["word"],
            },
            "settings": self.settings.to_json(),
            "unavailable": {"tiles": problem} if problem else {},
        }

    def _usable(self) -> CameraFit:
        fit, problem = self.camera_file.get()
        if problem:
            raise Refused(problem)
        return fit
