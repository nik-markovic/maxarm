"""The arm's own controls, for any page. Arm space throughout.

| request             | body    | does                                                       |
| ------------------- | ------- | ---------------------------------------------------------- |
| `POST /api/connect` |         | open the port again: the board resets and homes the arm    |
| `POST /api/move`    | x, y, z | straight there, corrected; a newer move replaces a waiting one |
| `POST /api/jump`    | x, y    | over to x, y at a safe height, and stay at that height     |
| `POST /api/land`    | x, y, z | down to x, y, z as the library lands, aimed past what the servos fall short; across no lower than the cup already is, as a jump |
| `POST /api/cup`     | angle   | the cup's servo, -90 to 90 degrees                         |
| `POST /api/suction` | on      | the pump on or off, waited out; on, whether the cup sealed |
| `POST /api/reset`   |         | stop everything, let go, lift clear and go home            |
| `GET /api/limits`   |         | reach, the base, the heights: for drawing and the sliders  |
| `GET /api/state`    |         | connection, what is running, where the arm was sent and is |

Positions are true, both ways. A page that shows the firmware's raw ones converts
with `true_z` and `true_xy` from the limits, the library's own constants
(`maxarm/raw.py`).
"""

from maxarm import Space, geometry, nozzle, path, raw
from maxarm.arm import home_position
from maxarm.geometry import Position

from applib.motion import HOVER_MM

from .server import Server, number
from .worker import Worker

# A landing from the desk page crosses ABOVE_TARGET_MM over its target, never under CROSS_SAFE_MM (the page's SAFE,
# clear of 40 mm cubes) -- or, when the cup is already lower, at the cup's own height, never under the hover. Higher
# than that the turn aside can ask the elbow for more than it has.
ABOVE_TARGET_MM = 20.0
CROSS_SAFE_MM = 90.0


def register(server: Server, worker: Worker) -> None:
    server.add_state(worker.state)
    server.add_limits(limits)
    server.post("/api/connect", lambda body: worker.connect())
    server.post("/api/reset", lambda body: worker.reset())

    def move(body: dict) -> dict:
        target = (number(body, "x"), number(body, "y"), number(body, "z"))
        return worker.run("move", f"moving to {describe(target)}",
                          lambda motion: motion.direct(Space.ARM, target, precise=True), latest_only=True)

    def jump(body: dict) -> dict:
        x, y = number(body, "x"), number(body, "y")
        return worker.run("jump", f"over to ({x:.1f}, {y:.1f})",
                          lambda motion: motion.jump(Space.ARM, x, y), latest_only=True)

    def land(body: dict) -> dict:
        target = (number(body, "x"), number(body, "y"), number(body, "z"))
        return worker.run("land", f"landing on {describe(target)}",
                          lambda motion: motion.down_to(Space.ARM, target, hover=crossing(motion, target)),
                          latest_only=True)

    def cup(body: dict) -> dict:
        angle = number(body, "angle", -nozzle.NOZZLE_RANGE_DEG, nozzle.NOZZLE_RANGE_DEG)
        return worker.run("cup", f"turning the cup to {angle:+.0f} deg",
                          lambda motion: motion.turn(angle), latest_only=True)

    def suction(body: dict) -> dict:
        is_on = bool(body.get("on"))
        return worker.run("suction", f"suction {'on' if is_on else 'off'}",
                          lambda motion: motion.suction(is_on))

    for name, handler in (("move", move), ("jump", jump), ("land", land), ("cup", cup), ("suction", suction)):
        server.post(f"/api/{name}", handler)


def limits() -> dict:
    return {
        "reach": path.REACH_MAX_MM,
        "reach_min": path.REACH_MIN_MM,
        "fan": geometry.BASE_FAN_DEG,
        "base_half": path.BASE_LIMIT_HALF_MM,
        "home": list(home_position()),
        "cup": nozzle.NOZZLE_RANGE_DEG,
        "hover": {space.value: height for space, height in HOVER_MM.items()},
        "true_z": {"per_mm_of_reach": raw.TRUE_Z_PER_MM_OF_REACH, "level_reach": raw.LEVEL_REACH_MM,
                   "desk_z": raw.DESK_Z_MM, "stack": raw.STACK_MM, "tall_at_stack": raw.TALL_AT_STACK_MM,
                   "tall_per_mm_of_reach": raw.TALL_PER_MM_OF_REACH, "desk_wave": raw.DESK_WAVE_MM},
        "true_xy": {"reach_at_desk": raw.REACH_AT_DESK_MM, "reach_less_per_mm": raw.REACH_LESS_PER_MM,
                    "reach_top": raw.REACH_TOP_MM, "turn": raw.TURN_DEG,
                    "bearing_rounds": raw.BEARING_ROUNDS},
    }


def crossing(motion, target: Position) -> float:
    """The height a desk-page landing crosses at (ABOVE_TARGET_MM, CROSS_SAFE_MM)."""
    over = max(target[2] + ABOVE_TARGET_MM, CROSS_SAFE_MM)
    return max(HOVER_MM[Space.ARM], min(motion.position(Space.ARM)[2], over))


def describe(point: Position) -> str:
    return "({:.1f}, {:.1f}, {:.1f})".format(*point)
