"""The arm's own controls, for any page. Arm space throughout.

| request             | body    | does                                                       |
| ------------------- | ------- | ---------------------------------------------------------- |
| `POST /api/connect` |         | open the port again: the board resets and homes the arm    |
| `POST /api/move`    | x, y, z | straight there; a newer move replaces one still waiting    |
| `POST /api/jump`    | x, y    | over to x, y at a safe height, and stay at that height     |
| `POST /api/cup`     | angle   | the cup's servo, -90 to 90 degrees                         |
| `POST /api/suction` | on      | the pump on or off, waited out                             |
| `POST /api/reset`   |         | stop everything, let go, lift clear and go home            |
| `GET /api/limits`   |         | reach, the base, the heights: for drawing and the sliders  |
| `GET /api/state`    |         | connection, what is running, where the arm was sent and is |
"""

import math

from maxarm import Space, geometry, nozzle, path
from maxarm.geometry import Position

from .motion import HOVER_MM
from .server import Server, number
from .worker import Worker

# The height slider's floor, the owner's (PILOT-falcon): just under the lowest the
# desk has read, 36 at full reach (condor). Going lower takes the LOW REACH button.
Z_MIN_MM = 34.0


def register(server: Server, worker: Worker) -> None:
    server.add_state(worker.state)
    server.add_limits(limits)
    server.post("/api/connect", lambda body: worker.connect())
    server.post("/api/reset", lambda body: worker.reset())

    def move(body: dict) -> dict:
        target = (number(body, "x"), number(body, "y"), number(body, "z"))
        return worker.run("move", f"moving to {describe(target)}",
                          lambda motion: motion.direct(Space.ARM, target), latest_only=True)

    def jump(body: dict) -> dict:
        x, y = number(body, "x"), number(body, "y")
        return worker.run("jump", f"over to ({x:.1f}, {y:.1f})",
                          lambda motion: motion.jump(Space.ARM, x, y), latest_only=True)

    def cup(body: dict) -> dict:
        angle = number(body, "angle", -nozzle.NOZZLE_RANGE_DEG, nozzle.NOZZLE_RANGE_DEG)
        return worker.run("cup", f"turning the cup to {angle:+.0f} deg",
                          lambda motion: motion.turn(angle), latest_only=True)

    def suction(body: dict) -> dict:
        is_on = bool(body.get("on"))
        return worker.run("suction", f"suction {'on' if is_on else 'off'}",
                          lambda motion: motion.suction(is_on))

    for name, handler in (("move", move), ("jump", jump), ("cup", cup), ("suction", suction)):
        server.post(f"/api/{name}", handler)


def limits() -> dict:
    return {
        "reach": path.REACH_MAX_MM,
        "fan": geometry.BASE_FAN_DEG,
        "base_half": path.BASE_LIMIT_HALF_MM,
        "home": list(geometry.HOME),
        "z_min": Z_MIN_MM,
        "z_low": lowest_z(),
        "z_max": geometry.HOME[2],
        "cup": nozzle.NOZZLE_RANGE_DEG,
        "hover": {space.value: height for space, height in HOVER_MM.items()},
    }


def lowest_z() -> float:
    """The LOW REACH floor: the lowest z the firmware takes at full reach, and so anywhere in reach."""
    z = float(math.floor(-geometry.L0 - geometry.L2 - geometry.L3))
    while geometry.limit_reason((0.0, -path.REACH_MAX_MM, z)) is not None:
        z += 1.0
    return z


def describe(point: Position) -> str:
    return "({:.1f}, {:.1f}, {:.1f})".format(*point)
