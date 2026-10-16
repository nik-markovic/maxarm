"""MaxArm control library: move, grab and drop, in whichever coordinates you work in.

    from maxarm import MaxArm, Space

    with MaxArm() as arm:
        arm.use_coordinates(Space.DESK)
        arm.set_clearance(14.0)
        arm.grab(120.0, -200.0, 4.0, angle=30.0)
        arm.drop(-60.0, -180.0, 5.0, angle=0.0)
        arm.home()

| module      | what it owns                                                   |
| ----------- | -------------------------------------------------------------- |
| `arm`       | `MaxArm`: moves, grab and drop, and how long each one takes    |
| `nozzle`    | the cup's turn: which way it faces, and how long turning takes |
| `path`      | the limits, and the path round them -- arithmetic, no board    |
| `spaces`    | coordinate spaces, and the calibration file behind each one    |
| `geometry`  | the firmware's kinematics, host-side                           |
| `protocol`  | everything said to the board                                   |
| `transport` | the serial port                                                |
| `tunable`   | the `MAXARM_*` environment overrides                           |
"""

from .arm import MaxArm, MoveResult, MoveStatus, MoveType, NotConnectedError
from .protocol import BoardNotReadyError, ProtocolError
from .spaces import Camera, Space, Spaces
from .transport import TransportError

__all__ = [
    "BoardNotReadyError", "Camera", "MaxArm", "MoveResult", "MoveStatus", "MoveType",
    "NotConnectedError", "ProtocolError", "Space", "Spaces", "TransportError",
]
