"""MaxArm control library.

    from maxarm import MaxArm, Limits

    with MaxArm(limits=Limits(x=(0.0, None))) as arm:
        arm.move_to(150.0, -150.0, 60.0)

Modules, innermost first. The first three need no hardware at all, which is
why the whole test suite runs in twenty seconds on a laptop:

| module      | what it owns                                                   |
| ----------- | -------------------------------------------------------------- |
| `geometry`  | kinematics and the limits the firmware enforces                |
| `zones`     | exclusion zones, as positions and as paths                     |
| `tuning`    | the measured constants, with their provenance                  |
| `route`     | retargeting, detours, staging -- where the arm goes, no board  |
| `transport` | bytes on a wire                                                |
| `protocol`  | what the board understands                                     |
| `motion`    | driving a route with the readback in the loop                  |
| `maxarm`    | `MaxArm`, the class callers use                                |

`route`, `motion` and `tuning` are internal: nothing in them is a decision the
caller makes. Background and evidence for every constant: `work/STATUS-bison.md`.
"""

from .config import ArmConfig, ConnectionMethod, Limits
from .geometry import JointFrame, Position, Unreachable
from .maxarm import MaxArm, NotConnectedError
from .protocol import BoardNotReadyError, BoardProtocol, ProtocolError, ReplProtocol
from .status import ArmState, MoveResult, MoveStatus, Step
from .transport import SerialTransport, Transport, TransportError
from .zones import BASE_ZONE, ExclusionZone

__all__ = [
    "ArmConfig", "ArmState", "BASE_ZONE", "BoardNotReadyError", "BoardProtocol",
    "ConnectionMethod", "ExclusionZone", "JointFrame", "Limits", "MaxArm",
    "MoveResult", "MoveStatus", "NotConnectedError", "Position", "ProtocolError",
    "ReplProtocol", "SerialTransport", "Step", "Transport", "TransportError",
    "Unreachable",
]
