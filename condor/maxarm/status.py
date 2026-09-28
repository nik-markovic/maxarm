#!/usr/bin/env python3
"""What a move returns, and what a UI reads. Nothing here does anything.

Nine statuses, and the test for whether there should be a tenth is whether the
caller would *do* something different about it. A servo that clamped silently
and an obstruction on the desk both mean "it stopped early, look at the arm",
so they are one status with the difference in `detail`.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

Position = Tuple[float, float, float]


class MoveStatus(Enum):
    """Why a move ended. Only REACHED means "did exactly what you asked"."""

    REACHED = "reached"
    # Arrived, but at the nearest legal point rather than the one requested --
    # the edge of the envelope, or an operator limit. `residual_mm` says how
    # far off the request that leaves it.
    APPROXIMATED = "approximated"

    # --- refused, nothing sent to the board ---
    UNREACHABLE = "unreachable"       # no legal point near enough to aim at
    NO_FLY = "no_fly"                 # target, or every route to it, crosses a zone
    NOT_CONNECTED = "not_connected"   # call connect() first

    # --- started moving, then stopped ---
    DESK_CONTACT = "desk_contact"     # the nozzle touched down before the target
    # Moved, then stopped advancing: an obstruction, a silently clamped servo,
    # or a waypoint the board refused mid-path. `detail` says which.
    BLOCKED = "blocked"
    # Accepted every command and never moved at all. Not a limit -- the servos
    # are not listening. Torque off, motor mode, or servo power missing, none
    # of which is readable on this firmware. Nothing else will work until it is
    # fixed, so callers should stop rather than retry.
    NOT_RESPONDING = "not_responding"
    STOPPED = "stopped"               # stop() was called, or Ctrl-C

    @property
    def is_ok(self) -> bool:
        return self is MoveStatus.REACHED

    @property
    def is_moved(self) -> bool:
        """False for the three statuses that are refusals rather than outcomes."""
        return self not in _REFUSALS


_REFUSALS = frozenset({MoveStatus.UNREACHABLE, MoveStatus.NO_FLY,
                       MoveStatus.NOT_CONNECTED})


@dataclass
class Step:
    """One commanded increment and what came back. Handy for plots and logs."""

    commanded: Position
    measured: Optional[Position]
    status: MoveStatus


@dataclass
class MoveResult:
    """Outcome of one `move_to()`. Falsy unless the arm arrived.

    `requested` is what the caller asked for, `target` what was actually aimed
    at after any retargeting, and `position` where the arm ended up by
    readback -- never by assumption, because near the envelope edge the command
    and the pose differ by several millimetres.
    """

    status: MoveStatus
    requested: Position
    target: Position
    position: Position
    detail: str = ""
    steps: List[Step] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.status.is_ok

    @property
    def is_ok(self) -> bool:
        return self.status.is_ok

    @property
    def residual_mm(self) -> float:
        """Distance from what the caller asked for to where the arm stands."""
        return _distance(self.requested, self.position)

    @property
    def error_mm(self) -> float:
        """Distance from the aimed-at target -- the arm's own tracking error."""
        return _distance(self.target, self.position)

    def __str__(self) -> str:
        pose = "({:.1f}, {:.1f}, {:.1f})".format(*self.position)
        text = f"{self.status.value} at {pose}, {self.residual_mm:.1f} mm from request"
        return f"{text} -- {self.detail}" if self.detail else text


@dataclass(frozen=True)
class ArmState:
    """One coherent snapshot, for a status bar, a 3D view or a log line.

    `joints` and `pulses` are what the servos report rather than what was
    commanded, so a render built from them shows the arm as it is -- sag
    included.
    """

    is_connected: bool
    position: Optional[Position]
    joints: Optional[Tuple[float, float, float]] = None
    pulses: Optional[Tuple[float, float, float]] = None
    is_suction_on: bool = False
    nozzle_angle_deg: float = 0.0
    last_status: Optional[MoveStatus] = None


def _distance(a: Position, b: Position) -> float:
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5
