#!/usr/bin/env python3
"""Everything the caller configures. Four fields, and all of them optional.

    MaxArm()                                  # the desk this was written for
    MaxArm(limits=Limits(x=(0.0, None)))      # no room left of the centre line
    MaxArm(ArmConfig(device="/dev/ttyUSB1"))

Any field left as `None` takes its default, so overriding one thing never
means restating the rest.

What is deliberately *not* here: step sizes, speeds, settle times, margins and
sag thresholds. Those are measurements rather than preferences and they live in
`tuning.py`, next to the comment that says where the number came from. Five of
them answer to an environment variable if a different desk needs them.
"""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Sequence, Tuple

from .tuning import DEFAULT_BAUD, DEFAULT_DEVICE, DEFAULT_Z_FLOOR
from .zones import ExclusionZone

AxisLimit = Tuple[Optional[float], Optional[float]]
Position = Tuple[float, float, float]


class ConnectionMethod(Enum):
    """How to reach the board. Only USB is implemented on stock firmware."""

    USB = "usb"      # MicroPython REPL over the on-board USB-serial bridge
    UART = "uart"    # 4-pin header via an FTDI dongle -- needs a firmware swap
    BLE = "ble"      # Hiwonder's 55 55 jog protocol -- one-way, jog-only


@dataclass(frozen=True)
class Limits:
    """Operator bounds, as (min, max) per axis. `None` on either end is open.

    These are *not* the arm's limits -- the arm's are computed in `geometry.py`
    and cannot be relaxed. These are the room's: how much desk there is to the
    left, how close the nozzle may get to the surface. Only the z floor has a
    default, because scraping the cup is the one mistake that costs hardware.
    """

    x: AxisLimit = (None, None)
    y: AxisLimit = (None, None)
    z: AxisLimit = (DEFAULT_Z_FLOOR, None)

    def clamp(self, position: Position) -> Position:
        return tuple(_clamp(value, bound)
                     for value, bound in zip(position, (self.x, self.y, self.z)))

    def contains(self, position: Position) -> bool:
        return all(_is_within(value, bound)
                   for value, bound in zip(position, (self.x, self.y, self.z)))

    @property
    def z_floor(self) -> float:
        return self.z[0] if self.z[0] is not None else 0.0


@dataclass(frozen=True)
class ArmConfig:
    """Where the arm is, and what the room around it forbids."""

    connection: Optional[ConnectionMethod] = None   # default USB
    device: Optional[str] = None                    # default /dev/ttyUSB0
    limits: Optional[Limits] = None                 # default: z floor only
    # Operator exclusion zones -- a camera mount, a mug, the edge of the mat.
    # The square around the robot's own base is always present on top of these
    # and cannot be switched off; see `zones.BASE_ZONE`.
    zones: Sequence[ExclusionZone] = ()

    def resolved(self) -> "ArmConfig":
        """A copy with no `None` left, so the library never tests for one."""
        return replace(
            self,
            connection=self.connection or ConnectionMethod.USB,
            device=self.device or DEFAULT_DEVICE,
            limits=self.limits or Limits(),
            zones=tuple(self.zones),
        )

    @property
    def baud(self) -> int:
        return DEFAULT_BAUD


def _clamp(value: float, bound: AxisLimit) -> float:
    low, high = bound
    if low is not None:
        value = max(value, low)
    if high is not None:
        value = min(value, high)
    return value


def _is_within(value: float, bound: AxisLimit) -> bool:
    low, high = bound
    return (low is None or value >= low) and (high is None or value <= high)
