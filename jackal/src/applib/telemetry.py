"""What the device tells the cloud, kept in one small object nobody waits on (the property-alarm demo's design).

The values are made on the command threads -- a detection's letters, the tiles in
trouble -- and sent on one thread only, `iotc.py`'s publisher. The producers write
here and the publisher reads here, so no command waits on the network and the app
runs the same with no cloud at all.

    telemetry.set(tiles="team", err_no_reach="", err_too_close="ea")   # sent on every tick from now on
    telemetry.wake()                                                    # and now, not at the end of the tick

A value of None is left out of the message, not sent as null: a missing value
says "no reading this time", and the dashboard keeps the last one.
"""

from __future__ import annotations

from threading import Event, Lock
from typing import Dict, Union

TelemetryValue = Union[str, float, int, bool, None]


class TelemetryState:
    def __init__(self) -> None:
        self._values: Dict[str, TelemetryValue] = {}
        self._lock = Lock()
        self._wake = Event()

    def set(self, **values: TelemetryValue) -> None:
        """The latest of one or more values: sent on the next tick, and every tick after."""
        with self._lock:
            self._values.update(values)

    def wake(self) -> None:
        """Ask the publisher to send now instead of at the end of its interval."""
        self._wake.set()

    def wait_for_wake(self, timeout_s: float) -> bool:
        """The publisher's sleep: True if it was woken rather than timed out."""
        is_woken = self._wake.wait(timeout_s)
        self._wake.clear()
        return is_woken

    def collect(self) -> Dict[str, TelemetryValue]:
        """Everything to send now, the values that are None left out."""
        with self._lock:
            return {name: value for name, value in self._values.items() if value is not None}
