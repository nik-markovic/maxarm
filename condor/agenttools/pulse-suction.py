#!/usr/bin/env python3
"""Pulse the suction cup on and off, forever. No motion at all.

For watching and listening to the pump on its own, with nothing else going on.
`set_position()` is never called; the only movement is the `go_home()` the
firmware runs on every port-open reset.

    ../.venv/bin/python agenttools/pulse-suction.py          # what the library does
    ../.venv/bin/python agenttools/pulse-suction.py --vent    # plus a second vent pulse
    ../.venv/bin/python agenttools/pulse-suction.py -p 5      # five seconds a phase

By default this sends exactly what `release()` sends -- `nozzle.on()` and
`nozzle.off()`, the same two calls Hiwonder's PC software makes -- with the
library's waits around them.

`--vent` adds the second vent pulse that `release()` used to do and no longer
does, because it measured worse on the arm. It is kept here as the A/B: if
cubes start sticking to the cup again, run both and watch which one lets go.
`protocol.VENT_CUP` has the mechanism and why it is suspected.

Ctrl-C to stop; it turns the pump off on the way out.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maxarm import tuning                                     # noqa: E402
from maxarm.config import ArmConfig                           # noqa: E402
from maxarm.maxarm import MaxArm                              # noqa: E402
from maxarm.protocol import RELEASE_HOLD_MS, ReplProtocol     # noqa: E402
from maxarm.status import MoveStatus                          # noqa: E402
from maxarm.transport import SerialTransport, TransportError  # noqa: E402

# How long to hold the vent valve open a second time. `release()` used to do
# this and no longer does, so the number lives here with its only caller
# rather than in the library's tuning table.
VENT_PULSE_MS = 400


def main() -> int:
    options = parse_arguments()
    config = ArmConfig(device=options.device).resolved()
    transport = SerialTransport(config.device, config.baud)
    protocol = ReplProtocol(transport)
    # `MaxArm` shares the transport and protocol, so `--at` gets the library's
    # routing and limits while the pulsing below still goes straight to the
    # board with nothing of the library's timing in the way.
    arm = MaxArm(config, transport=transport, protocol=protocol)

    print(f"connecting to {config.device}; this resets the board and homes the arm")
    started = time.monotonic()
    try:
        arm.connect()
    except TransportError as error:
        print(f"no link to the arm: {error}")
        return 1
    print(f"up in {time.monotonic() - started:.1f} s, "
          f"{'with the extra vent' if options.is_vented else 'as release() does'}, "
          f"{options.period_s:.1f} s a phase. Ctrl-C to stop.")
    if options.at is not None:
        result = arm.move_to(*options.at)
        print(f"  moved to {_point(result.position)}: {result}")
        if result.status not in (MoveStatus.REACHED, MoveStatus.APPROXIMATED):
            return 1
    print()

    count = 0
    try:
        while options.cycles == 0 or count < options.cycles:
            count += 1
            pulse(protocol, count, is_on=True, is_vented=options.is_vented)
            settle(protocol, options.period_s)
            pulse(protocol, count, is_on=False, is_vented=options.is_vented)
            settle(protocol, options.period_s)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        # Whatever state the loop was interrupted in, leave the pump off and
        # the cup vented rather than holding onto whatever it had.
        protocol.set_suction(False)
        time.sleep(RELEASE_HOLD_MS / 1000.0)
        arm.disconnect()
    return 0


def pulse(protocol: ReplProtocol, count: int, is_on: bool, is_vented: bool) -> None:
    """One switch, with the tip read either side of it.

    The reading is the point. The cup is a bellows that contracts only against
    a seal, so what the tip does when the pump starts is the one piece of
    evidence this firmware can give about whether anything is held -- and
    whether the arm itself moves when it happens. Readback is whole
    millimetres, so a one-millimetre change is at the noise floor and only a
    repeated one means anything.
    """
    before = protocol.read_position()
    started = time.monotonic()
    protocol.set_suction(is_on)
    commanded_ms = (time.monotonic() - started) * 1000.0

    time.sleep(tuning.SUCTION_SETTLE_MS / 1000.0)
    detail = f"pump command {commanded_ms:4.0f} ms"
    if not is_on:
        # The board opens its vent valve for a second on a thread of its own.
        time.sleep(max(0.0, RELEASE_HOLD_MS - tuning.SUCTION_SETTLE_MS) / 1000.0)
        detail += f", valve held {RELEASE_HOLD_MS} ms"
        if is_vented:
            protocol.vent_cup(VENT_PULSE_MS)
            detail += f", vented again {VENT_PULSE_MS} ms"

    after = protocol.read_position()
    print(f"{count:4}  {'ON ' if is_on else 'OFF'}  "
          f"tip {_point(before)} -> {_point(after)}  {_delta(before, after)}  {detail}")


def settle(protocol: ReplProtocol, seconds: float) -> None:
    """Hold the phase, then read once more -- the late reading is the one that
    catches anything that happens a second after the switch."""
    time.sleep(seconds)
    print(f"      ..    tip {_point(protocol.read_position())} after {seconds:.1f} s")


def _point(position) -> str:
    return "(  ?,    ?,   ?)" if position is None else "({:3.0f}, {:4.0f}, {:3.0f})".format(*position)


def _delta(before, after) -> str:
    if before is None or after is None:
        return "dz    ?"
    return f"dz {after[2] - before[2]:+3.0f} mm"


def _position(text: str):
    x, y, z = (float(part) for part in text.split(","))
    return (x, y, z)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vent", dest="is_vented", action="store_true",
                        help="add the second vent pulse release() no longer does")
    parser.add_argument("-p", "--period", dest="period_s", type=float, default=2.0,
                        help="seconds to hold each phase (default 2)")
    parser.add_argument("--at", type=_position, default=None, metavar="X,Y,Z",
                        help="move here first; otherwise the arm is never commanded")
    parser.add_argument("-n", "--cycles", type=int, default=0,
                        help="stop after this many on/off cycles (default: forever)")
    parser.add_argument("-d", "--device", default=None,
                        help="serial port; defaults to $MAXARM_DEVICE or /dev/ttyUSB0")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
