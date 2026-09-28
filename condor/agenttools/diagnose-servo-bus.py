#!/usr/bin/env python3
"""Why did the arm accept a move and not make it? Reads only, no motion.

The symptom this exists for: `set_position()` returns `True`, the readback
never changes, and the arm stands still.

One deduction narrows it a long way. `get_position()` only returns a number if
a servo *replied with a valid checksum*, which means our query frame physically
reached it -- so the transmit path works, and a `SERVO_MOVE_TIME_WRITE` frame
reaches the servo too. The servo is therefore receiving move commands and
choosing not to act on them. That leaves three causes, and all three can be
read out without commanding anything:

  1. **Torque off.** `bus_servo.unload()` leaves a servo answering reads and
     ignoring moves. It survives until something loads it again.
  2. **Motor mode.** `set_mode(id, 1, speed)` turns the servo into a geared
     motor, where position commands are meaningless and ignored.
  3. **Supply voltage.** The servos run off the barrel jack, not USB. Low or
     absent servo power can leave them answering yet unable to drive.

`get_vin()` answers (3) directly. For (1) and (2) this script reports what it
can see and prints the one-line remedy rather than applying it -- both remedies
make a limp arm take up its own weight, which is motion, and motion is the
operator's call.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from maxarm import geometry                                   # noqa: E402
from maxarm.config import ArmConfig                           # noqa: E402
from maxarm.protocol import ReplProtocol                      # noqa: E402
from maxarm.transport import SerialTransport                  # noqa: E402

SERVO_IDS = (1, 2, 3)
ABSENT_ID = 5          # nothing is at this address: its reply proves the bus is honest

# LX-16A class servos want 6.0-8.4 V. Below about 6 V they answer and stall.
VIN_MIN_MV = 6000
VIN_NOMINAL_MV = 7400


def main() -> int:
    options = parse_arguments()
    config = ArmConfig(device=options.device).resolved()
    transport = SerialTransport(config.device, config.baud)
    protocol = ReplProtocol(transport)

    print(f"connecting to {config.device} (reads only -- nothing is commanded)")
    protocol.connect()
    try:
        positions = read_positions(protocol)
        voltages = read_voltages(protocol)
        identity = protocol.evaluate(f"bus_servo.get_ID({SERVO_IDS[0]})")
        cached = protocol.get_commanded_position()
        measured = protocol.read_position()
    finally:
        protocol.disconnect()

    report(positions, voltages, identity, cached, measured)
    return 0


def read_positions(protocol: ReplProtocol) -> dict:
    """Every servo, plus one that does not exist.

    The absent id is the control. If it answers with a number too, the replies
    are not coming from servos and nothing else in this report means anything.
    """
    ids = SERVO_IDS + (ABSENT_ID,)
    reply = protocol.evaluate(f"[bus_servo.get_position(i) for i in {ids}]")
    return dict(zip(ids, reply))


def read_voltages(protocol: ReplProtocol) -> dict:
    reply = protocol.evaluate(f"[bus_servo.get_vin(i) for i in {SERVO_IDS}]")
    return dict(zip(SERVO_IDS, reply))


def report(positions: dict, voltages: dict, identity, cached, measured) -> None:
    print("\nservo bus:")
    for servo_id in SERVO_IDS:
        pulse = positions.get(servo_id)
        volts = voltages.get(servo_id)
        volts_text = (f"{volts / 1000.0:.2f} V" if isinstance(volts, int) and volts
                      else "no answer")
        print(f"  servo {servo_id}: position {_value(pulse):>7}   supply {volts_text}")
    print(f"  servo {ABSENT_ID} (not fitted): {_value(positions.get(ABSENT_ID))}"
          "   <- must be False for the rest to mean anything")
    print(f"  get_ID(1) -> {_value(identity)}")

    print(f"\npose: board commanded {_format(cached)}, servos report {_format(measured)}")
    if measured is not None:
        try:
            pulses = geometry.position_to_pulses(measured)
            print(f"      which is pulses {_format(pulses)}")
        except geometry.Unreachable:
            print("      which does not solve -- the readback is not a real pose")

    print("\nreading:")
    if positions.get(ABSENT_ID) is not False:
        print("  ** The absent servo answered. The bus is echoing or the replies are")
        print("     defaults, so the position readback is not evidence of anything.")
        return

    live = [servo_id for servo_id in SERVO_IDS if positions.get(servo_id) is not False]
    if len(live) < len(SERVO_IDS):
        silent = sorted(set(SERVO_IDS) - set(live))
        print(f"  ** Servos {silent} did not answer at all. Check the cabling to them;")
        print("     a servo that cannot be read cannot be driven either.")
        return

    low = [servo_id for servo_id, volts in voltages.items()
           if not isinstance(volts, int) or volts < VIN_MIN_MV]
    if low:
        print(f"  ** Servo supply is low or unreadable on {low}.")
        print(f"     These want about {VIN_NOMINAL_MV / 1000:.1f} V from the barrel jack --")
        print("     USB alone powers the ESP32 but not the servos. Check the DC supply")
        print("     and its switch. A servo on low volts answers reads and will not move.")
        return

    print("  All three servos answer, and their supply is healthy. Since a reply proves")
    print("  our frames reach them, they are receiving move commands and refusing to act.")
    print("  That leaves torque or mode, neither of which is readable on this firmware:")
    print()
    print("    torque off (bus_servo.unload was called and never undone), or")
    print("    motor mode (set_mode(id, 1, ...) makes position commands meaningless).")
    print()
    print("  Both remedies are one line, and both make a limp arm take up its own weight,")
    print("  so hold the arm before running either:")
    print()
    print("    ./restore-servos.py            # loads torque and forces servo mode")
    print()
    print("  If neither helps, power-cycle the arm at the barrel jack: servo state is")
    print("  volatile and a cold start restores both.")


def _value(reply) -> str:
    return "False" if reply is False or reply is None else str(reply)


def _format(values) -> str:
    if values is None:
        return "(none)"
    return "(" + ", ".join(f"{value:.1f}" for value in values) + ")"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default=None, help="serial port (default /dev/ttyUSB0)")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
