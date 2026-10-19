#!/usr/bin/env python3
"""How the arm lands when told to move: big moves, then small steps, in servo pulses as well as millimetres.

    ../.venv/bin/python agenttools/probe-lag.py              # all phases, around the centre below
    ../.venv/bin/python agenttools/probe-lag.py steps        # one phase: large, steps, approach, overshoot

Everything stays within 30 mm of `CENTRE`, 40 mm below the reset pose and far above the desk.
After every move it reads the servos' own pulses (median of five, after settling) and logs, per
servo, the pulse sent against the pulse reached, and the tip that gives by the firmware's kinematics.
Lines go to agenttools/logs/ as they happen; Ctrl-C holds the arm where it is.
"""

import json
import sys
import time
from pathlib import Path
from statistics import median
from typing import List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from maxarm import geometry                                  # noqa: E402
from maxarm.protocol import ReplProtocol                     # noqa: E402
from maxarm.transport import DEVICE, SerialTransport         # noqa: E402

CENTRE = (0.0, -163.0, 180.0)
LARGE_MM = 30.0
STEPS_MM = 6
SETTLE_S = 0.8
READS = 5


class Probe:
    def __init__(self, log_path: Path) -> None:
        self.protocol = ReplProtocol(SerialTransport(DEVICE))
        self.log = log_path.open("a")
        self.target = geometry.HOME

    def send(self, target, ms: int, note: str) -> dict:
        problem = geometry.limit_reason(target)
        if problem:
            raise RuntimeError(f"{target}: {problem}")
        sent = [int(pulse) for pulse in geometry.deg_to_pulse(geometry.inverse(target))]
        accepted = self.protocol.set_position(target, ms)
        time.sleep(ms / 1000.0 + SETTLE_S)
        pulses = self.read_pulses()
        tip = geometry.forward(geometry.pulse_to_deg(pulses))
        entry = {"note": note, "target": list(target), "ms": ms, "accepted": accepted, "sent": sent,
                 "reached": pulses, "off_pulses": [r - s for r, s in zip(pulses, sent)],
                 "tip": [round(value, 2) for value in tip],
                 "off_mm": [round(t - g, 2) for t, g in zip(tip, target)]}
        self.log.write(json.dumps(entry) + "\n")
        self.log.flush()
        print(f"  {note:28} off mm {entry['off_mm']}  pulses sent {sent} reached {pulses}", flush=True)
        self.target = target
        return entry

    def read_pulses(self) -> List[float]:
        samples = []
        for _ in range(READS):
            reply = self.protocol.evaluate("[bus_servo.get_position(i) for i in (1, 2, 3)]")
            if all(isinstance(value, int) for value in reply):
                samples.append(reply)
        return [median(sample[i] for sample in samples) for i in range(3)]


def offset(axis: int, mm: float) -> Tuple[float, float, float]:
    point = list(CENTRE)
    point[axis] += mm
    return tuple(point)


def large(probe: Probe) -> None:
    print("large moves, 30 mm out and back on each axis:")
    probe.send(CENTRE, 1500, "centre")
    for axis, name in enumerate("xyz"):
        for sign in (1, -1):
            probe.send(offset(axis, sign * LARGE_MM), 1000, f"{name} {sign * LARGE_MM:+.0f}")
            probe.send(CENTRE, 1000, f"back from {name} {sign * LARGE_MM:+.0f}")


def steps(probe: Probe) -> None:
    print("1 mm steps from rest, out and back, as the owner did:")
    for axis, name in enumerate("xyz"):
        probe.send(CENTRE, 1000, f"centre before {name}")
        for mm in list(range(1, STEPS_MM + 1)) + list(range(STEPS_MM - 1, -STEPS_MM - 1, -1)) + list(
                range(-STEPS_MM + 1, 1)):
            probe.send(offset(axis, float(mm)), 300, f"{name} {mm:+d}")


def approach(probe: Probe) -> None:
    print("the centre, approached from either side, three times each:")
    for _ in range(3):
        for axis, name in enumerate("xyz"):
            for sign in (1, -1):
                probe.send(offset(axis, sign * 15.0), 600, f"{name} {sign * 15:+d} side")
                probe.send(CENTRE, 600, f"centre from {name} {sign * 15:+d}")


def overshoot(probe: Probe) -> None:
    print("a 1 mm step made by going 5 mm past it and back:")
    for axis, name in enumerate("xyz"):
        probe.send(CENTRE, 800, f"centre before {name}")
        for mm in (1, 2, 3):
            probe.send(offset(axis, mm + 5.0), 300, f"{name} past {mm + 5:+d}")
            probe.send(offset(axis, float(mm)), 300, f"{name} {mm:+d} from above")


def precise(probe: Probe) -> None:
    """The owner's 1 mm steps again, each followed by corrections: send target + error until it lands."""
    print("1 mm steps, each corrected from the servos' own reading:")
    for axis, name in enumerate("xyz"):
        probe.send(CENTRE, 1000, f"centre before {name}")
        for mm in list(range(1, 5)) + list(range(3, -5, -1)) + list(range(-3, 1)):
            target = offset(axis, float(mm))
            entry = probe.send(target, 300, f"{name} {mm:+d}")
            error = [-value for value in entry["off_mm"]]
            for round_ in range(1, PRECISE_ROUNDS + 1):
                if max(abs(value) for value in error) < PRECISE_TOLERANCE_MM:
                    break
                aim = tuple(t + max(-PRECISE_CAP_MM, min(PRECISE_CAP_MM, e)) for t, e in zip(target, error))
                entry = probe.send(aim, 150, f"  correction {round_}")
                previous, error = error, [t - g for t, g in zip(target, entry["tip"])]
                if max(map(abs, error)) > 0.8 * max(map(abs, previous)):
                    break
            print(f"    -> {name} {mm:+d} lands off by {[round(value, 2) for value in error]}", flush=True)


PRECISE_ROUNDS = 3
PRECISE_TOLERANCE_MM = 0.4
PRECISE_CAP_MM = 5.0

PHASES = {"large": large, "steps": steps, "approach": approach, "overshoot": overshoot, "precise": precise}


def main(argv: List[str]) -> int:
    wanted = argv[1:] or list(PHASES)
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / f"probe-lag-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    probe = Probe(log_path)
    print(f"connecting (the board resets and homes the arm); log: {log_path.relative_to(HERE.parent)}")
    probe.protocol.connect()
    probe.protocol.check_board()
    time.sleep(1.0)
    try:
        for name in wanted:
            PHASES[name](probe)
        probe.send(geometry.HOME, 1500, "home")
    except KeyboardInterrupt:
        print("\nstopped; the arm holds where it was last sent")
    finally:
        probe.protocol.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
