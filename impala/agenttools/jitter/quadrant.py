#!/usr/bin/env python3
"""The library's precise move (the nudge, maxarm/arm.py) round the arm's forward quadrant: at each place, arriving as
the app does (over the cubes' height, down to z 42), then 1 mm steps, 10 mm steps out and back, and z steps down to
the desk (2 mm, as ARM TO DESK) and back. Every read and command timed; WATCH_S of reads after each move.

    .venv/bin/python impala/agenttools/jitter/quadrant.py [out.json]   # MOVES THE ARM: the owner's say-so (30 Oct)

Places within +-45 deg; nothing low near the green cube (about -40 deg, 250 mm out, 30 Oct 21:xx).
"""

import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMPALA = HERE.parents[1]
sys.path[:0] = [str(IMPALA), str(IMPALA / "calibration")]

from maxarm import MaxArm, MoveType, Space                                 # noqa: E402
from common.motion import OVER_CUBES_MM, Motion                            # noqa: E402

WATCH_S = 3.0
PLACES = [(0, 160), (0, 220), (30, 150), (30, 210), (-30, 150), (-30, 200), (45, 190), (-45, 160)]


def steps(bearing, reach):
    """(label, desk x, y, z) for one place, after arriving there at z 42."""
    b = math.radians(bearing)
    at = lambda r, z, dx=0.0: (r * math.sin(b) + dx, -r * math.cos(b), z)
    return [("x +1", *at(reach, 42, 1.0)), ("x back", *at(reach, 42)), ("z +1", *at(reach, 43)), ("z back", *at(reach, 42)),
            ("out 10", *at(reach + 10, 42)), ("in 10", *at(reach, 42)), ("z 52", *at(reach, 52)), ("z 42", *at(reach, 42)),
            ("z 22", *at(reach, 22)), ("desk z 2", *at(reach, 2)), ("z 42 up", *at(reach, 42))]


def main():
    out = HERE / (sys.argv[1] if len(sys.argv) > 1 else "quadrant.json")
    arm = MaxArm(config_dir=None)
    motion = Motion(arm, IMPALA / "config")
    protocol = arm._protocol
    reads, commands = [], []
    read, send = protocol.read_pulses, protocol.set_position

    def read_pulses():
        pulses = read()
        reads.append([time.monotonic(), *(pulses or (None, None, None))])
        return pulses

    def set_position(position, duration_ms, nozzle=None):
        commands.append([time.monotonic(), int(duration_ms), list(position)])
        return send(position, duration_ms, nozzle)

    protocol.read_pulses, protocol.set_position = read_pulses, set_position
    moves = []
    try:
        print("connecting: the board resets and homes the arm", flush=True)
        motion.connect()
        motion.use(Space.DESK)
        for bearing, reach in PLACES:
            b = math.radians(bearing)
            plan = [("arrive", reach * math.sin(b), -reach * math.cos(b), 42.0)] + steps(bearing, reach)
            for label, x, y, z in plan:
                first = (len(reads), len(commands))
                started = time.monotonic()
                if label == "arrive":
                    result = motion.approach(Space.DESK, (x, y, z), precise=True, hover=OVER_CUBES_MM)
                else:
                    result = arm.move(x, y, z, MoveType.PRECISE)
                returned = time.monotonic()
                settled = list(arm._settled) if arm._settled else None
                while time.monotonic() - returned < WATCH_S:
                    read_pulses()
                moves.append({"bearing": bearing, "reach": reach, "label": label, "target": [x, y, z],
                              "status": result.status.value, "detail": result.detail, "started": started,
                              "returned": returned, "settled": settled, "reads": reads[first[0]:],
                              "commands": commands[first[1]:]})
                print(f"{bearing:+3d} deg {reach} {label:8s}: {result.status.value} in {returned - started:4.1f} s, "
                      f"{len(commands) - first[1]} command(s)", flush=True)
                out.write_text(json.dumps({"watch_s": WATCH_S, "moves": moves}))
    finally:
        protocol.read_pulses, protocol.set_position = read, send
        try:
            motion.reset()
        finally:
            arm.disconnect()
            out.write_text(json.dumps({"watch_s": WATCH_S, "moves": moves}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
