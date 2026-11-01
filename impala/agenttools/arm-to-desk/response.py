#!/usr/bin/env python3
"""How the servos' reports run after each command of a precise move, today's precise move unchanged.

    .venv/bin/python impala/agenttools/arm-to-desk/response.py [watch-seconds] [out.json]

MOVES THE ARM: run only with the owner's say-so, the arm free (the app stopped). Fine Tune's 20 points at z 42, each
as a grid click sends the arm there (`Motion.approach`, precise, over the cubes at 60), then the app's read. Every
settled read -- the precise rounds' and the app's -- reads on for `watch` seconds (4), each report timed, and returns
what today's read would: the median of the first 5 reads from 400 ms on. So the arm is moved and corrected as
always, only later. Every command is logged with its time and how long the servos were given.

Writes response.json beside this file (or `out.json`); response-check.py reads it.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IMPALA = HERE.parents[1]
sys.path[:0] = [str(IMPALA), str(IMPALA / "calibration")]

from maxarm import MaxArm, Space                                           # noqa: E402
from maxarm import geometry, raw                                           # noqa: E402
from maxarm.protocol import SETTLE_MS, SETTLE_READS                        # noqa: E402
from common.motion import OVER_CUBES_MM, Motion                            # noqa: E402

CONFIG = IMPALA / "config"
NOTCH_MM = 42.0


def main():
    watch = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    out = HERE / (sys.argv[2] if len(sys.argv) > 2 else "response.json")
    points = json.loads((CONFIG / "camera-refine-points.json").read_text())["points"]
    targets = [(round(point["x"]), round(point["y"])) for point in points]
    arm = MaxArm(config_dir=None)
    motion = Motion(arm, CONFIG)
    protocol = arm._protocol
    commands, reads = [], []
    send = protocol.set_position

    def set_position(position, duration_ms, nozzle=None):
        commands.append({"at": time.monotonic(), "ms": int(duration_ms), "true": list(position),
                         "pulses": [int(p) for p in geometry.deg_to_pulse(geometry.inverse(raw.raw_position(position)))]})
        return send(position, duration_ms, nozzle)

    def read_settled_pulses():
        called = time.monotonic()
        got = []
        while time.monotonic() - called < watch:
            pulses = protocol.read_pulses()
            got.append([time.monotonic(), *(pulses or (None, None, None))])
        reads.append({"called": called, "reads": got})
        today = [row[1:] for row in got if row[0] - called >= SETTLE_MS / 1000 and row[1] is not None][:SETTLE_READS]
        return tuple(float(np.median([row[servo] for row in today])) for servo in range(3)) if today else None

    protocol.set_position = set_position
    protocol.read_settled_pulses = read_settled_pulses
    runs = []
    try:
        print("connecting: the board resets and homes the arm", flush=True)
        motion.connect()
        for x, y in targets:
            target = (float(x), float(y), NOTCH_MM)
            first_command, first_read = len(commands), len(reads)
            started = time.monotonic()
            result = motion.approach(Space.DESK, target, precise=True, hover=OVER_CUBES_MM)
            done = time.monotonic()
            motion.read()                                                  # the app's read after the move
            wanted = geometry.deg_to_pulse(geometry.inverse(raw.raw_position(
                arm._spaces.arm.from_desk(arm._space.to_desk(target)))))
            runs.append({"target": target, "status": result.status.value, "started": started, "done": done,
                         "wanted": [int(p) for p in wanted], "commands": commands[first_command:],
                         "reads": reads[first_read:]})
            print(f"{target}: {result.status.value}, {len(reads) - first_read} settled reads", flush=True)
            out.write_text(json.dumps({"watch": watch, "runs": runs}))
    finally:
        protocol.read_settled_pulses = lambda: None
        try:
            motion.reset()
        finally:
            arm.disconnect()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
