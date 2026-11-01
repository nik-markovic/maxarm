#!/usr/bin/env python3
"""The servos' reads round precise moves along the arm's +45 deg line: 1 mm steps out and back, 10 mm steps out and
back, and z in 10 mm steps, as the library does them today (maxarm/arm.py: still a while, the last read). Every read
and command is timed; after each move the reads go on for WATCH_S more.

    .venv/bin/python impala/agenttools/jitter/walk.py      # MOVES THE ARM: the owner's say-so only (30 Oct, 20:xx)

Writes walk.json beside this file after every move; estimators-walk.py reads it.
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

WATCH_S = 4.0
BEARING = math.radians(45.0)
PLAN = ([(180, 42)] + [(r, 42) for r in (181, 182, 183, 184, 183, 182, 181, 180)]
        + [(r, 42) for r in (190, 200, 210, 220, 210, 200)]
        + [(200, z) for z in (52, 62, 52, 42, 32, 22, 32, 42)] + [(r, 42) for r in (190, 180)])


def main():
    arm = MaxArm(config_dir=None)
    motion = Motion(arm, IMPALA / "config")
    protocol = arm._protocol
    reads, commands, settles = [], [], []
    read, send, settle = protocol.read_pulses, protocol.set_position, arm._settle

    def read_pulses():
        pulses = read()
        reads.append([time.monotonic(), *(pulses or (None, None, None))])
        return pulses

    def set_position(position, duration_ms, nozzle=None):
        commands.append([time.monotonic(), int(duration_ms), list(position)])
        return send(position, duration_ms, nozzle)

    def settle_logged(reads_so_far, still_s):
        result = settle(reads_so_far, still_s)
        settles.append([time.monotonic(), still_s, time.monotonic() - arm._travel_ended,
                        list(result) if result else None])
        return result

    protocol.read_pulses, protocol.set_position, arm._settle = read_pulses, set_position, settle_logged
    moves = []
    try:
        print("connecting: the board resets and homes the arm", flush=True)
        motion.connect()
        motion.use(Space.DESK)
        for index, (reach, z) in enumerate(PLAN):
            target = (reach * math.sin(BEARING), -reach * math.cos(BEARING), float(z))
            first = (len(reads), len(commands), len(settles))
            started = time.monotonic()
            if index == 0:
                result = motion.approach(Space.DESK, target, precise=True, hover=OVER_CUBES_MM)
            else:
                result = arm.move(*target, MoveType.PRECISE)
            returned = time.monotonic()
            settled = list(arm._settled) if arm._settled else None
            while time.monotonic() - returned < WATCH_S:
                read_pulses()
            moves.append({"reach": reach, "z": z, "target": target, "status": result.status.value,
                          "detail": result.detail, "started": started, "returned": returned, "settled": settled,
                          "reads": reads[first[0]:], "commands": commands[first[1]:], "settles": settles[first[2]:]})
            last = settles[-1] if settles[first[2]:] else None
            print(f"{index:2d} reach {reach} z {z}: {result.status.value} in {returned - started:.1f} s, "
                  f"{len(commands) - first[1]} command(s); settled {settled}"
                  + (f" {last[2]:.2f} s after the travel" if last else ""), flush=True)
            (HERE / "walk.json").write_text(json.dumps({"watch_s": WATCH_S, "moves": moves}))
    finally:
        protocol.read_pulses, protocol.set_position, arm._settle = read, send, settle
        try:
            motion.reset()
        finally:
            arm.disconnect()
            (HERE / "walk.json").write_text(json.dumps({"watch_s": WATCH_S, "moves": moves}))
    print(f"wrote {HERE / 'walk.json'}")


if __name__ == "__main__":
    main()
