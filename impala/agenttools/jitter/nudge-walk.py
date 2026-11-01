#!/usr/bin/env python3
"""The owner's nudge, tried on the arm: the same +45 deg walk as walk.py, each target reached by a plain slow move and
then nudged, servo by servo. No library change.

    .venv/bin/python impala/agenttools/jitter/nudge-walk.py      # MOVES THE ARM: the owner's say-so only (30 Oct)

A round: wait until the servos are steady (from 1 s after the travel, the mean of the last 0.5 s and of the 0.5 s
before it within 0.35 pulse, every servo; at most 3 s), their mean against the target's exact pulses. Every servo
within TOL_PULSES: done. Otherwise the servos off it are pushed past their target by PUSH_PAST_MM at the cup, the
others held at theirs, quickly (PUSH_MS); PUSH_WAIT_S on, the target itself (BACK_MS). At most ROUNDS. After the move,
WATCH_S of reads. Writes nudge-walk.json beside this file after every move.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IMPALA = HERE.parents[1]
sys.path[:0] = [str(IMPALA), str(IMPALA / "calibration")]

from maxarm import MaxArm, MoveType, Space, geometry, raw                  # noqa: E402
from common.motion import OVER_CUBES_MM, Motion                            # noqa: E402

TOL_PULSES = 0.75
PUSH_PAST_MM = 1.0
# servo 2 turning + stops 3-6 pulses short (walk, 30 Oct) and turning - lands within about a pulse: when it is short on
# its + side, push it this far past, so that it comes back to the target turning -
SERVO2_PLUS_PAST_MM = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
OUT = sys.argv[2] if len(sys.argv) > 2 else "nudge-walk.json"
BASE_PAST_MM = float(sys.argv[3]) if len(sys.argv) > 3 else PUSH_PAST_MM      # the base, either way
PUSH_MS = 150
PUSH_WAIT_S = 0.5
BACK_MS = 150
ROUNDS = 3
WATCH_S = 4.0
BEARING = math.radians(45.0)
PLAN = ([(180, 42)] + [(r, 42) for r in (181, 182, 183, 184, 183, 182, 181, 180)]
        + [(r, 42) for r in (190, 200, 210, 220, 210, 200)]
        + [(200, z) for z in (52, 62, 52, 42, 32, 22, 32, 42)] + [(r, 42) for r in (190, 180)])


def pulses_of(true):
    return np.array(geometry.deg_to_pulse(geometry.inverse(raw.raw_position(tuple(true)))))


def true_of(pulses):
    return np.array(raw.true_position(geometry.forward(geometry.pulse_to_deg(tuple(pulses)))))


def mm_per_pulse(pulses):
    here = true_of(pulses)
    return np.array([np.linalg.norm(true_of(pulses + np.eye(3)[s]) - here) for s in range(3)])


def main():
    arm = MaxArm(config_dir=None)
    motion = Motion(arm, IMPALA / "config")
    protocol = arm._protocol
    log = []

    def read():
        pulses = protocol.read_pulses()
        log.append([time.monotonic(), *(pulses or (None, None, None))])
        return pulses

    def steady(travel_end):
        reads = []
        while True:
            pulses = read()
            now = time.monotonic()
            if pulses:
                reads.append((now, pulses))
            if now - travel_end >= 1.0:
                recent = np.array([p for w, p in reads if w > now - 0.5], float)
                before = np.array([p for w, p in reads if now - 1.0 < w <= now - 0.5], float)
                if len(before) >= 3 and len(recent) >= 3 and np.all(np.abs(recent.mean(0) - before.mean(0)) <= 0.35):
                    return np.vstack((before, recent)).mean(axis=0), now - travel_end
                if now - travel_end >= 3.0:
                    return np.vstack((before, recent)).mean(axis=0), now - travel_end

    def send(true, ms):
        at = time.monotonic()
        protocol.set_position(tuple(true), ms)
        return at

    moves = []
    try:
        print("connecting: the board resets and homes the arm", flush=True)
        motion.connect()
        motion.use(Space.DESK)
        for index, (reach, z) in enumerate(PLAN):
            desk = (reach * math.sin(BEARING), -reach * math.cos(BEARING), float(z))
            target = np.array(arm._spaces.arm.from_desk(arm._space.to_desk(desk)))
            want = pulses_of(target)
            size = mm_per_pulse(want)
            first_log = len(log)
            started = time.monotonic()
            if index == 0:
                result = motion.approach(Space.DESK, desk, precise=False, hover=OVER_CUBES_MM)
            else:
                result = arm.move(*desk, MoveType.SLOW)
            travel_end = time.monotonic()
            here, waited = steady(travel_end)
            rounds = [{"here": here.tolist(), "waited": waited}]
            for _ in range(ROUNDS):
                miss = want - here
                off = np.abs(miss) > TOL_PULSES
                if not off.any():
                    break
                past = np.full(3, PUSH_PAST_MM)
                past[0] = BASE_PAST_MM
                if miss[1] > 0:
                    past[1] = SERVO2_PLUS_PAST_MM
                push = want + np.where(off, np.sign(miss) * past / size, 0.0)
                pushed_at = send(true_of(push), PUSH_MS)
                while time.monotonic() - pushed_at < PUSH_WAIT_S:
                    read()
                back_at = send(target, BACK_MS)
                while time.monotonic() - back_at < BACK_MS / 1000:
                    read()
                here, waited = steady(time.monotonic())
                rounds.append({"push": push.tolist(), "pushed_at": pushed_at, "back_at": back_at,
                               "here": here.tolist(), "waited": waited})
            done = time.monotonic()
            while time.monotonic() - done < WATCH_S:
                read()
            arm._position = tuple(target)                 # the library's idea of where the cup is, for its next move
            moves.append({"reach": reach, "z": z, "desk": desk, "target": target.tolist(), "want": want.tolist(),
                          "mm_per_pulse": size.tolist(), "status": result.status.value, "started": started,
                          "travel_end": travel_end, "done": done, "rounds": rounds, "reads": log[first_log:]})
            final = np.array(rounds[-1]["here"])
            print(f"{index:2d} reach {reach} z {z}: landed {np.round(np.array(rounds[0]['here']) - want, 2).tolist()} "
                  f"pulses off; {len(rounds) - 1} nudge(s); ended {np.round(final - want, 2).tolist()} "
                  f"({np.linalg.norm(true_of(final) - target):.2f} mm) in {done - started:.1f} s", flush=True)
            (HERE / OUT).write_text(json.dumps({"moves": moves}))
    finally:
        try:
            motion.reset()
        finally:
            arm.disconnect()
            (HERE / OUT).write_text(json.dumps({"moves": moves}))
    print(f"wrote {HERE / OUT}")


if __name__ == "__main__":
    main()
