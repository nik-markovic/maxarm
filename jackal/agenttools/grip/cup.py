#!/usr/bin/env python3
"""The supply at pin 39 traced through every change: pump on, pump off (the vent valve), a move, a piece stuck on.
The pump is switched straight on the board, no library waits: the trace starts at the switch.

    cd jackal
    ../.venv/bin/python agenttools/grip/cup.py

Phases: empty (each pose: arrive, trace pump off; pump on, trace; pump off, trace the vent); tile (stick it on once,
the pump stays on; each pose: arrive, trace); cube (the same); empty again. Poses forward then back. Each trace is
printed as 0.5 s means from its start, so where it settles shows. Cup at Z_MM everywhere (a 40 mm cube hangs
clear). Ctrl-C: pump off, arm home. Logged to runs/cup-<time>.jsonl.
"""

import json
import sys
import time
from pathlib import Path
from statistics import fmean

HERE = Path(__file__).resolve().parent
JACKAL = HERE.parents[1]
sys.path.insert(0, str(JACKAL / "src"))

from maxarm import MaxArm, MoveType, Space      # noqa: E402

CONFIG = JACKAL / "config"
Z_MM = 70.0
POSES = {                                       # desk mm; the tiles in the lift run were at (24..49, -209)
    "near": (0.0, -160.0),
    "mid": (0.0, -210.0),
    "far": (0.0, -265.0),
}
LOAD_AT = (0.0, -180.0)                         # where the piece goes on and comes off
ARRIVED_S = 4.0                                 # traced after a move
PUMP_ON_S = 5.0                                 # traced after the pump goes on
PUMP_OFF_S = 5.0                                # traced after it goes off: the vent valve's second and after
BIN_S = 0.5


def trace(arm: MaxArm, seconds: float) -> list:
    """[seconds from now, supply] for `seconds`, as fast as the board answers."""
    reads, started = [], time.monotonic()
    while time.monotonic() - started < seconds:
        _, supply = arm._protocol.read_pulses_and_supply()
        if supply is not None:
            reads.append([round(time.monotonic() - started, 3), round(supply, 2)])
    return reads


def bins(reads: list) -> list:
    out = []
    for start in range(int(max(read[0] for read in reads) / BIN_S) + 1):
        values = [read[1] for read in reads if start * BIN_S <= read[0] < (start + 1) * BIN_S]
        out.append(round(fmean(values), 1) if values else None)
    return out


def main() -> None:
    arm = MaxArm(config_dir=CONFIG)
    log_path = HERE / "runs" / time.strftime("cup-%m%d-%H%M%S.jsonl")
    log_path.parent.mkdir(exist_ok=True)
    print(f"cup: connecting; log {log_path}", flush=True)
    arm.connect()
    arm.use_coordinates(Space.DESK)
    poses = [name for name in POSES if arm.can_reach(*POSES[name], Z_MM)]
    order = poses + list(reversed(poses))
    pump = arm._protocol.set_suction
    try:
        with log_path.open("a") as log:

            def record(phase: str, event: str, name: str, pass_: int, seconds: float) -> None:
                reads = trace(arm, seconds)
                entry = {"phase": phase, "event": event, "pose": name, "pass": pass_, "at": [*{**POSES, "load": LOAD_AT}[name], Z_MM],
                         "time": time.time(), "bins": bins(reads), "reads": reads}
                log.write(json.dumps(entry) + "\n")
                log.flush()
                print(f"    {event:9s} " + " ".join(f"{value:.0f}" if value else "—" for value in entry["bins"]),
                      flush=True)

            for phase in ("empty", "tile", "cube", "empty again"):
                carrying = phase in ("tile", "cube")
                print(f"  {phase}   (0.5 s means from each event)", flush=True)
                if carrying:
                    arm.move(*LOAD_AT, Z_MM, MoveType.SLOW)
                    pump(True)
                    if input(f"    stick the {phase} to the cup, let go, Enter (s = skip): ").strip().lower() == "s":
                        pump(False)
                        time.sleep(PUMP_OFF_S)
                        continue
                for step, name in enumerate(order):
                    pass_ = 1 + step // len(poses)
                    print(f"   {name} (pass {pass_})", flush=True)
                    arm.move(*POSES[name], Z_MM, MoveType.SLOW if carrying else MoveType.FAST)
                    record(phase, "arrived", name, pass_, ARRIVED_S)
                    if not carrying:
                        pump(True)
                        record(phase, "pump on", name, pass_, PUMP_ON_S)
                        pump(False)
                        record(phase, "pump off", name, pass_, PUMP_OFF_S)
                if carrying:
                    arm.move(*LOAD_AT, Z_MM, MoveType.SLOW)
                    input(f"    hold the {phase}, Enter, and the pump lets go: ")
                    pump(False)
                    record(phase, "let go", "load", 0, PUMP_OFF_S)
    except KeyboardInterrupt:
        print("\ncup: stopped", flush=True)
    finally:
        try:
            pump(False)
            time.sleep(1.5)
            arm.home()
        finally:
            arm.disconnect()
    print("cup: done", flush=True)


if __name__ == "__main__":
    main()
