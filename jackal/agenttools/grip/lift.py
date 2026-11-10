#!/usr/bin/env python3
"""Does the cup hold the tile after the lift? The supply voltage at the lift pose, pump off before the grab and pump
on after it; the truth from the camera: the tile set down aside, then found again. Arm and camera; no web app.

    cd jackal
    ../.venv/bin/python agenttools/grip/lift.py [cycles]

Each cycle: park, find the tile; over it at the lift height, pump off, read; grab at the next of GRIPS (+3.0 is 1 mm
over the tile's top: the cup lifts nothing); back at the lift height, pump on, read; set down DROP_ASIDE_MM aside;
park, find it: where it is says whether it came along. Logged to runs/lift-<time>.jsonl. `touch
agenttools/grip/STOP` ends it after the current step, the pump off and the arm home.
"""

import json
import math
import sys
import time
from pathlib import Path
from statistics import fmean

HERE = Path(__file__).resolve().parent
JACKAL = HERE.parents[1]
sys.path.insert(0, str(JACKAL / "src"))

from maxarm import Camera, MaxArm, MoveType, Space          # noqa: E402
from applib.camera.feed import Feed                          # noqa: E402
from applib.camera.store import load_camera                  # noqa: E402
from applib.tiles import find_tiles                          # noqa: E402

CONFIG = JACKAL / "config"
STOP = HERE / "STOP"
GRIPS = (2.0, 3.0, 1.5, 2.5, 1.0, 3.0)    # desk heights the cup is sent to; the tile's top is 2.0
CLEARANCE_MM = 5.0                         # the tiles app's (`word.CLEARANCE_MM`): lift to clearance + grip + 5
DROP_MM = 4.0                              # the tiles app's DROP_MM
DROP_ASIDE_MM = 25.0
SETTLE_S = 1.0                             # after a move, before reading
READ_S = 3.0
HELD_WITHIN_MM = 10.0                      # found this near the drop spot: it came along
STAYED_WITHIN_MM = 6.0                     # this near where it was picked up: it did not


class Stop(Exception):
    pass


def check_stop() -> None:
    if STOP.exists():
        raise Stop()


def read_for(arm: MaxArm, seconds: float) -> list:
    """[seconds from start, base, s2, s3, supply] for `seconds`, then three servo input voltages (mV)."""
    reads, started = [], time.monotonic()
    while time.monotonic() - started < seconds:
        pulses, supply = arm._protocol.read_pulses_and_supply()
        if pulses is not None:
            reads.append([round(time.monotonic() - started, 3), *pulses, supply])
    vin = [arm._protocol.evaluate("bus_servo.get_vin(1)") for _ in range(3)]
    return reads, vin


def summary(reads: list) -> dict:
    supplies = [read[4] for read in reads if read[4] is not None]
    late = [read[4] for read in reads if read[4] is not None and read[0] >= reads[-1][0] - 1.0]
    return {"mean": round(fmean(supplies), 2) if supplies else None,
            "last_1s": round(fmean(late), 2) if late else None,
            "min": min(supplies) if supplies else None, "max": max(supplies) if supplies else None}


def locate(arm: MaxArm, feed: Feed, camera: Camera, near=None):
    """Park, snapshot, the tiles found; the one nearest `near`, or the strongest."""
    check_stop()
    arm.home()
    frame, _ = feed.snapshot()
    tiles = find_tiles(frame, camera)
    if not tiles:
        return None, []
    if near is None:
        return max(tiles, key=lambda tile: tile.contrast), tiles
    return min(tiles, key=lambda tile: math.dist((tile.x, tile.y), near)), tiles


def main() -> None:
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    STOP.unlink(missing_ok=True)
    fit = load_camera(CONFIG)
    camera = Camera(fit.desk_to_pixel, fit.frame_size)
    feed = Feed(None, CONFIG)
    feed.start()
    arm = MaxArm(config_dir=CONFIG)
    log_path = HERE / "runs" / time.strftime("lift-%m%d-%H%M%S.jsonl")
    log_path.parent.mkdir(exist_ok=True)
    landing_seal = {}
    original = arm._switch_suction

    def switching(is_on):                  # the library's verdict at the landing, before the lift ends it
        stopped = original(is_on)
        if is_on:
            landing_seal.update(seal=arm._seal and [arm._seal[0].value, arm._seal[1], arm._seal[2]],
                                supply_off=arm._supply_off, rest_z=arm._rest_z)
        return stopped

    arm._switch_suction = switching
    print(f"lift: connecting; log {log_path}", flush=True)
    arm.connect()
    arm.use_coordinates(Space.DESK)
    arm.set_clearance(CLEARANCE_MM)
    side = 1.0
    try:
        tile, _ = locate(arm, feed, camera)
        if tile is None:
            raise SystemExit("no tile found on the desk")
        with log_path.open("a") as log:
            for cycle in range(cycles):
                grip = GRIPS[cycle % len(GRIPS)]
                x, y = tile.x, tile.y
                travel = CLEARANCE_MM + grip + 5.0
                drop = (x + side * DROP_ASIDE_MM, y)
                if not arm.can_reach(*drop, DROP_MM):
                    side = -side
                    drop = (x + side * DROP_ASIDE_MM, y)
                entry = {"cycle": cycle + 1, "grip": grip, "tile": [round(x, 2), round(y, 2)], "travel": travel,
                         "drop": [round(v, 2) for v in drop]}
                check_stop()
                arm.move(x, y, travel, MoveType.FAST)
                time.sleep(SETTLE_S)
                entry["off_reads"], entry["off_vin"] = read_for(arm, READ_S)
                landing_seal.clear()
                check_stop()
                grabbed = arm.grab(x, y, grip)
                entry["grab"] = str(grabbed)
                entry["landing"] = dict(landing_seal)
                time.sleep(SETTLE_S)
                entry["on_reads"], entry["on_vin"] = read_for(arm, READ_S)
                check_stop()
                entry["dropped"] = str(arm.drop(*drop, DROP_MM))
                found, all_tiles = locate(arm, feed, camera, near=drop)
                if found is None:
                    entry["truth"] = "lost"
                else:
                    to_drop, to_pick = math.dist((found.x, found.y), drop), math.dist((found.x, found.y), (x, y))
                    entry["found"] = [round(found.x, 2), round(found.y, 2)]
                    entry["truth"] = ("held" if to_drop <= HELD_WITHIN_MM else "stayed" if to_pick <= STAYED_WITHIN_MM
                                      else "moved")
                    if entry["truth"] == "held":
                        side = -side
                    tile = found
                entry["tiles_seen"] = len(all_tiles)
                off, on = summary(entry["off_reads"]), summary(entry["on_reads"])
                entry["off"], entry["on"] = off, on
                log.write(json.dumps(entry) + "\n")
                log.flush()
                lifted_dip = (off["last_1s"] - on["last_1s"]) if off["last_1s"] and on["last_1s"] else None
                seal = landing_seal.get("seal")
                print(f"  {cycle + 1}/{cycles} grip +{grip:.1f}: {entry['truth']:6s}  lifted dip "
                      f"{lifted_dip if lifted_dip is None else round(lifted_dip, 1)}  (off {off['last_1s']}, "
                      f"on {on['last_1s']})  landing {seal}", flush=True)
                if found is None:
                    break
    except Stop:
        print("lift: stopped", flush=True)
    finally:
        try:
            arm.set_suction(False)
            arm.home()
        finally:
            arm.disconnect()
            feed.close()
    print("lift: done", flush=True)


if __name__ == "__main__":
    main()
