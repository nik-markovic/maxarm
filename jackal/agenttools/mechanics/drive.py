#!/usr/bin/env python3
"""Drive the arm and log how its servos really move: the physical arm only, no camera.

    ../.venv/bin/python agenttools/mechanics/drive.py descent|small|base|all

Run from `jackal/`, the calibration app stopped (it holds the port). The desk must be clear. Every measured leg is
one command sent straight to the board, the servos read every few tens of ms while it runs and for a while after;
travel between places goes through `MaxArm.move()` and its limits, at TRAVEL_MM over the desk. Each leg is a line
in `runs/<phase>-<time>.jsonl`. `touch agenttools/mechanics/STOP` ends the run after the current leg, the arm
lifted to TRAVEL_MM.

- descent: at points across reach and +-60 deg: down to 40 mm over the desk, down to 2, up to 40, down to 2 again.
- small:   one servo at a time, steps of 1-6 whole pulses each way, at 5 and 40 mm, inner, middle and outer reach.
- base:    the base turned onto a place from each side, by 2 to 60 pulses, at 5 and 40 mm.
- sideways: targets between the base's whole pulses, from each side, then down to 2 mm.
- pickup:  pickups on the centre line: start height, extra pulses on servos 2 and 3, the base's side and aim varied.
"""

import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from maxarm import MaxArm, MoveType, Space, geometry, raw        # noqa: E402

HERE = Path(__file__).resolve().parent
STOP = HERE / "STOP"

TRAVEL_MM = 80.0            # over the desk, between places
HIGH_MM = 40.0              # the less critical level
LOW_MM = 2.0                # the critical one: a tile's pickup
STEP_LOW_MM = 5.0           # single-servo steps and base turns near the desk: room for a 6-pulse step down
SLOW_MM_S = 10.0            # the library's slow move: how grab() comes down
FAST_MM_S = 200.0
MIN_MS = 150
WATCH_S = {"travel": 2.0, "descent": 4.0, "ascent": 3.0, "step": 2.0, "turn": 2.5, "approach": 1.5, "pickup": 3.0}

# The base's suction cups (the owner's, find-desk.py): nothing low comes nearer than this.
LEG_CUPS = ((80.0, -80.0), (-80.0, -80.0), (80.0, 80.0), (-80.0, 80.0))
LEG_CUP_CLEAR_MM = 45.0

DESCENT_REACHES = (100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0, 268.0)
DESCENT_BEARINGS = (-60.0, -30.0, 0.0, 30.0, 60.0)
SMALL_PLACES = ((130.0, 0.0), (190.0, 0.0), (250.0, 0.0))
SMALL_STEPS = (1, 1, 1, -1, -1, -1, 2, -2, 3, -3, 4, -4, 6, -6)
BASE_PLACES = ((130.0, 0.0), (190.0, 20.0), (250.0, -20.0))
BASE_TURNS = (2, 4, 8, 20, 60)


class Stop(Exception):
    pass


def at(reach: float, bearing: float) -> Tuple[float, float]:
    b = math.radians(bearing)
    return reach * math.sin(b), -reach * math.cos(b)


def clear_of_cups(x: float, y: float) -> bool:
    return all(math.hypot(x - cx, y - cy) >= LEG_CUP_CLEAR_MM for cx, cy in LEG_CUPS)


class Rig:
    def __init__(self, phase: str) -> None:
        self.arm = MaxArm()
        self.arm.use_coordinates(Space.DESK)
        self.protocol = self.arm._protocol
        self.spaces = self.arm._spaces
        self.out = HERE / "runs" / f"{phase}-{time.strftime('%m%d-%H%M%S')}.jsonl"
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.desk = (0.0, -163.0, 168.0)     # desk space, where it was last sent: from the arm once connected
        self.last_reads: List[list] = []

    # --- logging ---------------------------------------------------------------------------------------------

    def write(self, entry: dict) -> None:
        with self.out.open("a") as file:
            file.write(json.dumps(entry) + "\n")

    # --- moving ----------------------------------------------------------------------------------------------

    def check_stop(self) -> None:
        if STOP.exists():
            raise Stop()

    def travel(self, x: float, y: float, z: float, label: str, height: float = TRAVEL_MM) -> None:
        """Through the library and its limits; up to `height` first, across, down to z."""
        self.check_stop()
        for point in ((self.desk[0], self.desk[1], max(self.desk[2], height)), (x, y, height), (x, y, z)):
            if math.dist(point, self.desk) < 0.01:
                continue
            result = self.arm.move(*point, MoveType.FAST)
            if not result:
                raise RuntimeError(f"travel to {point} refused: {result}")
            self.desk = point
        self.watch_leg("travel", label, None, WATCH_S["travel"], sent_at=time.monotonic(), move_ms=0)

    def leg(self, kind: str, label: str, arm_point: Tuple[float, float, float], speed: float,
            watch_key: str, extra: Optional[dict] = None, floor: float = 0.5) -> dict:
        """One command straight to the board, then the servos read while it runs and WATCH_S after."""
        self.check_stop()
        desk_point = self.spaces.arm.to_desk(arm_point)
        problem = self.arm._planner.problem_with(desk_point)
        if problem:
            raise RuntimeError(f"{label}: {problem}")
        if desk_point[2] < floor or (desk_point[2] < TRAVEL_MM - 1 and not clear_of_cups(*arm_point[:2])):
            raise RuntimeError(f"{label}: too low or too near the base's suction cups: {desk_point}")
        here = self.spaces.arm.from_desk(self.desk)
        move_ms = max(MIN_MS, int(math.dist(here, arm_point) / speed * 1000.0))
        target_raw = raw.raw_position(arm_point)
        sent_at = time.monotonic()
        accepted = self.protocol.set_position(arm_point, move_ms)
        if not accepted:
            raise RuntimeError(f"{label}: the board refused {arm_point}")
        self.desk = desk_point
        self.arm._position = arm_point                 # keep MaxArm's own idea of where it was sent
        entry = {"kind": kind, "label": label, "desk": desk_point, "arm": arm_point, "raw": target_raw,
                 "exact": geometry.deg_to_pulse(geometry.inverse(target_raw)),
                 "commanded": geometry.commanded_pulses(target_raw),
                 "before": self.settled_of(self.last_reads), "move_ms": move_ms, "speed": speed}
        entry.update(extra or {})
        return self.watch_leg(kind, label, entry, WATCH_S[watch_key], sent_at, move_ms)

    def watch_leg(self, kind: str, label: str, entry: Optional[dict], watch_s: float, sent_at: float,
                  move_ms: int) -> dict:
        reads = []
        until = sent_at + move_ms / 1000.0 + watch_s
        while time.monotonic() < until:
            pulses = self.protocol.read_pulses()
            if pulses is not None:
                reads.append([round(time.monotonic() - sent_at, 3), *pulses])
        self.last_reads = reads
        entry = entry or {"kind": kind, "label": label, "desk": self.desk}
        entry["reads"] = reads
        entry["time"] = time.time()
        self.write(entry)
        return entry

    @staticmethod
    def settled_of(reads: List[list]) -> Optional[List[float]]:
        """The mean of the last 0.5 s of reads."""
        if not reads:
            return None
        last = [read for read in reads if read[0] >= reads[-1][0] - 0.5]
        return [statistics.mean(read[i] for read in last) for i in (1, 2, 3)]

    def arm_at_desk(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        return self.spaces.arm.from_desk((x, y, z))

    def arm_of_pulses(self, pulses) -> Tuple[float, float, float]:
        """The arm-space position for whole pulses: sent through the library, the servos are given exactly them."""
        return raw.true_position(geometry.forward(geometry.pulse_to_deg(tuple(float(p) for p in pulses))))

    def finish(self) -> None:
        try:
            if self.desk[2] < TRAVEL_MM:
                self.arm.move(self.desk[0], self.desk[1], TRAVEL_MM, MoveType.SLOW)
        finally:
            self.arm.disconnect()


# --- phases ----------------------------------------------------------------------------------------------------

def descent(rig: Rig) -> None:
    places = [(r, b) for r in DESCENT_REACHES for b in DESCENT_BEARINGS if clear_of_cups(*at(r, b))]
    # Snake across so that travel stays short.
    places.sort(key=lambda rb: (rb[0], rb[1] if DESCENT_REACHES.index(rb[0]) % 2 == 0 else -rb[1]))
    print(f"descent: {len(places)} places", flush=True)
    for number, (r, b) in enumerate(places, 1):
        x, y = at(r, b)
        label = f"r{r:.0f} b{b:+.0f}"
        try:
            rig.travel(x, y, TRAVEL_MM, label)
            rig.leg("descent40", label, rig.arm_at_desk(x, y, HIGH_MM), SLOW_MM_S, "descent")
            rig.leg("descent2", label, rig.arm_at_desk(x, y, LOW_MM), SLOW_MM_S, "descent")
            rig.leg("ascent40", label, rig.arm_at_desk(x, y, HIGH_MM), SLOW_MM_S, "ascent")
            rig.leg("descent2", label, rig.arm_at_desk(x, y, LOW_MM), SLOW_MM_S, "descent", {"again": True})
            rig.leg("ascent40", label, rig.arm_at_desk(x, y, HIGH_MM), SLOW_MM_S, "ascent", {"again": True})
        except RuntimeError as problem:
            print(f"  {label}: skipped: {problem}", flush=True)
            rig.write({"kind": "skipped", "label": label, "why": str(problem), "time": time.time()})
            continue
        print(f"  {number}/{len(places)} {label}", flush=True)


def small(rig: Rig) -> None:
    for r, b in SMALL_PLACES:
        x, y = at(r, b)
        for height in (STEP_LOW_MM, HIGH_MM):
            label = f"r{r:.0f} b{b:+.0f} z{height:.0f}"
            rig.travel(x, y, height, label)
            start = geometry.commanded_pulses(raw.raw_position(rig.arm_at_desk(x, y, height)))
            for servo in (0, 1, 2):
                rig.leg("home", label, rig.arm_of_pulses(start), FAST_MM_S, "step", {"servo": servo})
                pulses = list(start)
                for step in SMALL_STEPS:
                    pulses[servo] += step
                    try:
                        rig.leg("step", label, rig.arm_of_pulses(pulses), FAST_MM_S, "step",
                                {"servo": servo, "step": step})
                    except RuntimeError as problem:
                        print(f"  {label} servo {servo + 1} step {step}: skipped: {problem}", flush=True)
                        pulses[servo] -= step
            rig.leg("home", label, rig.arm_of_pulses(start), FAST_MM_S, "step")
            print(f"  {label} done", flush=True)


def base(rig: Rig) -> None:
    for r, b in BASE_PLACES:
        x, y = at(r, b)
        for height in (STEP_LOW_MM, HIGH_MM):
            label = f"r{r:.0f} b{b:+.0f} z{height:.0f}"
            rig.travel(x, y, height, label)
            start = geometry.commanded_pulses(raw.raw_position(rig.arm_at_desk(x, y, height)))
            for repeat in (1, 2):
                for turn in BASE_TURNS:
                    for side in (-1, 1):
                        off = (start[0] + side * turn, start[1], start[2])
                        try:
                            rig.leg("approach", label, rig.arm_of_pulses(off), FAST_MM_S, "approach",
                                    {"turn": turn, "side": side})
                            rig.leg("turn", label, rig.arm_of_pulses(start), FAST_MM_S, "turn",
                                    {"turn": turn, "side": side, "repeat": repeat})
                        except RuntimeError as problem:
                            print(f"  {label} turn {side * turn}: skipped: {problem}", flush=True)
            print(f"  {label} done", flush=True)


SIDEWAYS_PLACES = ((190.0, 0.0), (250.0, 0.0))
SIDEWAYS_QUARTERS = 9            # targets a quarter of a base pulse apart: two whole pulses swept
SIDEWAYS_FROM_PULSES = 12        # each approached from this far round, from one side and then the other


def sideways(rig: Rig) -> None:
    """Targets between the base's whole pulses: where the base ends against the exact target, approached from each
    side at HIGH_MM, and whether coming down to LOW_MM moves it."""
    for r, b in SIDEWAYS_PLACES:
        x, y = at(r, b)
        label = f"r{r:.0f} b{b:+.0f}"
        rig.travel(x, y, HIGH_MM, label)
        start = geometry.commanded_pulses(raw.raw_position(rig.arm_at_desk(x, y, HIGH_MM)))
        exact_up = geometry.deg_to_pulse(geometry.inverse(raw.raw_position(rig.arm_at_desk(x, y, HIGH_MM))))
        low = geometry.deg_to_pulse(geometry.inverse(raw.raw_position(rig.arm_at_desk(x, y, LOW_MM))))
        for side in (-1, 1):
            for quarter in range(SIDEWAYS_QUARTERS):
                base_pulse = start[0] + quarter * 0.25
                target_high = (base_pulse, exact_up[1], exact_up[2])
                target_low = (base_pulse, low[1], low[2])
                extra = {"side": side, "quarter": quarter, "exact_base": base_pulse}
                try:
                    rig.leg("approach", label, rig.arm_of_pulses((base_pulse + side * SIDEWAYS_FROM_PULSES,
                                                                  exact_up[1], exact_up[2])),
                            FAST_MM_S, "approach", extra)
                    rig.leg("between", label, rig.arm_of_pulses(target_high), FAST_MM_S, "turn", extra)
                    rig.leg("between_down", label, rig.arm_of_pulses(target_low), SLOW_MM_S, "descent", extra)
                    rig.leg("between_up", label, rig.arm_of_pulses(target_high), SLOW_MM_S, "ascent", extra)
                except RuntimeError as problem:
                    print(f"  {label} side {side} quarter {quarter}: skipped: {problem}", flush=True)
            print(f"  {label} side {side:+d} done", flush=True)


PICKUP_REACHES = (130.0, 160.0, 190.0, 220.0, 250.0)
PICKUP_DROPS_MM = (3.0, 6.0, 15.0, 40.0)               # how far over the target the descent starts
PICKUP_EXTRAS = ((0, 0), (-1, 2), (-1, 3), (0, 3))     # added to servo 2's and servo 3's whole-pulse target
# (side, past): approaching from the base's higher pulses (+1, turning -) or lower (-1, turning +), aimed this many
# pulses past the target.
PICKUP_BASE = ((1, 0), (1, 1), (1, 2), (-1, 1), (-1, 2), (-1, 3))
PICKUP_APPROACH_PULSES = 20
PICKUP_FLOOR_MM = -0.5      # a command may go this far under the desk: the cup stops short of it, and is a bellows


def pickup(rig: Rig) -> None:
    """A pickup, again and again: down to a start height, the base turned onto a target between its pulses from a
    chosen side and aimed past it, then straight down slowly to LOW_MM with extra pulses on servos 2 and 3."""
    trial = 0
    for repeat in (1, 2):
        for r in PICKUP_REACHES:
            x, y = at(r, 0.0)
            for drop in PICKUP_DROPS_MM:
                for extra2, extra3 in PICKUP_EXTRAS:
                    side, past = PICKUP_BASE[trial % len(PICKUP_BASE)]
                    quarter = (trial * 3) % 4
                    trial += 1
                    low = geometry.deg_to_pulse(geometry.inverse(raw.raw_position(rig.arm_at_desk(x, y, LOW_MM))))
                    high = geometry.deg_to_pulse(geometry.inverse(raw.raw_position(rig.arm_at_desk(x, y, LOW_MM + drop))))
                    exact_base = round(low[0]) + quarter * 0.25
                    label = f"r{r:.0f} d{drop:.0f} e{extra2:+d}{extra3:+d} s{side:+d}p{past}"
                    info = {"reach": r, "drop": drop, "extra2": extra2, "extra3": extra3, "side": side, "past": past,
                            "exact_base": exact_base, "exact_low": [exact_base, low[1], low[2]], "repeat": repeat}
                    try:
                        start = rig.arm_of_pulses((exact_base + side * PICKUP_APPROACH_PULSES, high[1], high[2]))
                        sx, sy, sz = rig.spaces.arm.to_desk(start)
                        rig.travel(sx, sy, sz, label)
                        rig.leg("pickup_turn", label,
                                rig.arm_of_pulses((exact_base - side * past, high[1], high[2])), FAST_MM_S, "turn", info)
                        rig.leg("pickup_down", label,
                                rig.arm_of_pulses((exact_base - side * past, low[1] + extra2, low[2] + extra3)),
                                SLOW_MM_S, "pickup", info, floor=PICKUP_FLOOR_MM)
                    except RuntimeError as problem:
                        print(f"  {label}: skipped: {problem}", flush=True)
                        rig.write({"kind": "skipped", "label": label, "why": str(problem), "time": time.time()})
                        continue
                    print(f"  {trial}/{2 * len(PICKUP_REACHES) * len(PICKUP_DROPS_MM) * len(PICKUP_EXTRAS)} {label}",
                          flush=True)


# The best guess from the pickup phase (6 Nov): extra pulses on servo 3 and servo 2 by reach, straight between rows.
TUNE_GUESS = ((120.0, 0.0, 3.6), (130.0, 0.0, 3.6), (150.0, -0.5, 3.5), (160.0, -0.5, 3.4), (190.0, -0.5, 2.25),
              (220.0, -0.5, 1.75), (250.0, -0.5, 1.5), (265.0, -0.5, 1.4))
TUNE_REACHES = (120.0, 140.0, 160.0, 175.0, 190.0, 205.0, 220.0, 235.0, 250.0, 265.0)
# (servo 2 offset from the guess, servo 3 offset from the guess, repeats)
TUNE_VARIANTS = ((0.0, 0.0, 3), (0.0, -0.5, 3), (0.0, 0.5, 3), (-0.5, 0.0, 2), (0.5, 0.0, 2))
TUNE_DROP_MM = 6.0
TUNE_TRAVEL_MM = 30.0
TUNE_BASE = ((1, 1), (-1, 2))      # the base aims that landed within half a pulse: turning - 1 past, turning + 2 past


def guessed_extras(reach: float) -> Tuple[float, float]:
    rows = TUNE_GUESS
    if reach <= rows[0][0]:
        return rows[0][1], rows[0][2]
    for (near, n2, n3), (far, f2, f3) in zip(rows, rows[1:]):
        if reach <= far:
            share = (reach - near) / (far - near)
            return n2 + (f2 - n2) * share, n3 + (f3 - n3) * share
    return rows[-1][1], rows[-1][2]


def tune(rig: Rig) -> None:
    """Pickups around the best guess: by reach, the guessed extras on servos 2 and 3 and half a pulse either side,
    each repeated; the base always on one of the two aims that worked."""
    trials = [(r, d2, d3) for repeat in range(3) for r in TUNE_REACHES for d2, d3, times in TUNE_VARIANTS
              if repeat < times]
    for number, (r, d2, d3) in enumerate(trials, 1):
        x, y = at(r, 0.0)
        guess2, guess3 = guessed_extras(r)
        extra2, extra3 = guess2 + d2, guess3 + d3
        side, past = TUNE_BASE[number % 2]
        quarter = (number * 3) % 4
        low = geometry.deg_to_pulse(geometry.inverse(raw.raw_position(rig.arm_at_desk(x, y, LOW_MM))))
        high = geometry.deg_to_pulse(geometry.inverse(raw.raw_position(rig.arm_at_desk(x, y, LOW_MM + TUNE_DROP_MM))))
        exact_base = round(low[0]) + quarter * 0.25
        label = f"r{r:.0f} e{extra2:+.2f}{extra3:+.2f} s{side:+d}p{past}"
        info = {"reach": r, "drop": TUNE_DROP_MM, "extra2": extra2, "extra3": extra3, "d2": d2, "d3": d3,
                "side": side, "past": past, "exact_base": exact_base, "exact_low": [exact_base, low[1], low[2]]}
        try:
            start = rig.arm_of_pulses((exact_base + side * PICKUP_APPROACH_PULSES, high[1], high[2]))
            sx, sy, sz = rig.spaces.arm.to_desk(start)
            rig.travel(sx, sy, sz, label, height=TUNE_TRAVEL_MM)
            rig.leg("pickup_turn", label, rig.arm_of_pulses((exact_base - side * past, high[1], high[2])),
                    FAST_MM_S, "turn", info)
            rig.leg("pickup_down", label, rig.arm_of_pulses((exact_base - side * past, low[1] + extra2,
                                                             low[2] + extra3)),
                    SLOW_MM_S, "pickup", info, floor=PICKUP_FLOOR_MM)
        except RuntimeError as problem:
            print(f"  {label}: skipped: {problem}", flush=True)
            rig.write({"kind": "skipped", "label": label, "why": str(problem), "time": time.time()})
            continue
        print(f"  {number}/{len(trials)} {label}", flush=True)


PHASES = {"descent": descent, "small": small, "base": base, "sideways": sideways, "pickup": pickup, "tune": tune}


def main() -> None:
    names = sys.argv[1:] or ["all"]
    if names == ["all"]:
        names = list(PHASES)
    if any(name not in PHASES for name in names):
        raise SystemExit(f"usage: drive.py {'|'.join(PHASES)}|all")
    STOP.unlink(missing_ok=True)
    for name in names:
        rig = Rig(name)
        print(f"{name}: connecting, the arm homes; log {rig.out}", flush=True)
        rig.arm.connect()
        rig.desk = rig.arm._desk_position()
        try:
            PHASES[name](rig)
            print(f"{name}: done", flush=True)
        except Stop:
            print(f"{name}: stopped (STOP file)", flush=True)
            break
        finally:
            rig.finish()
    print("finished", flush=True)


if __name__ == "__main__":
    main()
