#!/usr/bin/env python3
"""The desk, found by the cup's seal: at each point in front of the arm, the highest the cup comes down to and seals.

    cd jackal                                           # the calibration app stopped: this opens the arm's port
    ../.venv/bin/python agenttools/true-z/find-desk.py
    ../.venv/bin/python agenttools/true-z/fit-desk.py   # then, no arm: the true z formula from what was found

The points cover a letter sheet, landscape, in front of the arm round its centre line, clear of the base's front
suction cups; one place is found again every few points and last, to see drift. At each point the cup comes down
onto the desk, the last few millimetres PRECISE, the way grab() comes down, and the pump goes on: the library says
whether it sealed (`maxarm/seal.py`). It steps down until the cup seals, then closes in on the highest reading that
sealed and the lowest that did not, a little off the point each time, where the heights the arm can hold fall in
between. A point where it finds no seal, or the arm does not get there, says so and is passed. Arm space, no
calibration; every try is kept as the firmware's raw position too.

Each try is written to desk-seal.jsonl as it is made. Run again and the points already done are skipped; delete
the file to start over. Ctrl-C holds the arm where it is and lets go of the desk; nothing more moves. Ctrl-C again
quits at once.
"""

import json
import math
import signal
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from maxarm import MaxArm, MoveStatus, MoveType, SealState, raw     # noqa: E402
from maxarm.arm import home_position                               # noqa: E402
from maxarm.geometry import Position                               # noqa: E402

OUT = Path(__file__).with_name("desk-seal.jsonl")

# A letter sheet, landscape, in front of the arm round its centre line (the owner's); arm mm, out is -y.
AREA_HALF_X_MM = 140.0
AREA_Y_MM = (-270.0, -55.0)
RINGS_MM = (120.0, 145.0, 170.0, 195.0, 220.0, 245.0, 265.0)
ALONG_RING_MM = 45.0
BEARING_MAX_DEG = 60.0
# The base's front suction cups, which the cup is kept clear of (the owner's).
LEG_CUPS = ((80.0, -80.0), (-80.0, -80.0))
LEG_CUP_CLEAR_MM = 45.0
REFERENCE = (0.0, -190.0)
REFERENCE_EVERY = 10

# Where the desk is first looked for, true z: where the cup sealed straight out (STATUS-jackal §14). After that,
# the edge at the nearest point found within NEIGHBOUR_MM; never more than GUESS_SPAN_MM off the first.
FIRST_GUESS_MM = 44.4
NEIGHBOUR_MM = 80.0
GUESS_SPAN_MM = 3.0
# Down onto the desk: the first try START_MM over the guess (START_ALONE_MM with no point near found yet), then
# STEP_MM lower each time until the cup seals, so that it is never pressed much past where it seals; a seal on the
# first try, UP_MM higher each time until it does not.
START_MM = 0.8
START_ALONE_MM = 1.5
STEP_MM = 0.5
UP_MM = 1.0
# Then the edge is closed in on, half-way each time, to RESOLUTION_MM. At one x, y the firmware holds only whole
# pulses, a few tenths of a millimetre apart in height (STATUS-jackal §15), so each of these tries goes up to
# DITHER_MM off the point, a different place each time, where the heights held fall in between.
RESOLUTION_MM = 0.15
DITHER_MM = 1.0
MAX_TRIES = 10
REPEAT_MM = 0.05
# Each try comes down PRECISE from this far over it. A seal leaves the cup holding a little vacuum after the
# release, so after one the cup lifts TRAVEL_MM over the guess first, where it is known to come free.
DOWN_MM = 4.0
TRAVEL_MM = 10.0
# Never sent lower than this under the guess, or higher over it.
LOWEST_MM = 2.0
HIGHEST_MM = 4.0
# Coming down, the arm stops over where it is sent (STATUS-jackal §14): about this much until the misses say,
# from the last SHORTFALL_FROM of them that lie in SHORTFALL_SPAN_MM.
SHORTFALL_MM = 0.8
SHORTFALL_FROM = 12
SHORTFALL_SPAN_MM = (-0.5, 2.5)
# A reading further than this from where the cup was sent ends the point: the arm did not get there.
OFF_MM = 5.0
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
GOLDEN_FRACTION = (math.sqrt(5.0) - 1.0) / 2.0


class Stopped(Exception):
    """Ctrl-C."""


class Skipped(Exception):
    """The point cannot be done: the library would not make a move, or the arm did not get there."""


@dataclass(frozen=True)
class Try:
    sent: float                  # true z sent
    read: Position               # true: where the servos said the cup came down, before the pump
    state: SealState
    dip: Optional[float]
    pulled: Optional[float]


def layout() -> List[Tuple[str, float, float]]:
    """(label, x, y): ring by ring, out from the base, each the other way round from the last; the reference
    first, every REFERENCE_EVERY points and last."""
    grid = []
    for index, reach in enumerate(RINGS_MM):
        step = math.degrees(ALONG_RING_MM / reach)
        count = int(BEARING_MAX_DEG // step)
        bearings = [step * k for k in range(-count, count + 1)]
        for bearing in reversed(bearings) if index % 2 else bearings:
            x, y = reach * math.sin(math.radians(bearing)), -reach * math.cos(math.radians(bearing))
            if abs(x) > AREA_HALF_X_MM or not AREA_Y_MM[0] <= y <= AREA_Y_MM[1]:
                continue
            if any(math.dist((x, y), cup) < LEG_CUP_CLEAR_MM for cup in LEG_CUPS):
                continue
            grid.append((f"r{reach:.0f} b{bearing:+.0f}", x, y))
    points, references = [], 0
    for index, point in enumerate(grid):
        if index % REFERENCE_EVERY == 0:
            references += 1
            points.append((f"ref {references}", *REFERENCE))
        points.append(point)
    return points + [(f"ref {references + 1}", *REFERENCE)]


def next_try(tries: List[Try], guess: float, start: float,
             shortfall: float) -> Optional[Tuple[float, float, float]]:
    """The next try: how far off the point in x and y, and the height to send, true z; None when the search is
    over."""
    sealed, missed = _split(tries)
    if len(tries) >= MAX_TRIES:
        return None
    if sealed and missed:
        top, bottom = max(sealed), min(missed)
        if bottom - top <= RESOLUTION_MM:
            return None
        angle = len(tries) * GOLDEN_ANGLE
        radius = DITHER_MM * (0.4 + 0.6 * (len(tries) * GOLDEN_FRACTION % 1.0))
        return radius * math.cos(angle), radius * math.sin(angle), (top + bottom) / 2.0 - shortfall
    if sealed:
        sent = max(done.sent for done in tries) + UP_MM
    elif missed:
        sent = min(done.sent for done in tries) - STEP_MM
    else:
        sent = guess + start - shortfall
    sent = min(max(sent, guess - LOWEST_MM), guess + HIGHEST_MM)
    return None if any(abs(sent - done.sent) < REPEAT_MM for done in tries) else (0.0, 0.0, sent)


def _split(tries: List[Try]) -> Tuple[List[float], List[float]]:
    """The heights read where the cup sealed, and where it did not -- an uncertain seal is not one."""
    return ([done.read[2] for done in tries if done.state is SealState.SEALED],
            [done.read[2] for done in tries if done.state in (SealState.LOOSE, SealState.UNCERTAIN)])


class Prober:
    def __init__(self, arm: MaxArm, done: List[dict]) -> None:
        self.arm = arm
        self.found = [entry for entry in done if entry["found"]]
        self.here: Position = home_position()     # where the arm was last sent
        self.shortfalls = _shortfalls()          # read less sent, where the cup came down in the air
        self.is_pump_on = False
        self.is_lift_needed = False
        self.stopping = threading.Event()

    # --- one point ----------------------------------------------------------

    def find(self, label: str, x: float, y: float) -> dict:
        guess, is_near = self.guess(x, y)
        travel = guess + TRAVEL_MM
        self.go(*self.here[:2], max(self.here[2], travel), MoveType.SLOW)
        self.go(x, y, travel, MoveType.FAST)
        tries: List[Try] = []
        while True:
            chosen = next_try(tries, guess, START_MM if is_near else START_ALONE_MM, self.shortfall())
            if chosen is None:
                break
            at_x, at_y, sent = x + chosen[0], y + chosen[1], chosen[2]
            tried = self.try_at(at_x, at_y, sent, travel)
            tries.append(tried)
            _write({"kind": "try", "point": label, "x": at_x, "y": at_y, "sent": sent,
                    "sent_raw": raw.raw_position((at_x, at_y, sent)), "read": tried.read,
                    "read_raw": raw.raw_position(tried.read), "state": tried.state.value, "dip": tried.dip,
                    "pulled": tried.pulled, "time": time.time()})
            if tried.state is SealState.UNKNOWN:
                raise SystemExit("the seal could not be told: this board does not read the pump's supply")
            if math.dist(tried.read, (at_x, at_y, sent)) > OFF_MM:
                raise Skipped("the arm read ({:.1f}, {:.1f}, {:.1f})".format(*tried.read)
                              + f", sent ({at_x:.1f}, {at_y:.1f}, {sent:.1f})")
            if tried.state is SealState.LOOSE and _is_shortfall(tried.read[2] - sent):
                self.shortfalls.append(tried.read[2] - sent)
        summary = _summary(label, x, y, guess, tries)
        _write(summary)
        if summary["found"]:
            self.found.append(summary)
        return summary

    def try_at(self, x: float, y: float, sent: float, travel: float) -> Try:
        """Straight up off the desk -- clear of it after a seal -- over to (x, y), and down onto it."""
        self.go(*self.here[:2], max(self.here[2], travel if self.is_lift_needed else sent + DOWN_MM), MoveType.SLOW)
        self.go(x, y, sent + DOWN_MM, MoveType.SLOW)
        self.go(x, y, sent, MoveType.PRECISE)
        read = self.arm.get_position()
        self.pump(True)
        state = self.arm.get_seal() or SealState.UNKNOWN
        dip, pulled = self.arm.get_seal_measures()
        self.pump(False)
        self.is_lift_needed = state is not SealState.LOOSE
        return Try(sent, read, state, dip, pulled)

    def guess(self, x: float, y: float) -> Tuple[float, bool]:
        """Where to look for the desk, and whether a point near was found to say so."""
        near = [(math.dist((x, y), (entry["x"], entry["y"])), (entry["top"] + entry["bottom"]) / 2.0)
                for entry in self.found]
        near = [pair for pair in near if pair[0] <= NEIGHBOUR_MM]
        guess = min(near)[1] if near else FIRST_GUESS_MM
        return min(max(guess, FIRST_GUESS_MM - GUESS_SPAN_MM), FIRST_GUESS_MM + GUESS_SPAN_MM), bool(near)

    def shortfall(self) -> float:
        recent = self.shortfalls[-SHORTFALL_FROM:]
        return statistics.median(recent) if len(recent) >= 3 else SHORTFALL_MM

    # --- the arm --------------------------------------------------------------

    def go(self, x: float, y: float, z: float, move_type: MoveType) -> None:
        if self.stopping.is_set():
            raise Stopped()
        result = self.arm.move(x, y, z, move_type)
        if result.status is MoveStatus.STOPPED:
            raise Stopped()
        if not result:
            raise Skipped(f"refused: {result.detail}")
        self.here = result.position

    def pump(self, is_on: bool) -> None:
        if self.stopping.is_set() and is_on:
            raise Stopped()
        self.is_pump_on = self.is_pump_on or is_on
        result = self.arm.set_suction(is_on)
        if result.status is MoveStatus.STOPPED:
            raise Stopped()
        self.is_pump_on = is_on

    def stop(self) -> None:
        self.stopping.set()
        self.arm.stop()


def _summary(label: str, x: float, y: float, guess: float, tries: List[Try]) -> dict:
    sealed, missed = _split(tries)
    summary = {"kind": "point", "point": label, "x": x, "y": y, "guess": guess, "tries": len(tries),
               "found": bool(sealed and missed), "time": time.time()}
    if sealed and missed:
        top = max((done for done in tries if done.state is SealState.SEALED), key=lambda done: done.read[2])
        summary.update(top=top.read[2], top_raw=raw.raw_position(top.read), bottom=min(missed),
                       is_overlap=max(sealed) > min(missed))
    elif sealed:
        summary["why"] = f"sealed at every height tried, up to {max(sealed):.1f}"
    elif missed:
        summary["why"] = f"no seal down to {min(missed):.1f}"
    else:
        summary["why"] = "nothing tried"
    return summary


def _write(entry: dict) -> None:
    with OUT.open("a") as out:
        out.write(json.dumps(entry) + "\n")


def _read() -> List[dict]:
    if not OUT.exists():
        return []
    return [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()]


def _shortfalls() -> List[float]:
    return [entry["read"][2] - entry["sent"] for entry in _read() if entry["kind"] == "try"
            and entry["state"] == SealState.LOOSE.value and _is_shortfall(entry["read"][2] - entry["sent"])]


def _is_shortfall(millimetres: float) -> bool:
    return SHORTFALL_SPAN_MM[0] <= millimetres <= SHORTFALL_SPAN_MM[1]


def main() -> None:
    arm = MaxArm(config_dir=None)
    done = [entry for entry in _read() if entry["kind"] == "point"]
    labels = {entry["point"] for entry in done}
    schedule = [point for point in layout() if point[0] not in labels]
    lowest, highest = FIRST_GUESS_MM - GUESS_SPAN_MM - LOWEST_MM, FIRST_GUESS_MM + GUESS_SPAN_MM + TRAVEL_MM
    unreachable = [label for label, x, y in schedule
                   if not all(arm.can_reach(x + dx, y + dy, z) for z in (lowest, highest)
                              for dx, dy in ((0.0, 0.0), (DITHER_MM, 0.0), (-DITHER_MM, 0.0), (0.0, DITHER_MM),
                                             (0.0, -DITHER_MM)))]
    if unreachable:
        raise SystemExit(f"out of reach, nothing moved: {', '.join(unreachable)}")
    if not schedule:
        raise SystemExit(f"every point is done: {OUT}. Delete it to start over.")
    print(f"{len(schedule)} points to do, about {len(schedule) * 1.0:.0f} minutes"
          + (f"; {len(done)} already in {OUT.name}" if done else "") + ". Connecting: the arm homes.")

    prober = Prober(arm, done)

    def on_interrupt(signum, frame) -> None:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n  stopping: the arm holds where it is and lets go of the desk; Ctrl-C again to quit",
              file=sys.stderr)
        prober.stop()

    signal.signal(signal.SIGINT, on_interrupt)
    arm.connect()
    try:
        for number, (label, x, y) in enumerate(schedule, 1):
            try:
                found = prober.find(label, x, y)
            except Skipped as reason:
                _write({"kind": "point", "point": label, "x": x, "y": y, "found": False, "why": str(reason),
                        "time": time.time()})
                print(f"{number:3d}/{len(schedule)} {label:12s} skipped: {reason}", flush=True)
                continue
            said = (f"seals up to {found['top']:.2f} (raw {found['top_raw'][2]:.2f}), not at {found['bottom']:.2f}"
                    + (" -- a seal read over a miss" if found["is_overlap"] else "")
                    if found["found"] else found["why"])
            print(f"{number:3d}/{len(schedule)} {label:12s} {said}; {found['tries']} tries", flush=True)
        prober.go(*prober.here[:2], prober.here[2] + TRAVEL_MM, MoveType.SLOW)
        arm.home()
        print(f"done: {OUT}\nnext: ../.venv/bin/python agenttools/true-z/fit-desk.py")
    except Stopped:
        if prober.is_pump_on:
            arm.set_suction(False)
        print(f"stopped; what was found is in {OUT}. Run again to go on.")
    finally:
        if prober.is_pump_on and not prober.stopping.is_set():
            arm.set_suction(False)
        arm.disconnect()


if __name__ == "__main__":
    main()
