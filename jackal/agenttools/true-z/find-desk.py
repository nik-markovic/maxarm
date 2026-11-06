#!/usr/bin/env python3
"""The desk, found by the cup's seal: at each point in front of the arm, the highest the cup comes down to and seals.

    cd jackal                                           # the calibration app stopped: this opens the arm's port
    ../.venv/bin/python agenttools/true-z/find-desk.py [letter | affirm]
    ../.venv/bin/python agenttools/true-z/fit-desk.py   # then, no arm: the true z formula from what was found

`letter` (the default) covers a letter sheet, landscape, in front of the arm round its centre line; `affirm` finds
again each ring's points near the centre line and those where no seal was found, and goes all round the ring at
ALL_ROUND_REACH_MM, sideways and a little behind the base, for the arm's mounting tilt. Each keeps clear of the
base's suction cups; one place is found again every few points and last, to see drift. At each point the cup comes down
onto the desk, the last few millimetres PRECISE, the way grab() comes down, and the pump goes on: a seal is the
library's verdict, kept only if the seal pulled the arm down (`seal_edge.judged()`). It steps down until the cup
seals; then, between the highest reading that sealed and the lowest that did not, it aims at the heights the arm
can be held at -- whole pulses of its servos -- each command allowing for how many pulses each servo stops short,
until the arm holds none between the two. A point where it finds no seal, or the arm does not get there, says so
and is passed. Arm space, no calibration; every try is kept as the firmware's raw position too.

Each try is written to desk-seal.jsonl (`affirm`: desk-seal-affirm.jsonl) as it is made. Run again and the points
already done are skipped, the reference found again first; delete the file to start over. Ctrl-C holds the arm
where it is and lets go of the desk; nothing more moves. Ctrl-C again quits at once.
"""

import json
import math
import signal
import statistics
import sys
import threading
import time
from pathlib import Path
from collections import Counter
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from maxarm import MaxArm, MoveStatus, MoveType, SealState, geometry, raw     # noqa: E402
from maxarm.arm import home_position                                         # noqa: E402
from maxarm.geometry import Position                                         # noqa: E402

from seal_edge import SAME_MM, Try, held_heights, judged, pulses_of, rebuild, split, summary   # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "desk-seal.jsonl"

# A letter sheet, landscape, in front of the arm round its centre line (the owner's); arm mm, out is -y.
AREA_HALF_X_MM = 140.0
AREA_Y_MM = (-270.0, -55.0)
RINGS_MM = tuple(float(reach) for reach in range(115, 266, 15))
ALONG_RING_MM = 60.0
BEARING_MAX_DEG = 60.0
# `affirm`: the letter's points this near the centre line, and those where no seal was found; and one ring all
# round, inside the desk's bounds (the owner's: off the desk past x 200 or y 60).
AFFIRM_BEARING_DEG = 60.0
AFFIRM_AGAIN = ("r220 b+23", "r235 b+29")
ALL_ROUND_REACH_MM = 175.0
ALL_ROUND_DEG = 105.0
ALL_ROUND_STEP_DEG = 15.0
DESK_X_MAX_MM = 200.0
DESK_Y_MAX_MM = 60.0
# The base's suction cups, which the cup is kept clear of (the owner's). They stand outside the library's arm-base
# limit, so a move between points that would pass nearer one goes round instead, along an arc at least ARC_MIN_MM
# out, in steps of ARC_STEP_MM.
LEG_CUPS = ((80.0, -80.0), (-80.0, -80.0), (80.0, 80.0), (-80.0, 80.0))
LEG_CUP_CLEAR_MM = 45.0
ARC_MIN_MM = 160.0
ARC_STEP_MM = 40.0
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
REPEAT_MM = 0.05
MAX_TRIES = 12
# A servo stops a pulse more or less short from one try to the next, so a held height is aimed at up to this often.
AIMS_EACH = 2
# Each try comes down PRECISE from this far over it. A seal leaves the cup holding a little vacuum after the
# release, so after one the cup lifts TRAVEL_MM over the guess first, where it is known to come free.
DOWN_MM = 4.0
TRAVEL_MM = 10.0
# Never sent lower than this under the guess, or higher over it.
LOWEST_MM = 2.0
HIGHEST_MM = 4.0
# Coming down, the arm stops over where it is sent (STATUS-jackal §14): about this much until the misses say, and
# each servo this many pulses past its command (STATUS-jackal §15); from the last LEARN_FROM misses.
SHORTFALL_MM = 0.8
SHORTFALL_SPAN_MM = (-0.5, 2.5)
STOP_SHORT_PULSES = (0.0, 1.0, -3.0)
LEARN_FROM = 12
# A reading further than this from where the cup was sent ends the point: the arm did not get there.
OFF_MM = 5.0
# The first pump of a session has read like a seal in the air (STATUS-jackal §15): one is spent before any try.

Aim = Tuple[int, int, int]


class Stopped(Exception):
    """Ctrl-C."""


class Skipped(Exception):
    """The point cannot be done: the library would not make a move, or the arm did not get there."""


def letter_rings() -> List[Tuple[float, List[float]]]:
    """Each ring's reach and bearings for `letter`: every other ring's bearings half a step round."""
    rings = []
    for index, reach in enumerate(RINGS_MM):
        step = math.degrees(ALONG_RING_MM / reach)
        shift = step / 2.0 if index % 2 else 0.0
        count = int(BEARING_MAX_DEG // step) + 1
        rings.append((reach, [shift + step * k for k in range(-count, count + 1)
                              if abs(shift + step * k) <= BEARING_MAX_DEG]))
    return rings


def affirm_rings() -> List[Tuple[float, List[float]]]:
    rings = []
    for reach, bearings in letter_rings():
        if reach == ALL_ROUND_REACH_MM:
            count = int(ALL_ROUND_DEG // ALL_ROUND_STEP_DEG)
            bearings = [ALL_ROUND_STEP_DEG * k for k in range(-count, count + 1)]
        else:
            bearings = [bearing for bearing in bearings if abs(bearing) <= AFFIRM_BEARING_DEG
                        or f"r{reach:.0f} b{bearing:+.0f}" in AFFIRM_AGAIN]
        rings.append((reach, bearings))
    return rings


def layout(plan: str) -> List[Tuple[str, float, float]]:
    """(label, x, y): ring by ring, out from the base, each the other way round from the last; the reference first,
    every REFERENCE_EVERY points and last."""
    is_letter = plan == "letter"
    grid = []
    for index, (reach, bearings) in enumerate(letter_rings() if is_letter else affirm_rings()):
        for bearing in reversed(bearings) if index % 2 else bearings:
            x, y = reach * math.sin(math.radians(bearing)), -reach * math.cos(math.radians(bearing))
            if is_letter and (abs(x) > AREA_HALF_X_MM or not AREA_Y_MM[0] <= y <= AREA_Y_MM[1]):
                continue
            if abs(x) > DESK_X_MAX_MM or y > DESK_Y_MAX_MM:
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


def next_try(x: float, y: float, tries: List[Try], aimed: Counter, guess: float, start: float, shortfall: float,
             stop_short: Tuple[float, float, float]) -> Optional[Tuple[Position, Optional[Aim]]]:
    """The next try: where to send the cup, true, and the pulses it is to be held at, if it aims at some; None when
    the search is over."""
    sealed, missed = split(tries)
    if len(tries) >= MAX_TRIES:
        return None
    if sealed and missed:
        top, bottom = max(sealed), min(missed)
        between = [(aimed[pulses], abs(height - (top + bottom) / 2.0), pulses) for height, pulses
                   in held_heights(x, y, top + SAME_MM, bottom - SAME_MM) if aimed[pulses] < AIMS_EACH]
        if top > bottom or not between:
            return None
        aim = min(between)[2]
        command = tuple(round(pulse - short) for pulse, short in zip(aim, stop_short))
        # The command's whole pulses themselves: the library sends half a pulse past them, so that the firmware's
        # cut to whole ones lands on them (`geometry.nearest_step_command()`). Adding the half here as well would
        # land a whole pulse high.
        target = raw.true_position(geometry.forward(geometry.pulse_to_deg(command)))
        return target, aim
    if sealed:
        sent = max(done.sent for done in tries) + UP_MM
    elif missed:
        sent = min(done.sent for done in tries) - STEP_MM
    else:
        sent = guess + start - shortfall
    sent = min(max(sent, guess - LOWEST_MM), guess + HIGHEST_MM)
    return None if any(abs(sent - done.sent) < REPEAT_MM for done in tries) else ((x, y, sent), None)


def around_the_cups(start: Tuple[float, float], end: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Where to pass on the way from `start` to `end`: nowhere if the straight line keeps clear of the base's
    suction cups, else an arc round the base axis."""
    if all(_to_segment(cup, start, end) >= LEG_CUP_CLEAR_MM for cup in LEG_CUPS):
        return []
    radius = max(math.hypot(*start), math.hypot(*end), ARC_MIN_MM)
    first, last = geometry.bearing_degrees(*start), geometry.bearing_degrees(*end)
    steps = max(1, math.ceil(math.radians(abs(last - first)) * radius / ARC_STEP_MM))
    bearings = [math.radians(first + (last - first) * step / steps) for step in range(steps + 1)]
    return [(radius * math.sin(bearing), -radius * math.cos(bearing)) for bearing in bearings]


def _to_segment(point: Tuple[float, float], start: Tuple[float, float], end: Tuple[float, float]) -> float:
    (px, py), (ax, ay), (bx, by) = point, start, end
    length = (bx - ax) ** 2 + (by - ay) ** 2
    t = 0.0 if length == 0.0 else min(1.0, max(0.0, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / length))
    return math.hypot(px - (ax + t * (bx - ax)), py - (ay + t * (by - ay)))


def stop_short_of(entry: dict) -> Tuple[float, float, float]:
    """Each servo's pulses read, less those it was commanded, for a try written to the file. Tries written before
    6 Nov have no `commanded`: the firmware then cut the raw target's pulses down."""
    commanded = entry.get("commanded") or [int(pulse) for pulse in pulses_of(tuple(entry["sent_raw"]))]
    return tuple(read - sent for read, sent in zip(pulses_of(tuple(entry["read_raw"])), commanded))


class Prober:
    """`entries` are this run's so far; `guides` another run's, whose edges only say where to look first."""

    def __init__(self, arm: MaxArm, entries: List[dict], guides: List[dict]) -> None:
        self.arm = arm
        self.found = [entry for entry in rebuild(entries) + rebuild(guides) if entry["found"]]
        self.here: Position = home_position()     # where the arm was last sent
        misses = [entry for entry in entries if entry["kind"] == "try" and entry["state"] == SealState.LOOSE.value]
        self.shortfalls = [entry["read"][2] - entry["sent"] for entry in misses]     # in the air: read less sent
        self.stop_shorts = [stop_short_of(entry) for entry in misses]
        self.is_pump_on = False
        self.is_lift_needed = False
        self.stopping = threading.Event()

    # --- one point ----------------------------------------------------------

    def find(self, label: str, x: float, y: float) -> dict:
        guess, is_near = self.guess(x, y)
        travel = guess + TRAVEL_MM
        self.go(*self.here[:2], max(self.here[2], travel), MoveType.SLOW)
        for via in around_the_cups(self.here[:2], (x, y)):
            self.go(*via, travel, MoveType.FAST)
        self.go(x, y, travel, MoveType.FAST)
        tries: List[Try] = []
        aimed: Counter = Counter()
        while True:
            chosen = next_try(x, y, tries, aimed, guess, START_MM if is_near else START_ALONE_MM, self.shortfall(),
                              self.stop_short())
            if chosen is None:
                break
            target, aim = chosen
            if aim:
                aimed[aim] += 1
            tried, library = self.try_at(target, travel)
            tries.append(tried)
            entry = {"kind": "try", "point": label, "x": target[0], "y": target[1], "sent": target[2],
                     "sent_raw": raw.raw_position(target),
                     "commanded": geometry.commanded_pulses(raw.raw_position(target)),
                     "aim": aim, "read": tried.read, "read_raw": tried.read_raw,
                     "state": tried.state.value, "library": library.value, "dip": tried.dip, "pulled": tried.pulled,
                     "time": time.time()}
            _write(entry)
            if tried.state is SealState.UNKNOWN:
                raise SystemExit("the seal could not be told: this board does not read the pump's supply")
            if math.dist(tried.read, target) > OFF_MM:
                raise Skipped("the arm read ({:.1f}, {:.1f}, {:.1f}), sent ({:.1f}, {:.1f}, {:.1f})".format(
                    *tried.read, *target))
            if tried.state is SealState.LOOSE:
                self.shortfalls.append(tried.read[2] - target[2])
                self.stop_shorts.append(stop_short_of(entry))
        found = summary(label, x, y, guess, tries, time.time())
        _write(found)
        if found["found"]:
            self.found.append(found)
        return found

    def try_at(self, target: Position, travel: float) -> Tuple[Try, SealState]:
        """Straight up off the desk -- clear of it after a seal -- over to the target, and down onto it. The try, and
        the library's own verdict."""
        x, y, z = target
        self.go(*self.here[:2], max(self.here[2], travel if self.is_lift_needed else z + DOWN_MM), MoveType.SLOW)
        self.go(x, y, z + DOWN_MM, MoveType.SLOW)
        self.go(x, y, z, MoveType.PRECISE)
        read = self.arm.get_position()
        self.pump(True)
        library = self.arm.get_seal() or SealState.UNKNOWN
        dip, pulled = self.arm.get_seal_measures()
        self.pump(False)
        state = judged(library, pulled)
        self.is_lift_needed = state is not SealState.LOOSE
        return Try(z, read, raw.raw_position(read), state, dip, pulled), library

    def guess(self, x: float, y: float) -> Tuple[float, bool]:
        """Where to look for the desk, and whether a point near was found to say so."""
        near = [(math.dist((x, y), (entry["x"], entry["y"])), entry["top"]) for entry in self.found]
        near = [pair for pair in near if pair[0] <= NEIGHBOUR_MM]
        guess = min(near)[1] if near else FIRST_GUESS_MM
        return min(max(guess, FIRST_GUESS_MM - GUESS_SPAN_MM), FIRST_GUESS_MM + GUESS_SPAN_MM), bool(near)

    def shortfall(self) -> float:
        recent = [value for value in self.shortfalls if SHORTFALL_SPAN_MM[0] <= value <= SHORTFALL_SPAN_MM[1]]
        recent = recent[-LEARN_FROM:]
        return statistics.median(recent) if len(recent) >= 3 else SHORTFALL_MM

    def stop_short(self) -> Tuple[float, float, float]:
        recent = self.stop_shorts[-LEARN_FROM:]
        if len(recent) < 3:
            return STOP_SHORT_PULSES
        return tuple(statistics.median(servo) for servo in zip(*recent))

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


def _write(entry: dict) -> None:
    with OUT.open("a") as out:
        out.write(json.dumps(entry) + "\n")


def _read(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _said(found: dict) -> str:
    if not found["found"]:
        return found["why"]
    return (f"seals up to {found['top']:.2f} (raw {found['top_raw'][2]:.2f}), not at {found['bottom']:.2f}"
            + (" -- a seal read over a miss" if found["is_overlap"] else "")
            + ("; no height held between" if found["is_resolved"] else ""))


def main() -> None:
    global OUT
    plan = sys.argv[1] if len(sys.argv) > 1 else "letter"
    if plan not in ("letter", "affirm"):
        raise SystemExit("usage: find-desk.py [letter | affirm]")
    OUT = HERE / ("desk-seal.jsonl" if plan == "letter" else f"desk-seal-{plan}.jsonl")
    arm = MaxArm(config_dir=None)
    entries = _read(OUT)
    done = [entry for entry in entries if entry["kind"] == "point"]
    labels = {entry["point"] for entry in done}
    schedule = [point for point in layout(plan) if point[0] not in labels]
    if done and schedule:
        schedule.insert(0, (f"ref again {len(done)}", *REFERENCE))      # the drift across the break
    lowest, highest = FIRST_GUESS_MM - GUESS_SPAN_MM - LOWEST_MM, FIRST_GUESS_MM + GUESS_SPAN_MM + TRAVEL_MM
    reach = 2.0
    unreachable = [label for label, x, y in schedule
                   if not all(arm.can_reach(x + dx, y + dy, z) for z in (lowest, highest)
                              for dx, dy in ((0.0, 0.0), (reach, 0.0), (-reach, 0.0), (0.0, reach), (0.0, -reach)))]
    if unreachable:
        raise SystemExit(f"out of reach, nothing moved: {', '.join(unreachable)}")
    if not schedule:
        raise SystemExit(f"every point is done: {OUT}. Delete it to start over.")
    print(f"{len(schedule)} points to do, about {len(schedule) * 1.0:.0f} minutes"
          + (f"; {len(done)} already in {OUT.name}" if done else "") + ". Connecting: the arm homes.")

    prober = Prober(arm, entries, [] if plan == "letter" else _read(HERE / "desk-seal.jsonl"))

    def on_interrupt(signum, frame) -> None:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\n  stopping: the arm holds where it is and lets go of the desk; Ctrl-C again to quit",
              file=sys.stderr)
        prober.stop()

    signal.signal(signal.SIGINT, on_interrupt)
    arm.connect()
    try:
        prober.pump(True)
        prober.pump(False)
        for number, (label, x, y) in enumerate(schedule, 1):
            try:
                found = prober.find(label, x, y)
            except Skipped as reason:
                _write({"kind": "point", "point": label, "x": x, "y": y, "found": False, "why": str(reason),
                        "time": time.time()})
                print(f"{number:3d}/{len(schedule)} {label:12s} skipped: {reason}", flush=True)
                continue
            print(f"{number:3d}/{len(schedule)} {label:12s} {_said(found)}; {found['tries']} tries", flush=True)
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
