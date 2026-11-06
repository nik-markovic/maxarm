"""What find-desk.py and fit-desk.py share: a try, whether it sealed, a point's edge from its tries, and the heights
the arm can be held at."""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from maxarm import SealState, geometry, raw
from maxarm.geometry import Position

Pulses = Tuple[float, float, float]

# A seal pulls the arm down. The library also calls a seal by the pump's dip alone; one such call came with no pull
# at all, over a miss, in the air (STATUS-jackal §15). So a seal here must have pulled the arm down this far too;
# without it, it is uncertain.
PULL_MIN_MM = 0.1
# The heights held "over a point": within this much of its reach, wherever the base's pulse turns it.
REACH_WINDOW_MM = 1.5
# Servos 2 and 3 searched this many pulses either side for them.
PULSE_SPAN = (8, 14)
# Two heights read this close are one.
SAME_MM = 0.02


@dataclass(frozen=True)
class Try:
    sent: float                  # true z sent
    read: Position               # true: where the servos said the cup came down, before the pump
    read_raw: Position           # the same, the firmware's own
    state: SealState             # judged()
    dip: Optional[float]
    pulled: Optional[float]


def judged(library: SealState, pulled: Optional[float]) -> SealState:
    """The library's verdict, a seal kept only if it pulled the arm down."""
    if library is SealState.SEALED and (pulled is None or pulled < PULL_MIN_MM):
        return SealState.UNCERTAIN
    return library


def split(tries: List[Try]) -> Tuple[List[float], List[float]]:
    """The heights read where the cup sealed, and where it did not -- an uncertain seal is not one."""
    return ([done.read[2] for done in tries if done.state is SealState.SEALED],
            [done.read[2] for done in tries if done.state in (SealState.LOOSE, SealState.UNCERTAIN)])


def summary(label: str, x: float, y: float, guess: Optional[float], tries: List[Try], when: float) -> dict:
    """A point: the highest reading that sealed (`top`, and `top_raw` its raw position), the lowest that did not
    (`bottom`), and whether the arm holds no height between them (`is_resolved`); or why there is no edge."""
    sealed, missed = split(tries)
    found = {"kind": "point", "point": label, "x": x, "y": y, "guess": guess, "tries": len(tries),
             "found": bool(sealed and missed), "time": when}
    if sealed and missed:
        top = max((done for done in tries if done.state is SealState.SEALED), key=lambda done: done.read[2])
        between = held_heights(x, y, top.read[2] + SAME_MM, min(missed) - SAME_MM)
        found.update(top=top.read[2], top_raw=list(top.read_raw), bottom=min(missed),
                     is_overlap=max(sealed) > min(missed), is_resolved=max(sealed) <= min(missed) and not between)
    elif sealed:
        found["why"] = f"sealed at every height tried, up to {max(sealed):.1f}"
    elif missed:
        found["why"] = f"no seal down to {min(missed):.1f}"
    else:
        found["why"] = "nothing tried"
    return found


def pulses_of(position: Position) -> Pulses:
    """The servos' pulses, fractional, for a raw position: the firmware's kinematics backwards."""
    return geometry.deg_to_pulse(geometry.inverse(position))


def held_heights(x: float, y: float, low: float, high: float) -> List[Tuple[float, Tuple[int, int, int]]]:
    """The heights the arm can be held at, true z, from `low` to `high` over (x, y): each with the whole pulses that
    hold it, lowest first. The firmware holds whole pulses only; the base's moves the cup round, not up."""
    base, upper, fore = pulses_of(raw.raw_position((x, y, (low + high) / 2.0)))
    reach = math.hypot(x, y)
    held = []
    for pulse2 in range(round(upper) - PULSE_SPAN[0], round(upper) + PULSE_SPAN[0] + 1):
        for pulse3 in range(round(fore) - PULSE_SPAN[1], round(fore) + PULSE_SPAN[1] + 1):
            pulses = (round(base), pulse2, pulse3)
            at = raw.true_position(geometry.forward(geometry.pulse_to_deg(pulses)))
            if abs(math.hypot(at[0], at[1]) - reach) <= REACH_WINDOW_MM and low <= at[2] <= high:
                held.append((at[2], pulses))
    return sorted(held)


def rebuild(entries: List[dict]) -> List[dict]:
    """Every point finished in a desk-seal.jsonl, worked out again from its tries as judged() has them now. A
    point passed without an edge stays as it was written."""
    tries: Dict[str, List[Try]] = defaultdict(list)
    for entry in entries:
        if entry["kind"] == "try":
            state = judged(SealState(entry.get("library", entry["state"])), entry["pulled"])
            tries[entry["point"]].append(Try(entry["sent"], tuple(entry["read"]), tuple(entry["read_raw"]), state,
                                             entry["dip"], entry["pulled"]))
    return [summary(entry["point"], entry["x"], entry["y"], entry.get("guess"), tries[entry["point"]],
                    entry["time"]) if entry["found"] else entry
            for entry in entries if entry["kind"] == "point"]
