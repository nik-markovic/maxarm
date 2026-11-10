"""Whether the cup has sealed, told by the supply the pump runs off.

The servos and the pump run off the barrel jack's supply, and every position read samples it too (`protocol.py`,
`READ_SUPPLY`). The pump drawing air works harder and pulls the supply down; against a seal it hardly does. So the
supply is read with the pump off and then on, the arm standing still: how far it fell -- the pump's dip -- says
sealed, loose, or in between.

The servos pull on the same supply, by as much as the pump when pressed onto the desk or holding after a move up.
So both reads are taken where the arm stands, without a move between them, and a move ends the verdict. And the
seal itself changes what the servos do: the cup is a bellows, and against a seal it contracts, pulling the arm
down. Servos pushing the cup down are eased by that, and the dip shrinks; servos holding it short of a target above
strain against it, and the dip grows as if the cup drew air. So a cup pulled down is sealed, whatever the dip: the
pump alone never moves the arm.

It holds for a cup landed on what does not move -- the desk, a piece pressed onto it. A light piece is pulled up
into the cup instead of the cup down, and what the supply reads with a piece on the cup is not told apart from an
empty cup, there or after a lift: whether something was picked up is not this module's to say.
"""

from enum import Enum
from statistics import fmean
from typing import Optional, Sequence, Tuple


class SealState(Enum):
    SEALED = "sealed"
    LOOSE = "loose"              # the cup is drawing air
    UNCERTAIN = "uncertain"      # between the two: the cup only just touching, or an arm that reads otherwise
    UNKNOWN = "unknown"          # nothing to judge it by


# The pump's dip, in the pin's counts, set from the owner's arm (STATUS-jackal §14): between them, uncertain.
SEALED_BELOW = 10.0
LOOSE_ABOVE = 12.0
# Read at the top of the pin's range, the supply cannot show a dip.
SUPPLY_TOP = 4080.0
# Reads taken this long after the pump goes on judge the seal; the cup empties first (STATUS-jackal §14).
PUMP_START_S = 0.75
# How far down the cup must have pulled the arm, from where it stood before the pump, to have sealed: past the
# most a loose cup's readings moved (STATUS-jackal §14).
PULLED_MM = 0.5
# The board's vent valve draws from the same supply for a second after the pump goes off.
VENT_S = 1.2


def level(supplies: Sequence[Optional[float]]) -> Optional[float]:
    """The supply over a run of reads; None without one."""
    known = [supply for supply in supplies if supply is not None]
    return fmean(known) if known else None


def judge(pump_off: Optional[float], pump_on: Optional[float],
          pulled: Optional[float]) -> Tuple[SealState, Optional[float], Optional[float]]:
    """The seal, from the supply before the pump went on and after, and how far the cup pulled the arm down in mm;
    with the pump's dip and the pull it was judged by."""
    is_read = pump_off is not None and pump_on is not None and max(pump_off, pump_on) < SUPPLY_TOP
    dip = pump_off - pump_on if is_read else None
    if pulled is not None and pulled >= PULLED_MM:
        return SealState.SEALED, dip, pulled
    if dip is None:
        return SealState.UNKNOWN, None, pulled
    if dip < SEALED_BELOW:
        return SealState.SEALED, dip, pulled
    if dip > LOOSE_ABOVE:
        return SealState.LOOSE, dip, pulled
    return SealState.UNCERTAIN, dip, pulled


def describe(state: SealState, dip: Optional[float], pulled: Optional[float]) -> str:
    said = {SealState.SEALED: "the cup sealed", SealState.LOOSE: "the cup is not sealed: it draws air",
            SealState.UNCERTAIN: "the seal is uncertain: the cup may be only just touching",
            SealState.UNKNOWN: "the seal could not be told: the supply was not read"}[state]
    measures = ([] if dip is None else [f"pump dip {dip:.1f}"]) + \
               ([] if pulled is None else [f"pulled down {pulled:.1f} mm"])
    return f"{said} ({', '.join(measures)})" if measures else said
