# anaconda/agenttools — tools the owner will not normally run

Tests and investigations. None of this is part of the demo; expect it to be dropped before the
final project.

| File | Role |
| --- | --- |
| `kinematics_reference.py` | **Host port of the board's IK, with its limits as predicates.** Validated against the live board. |
| `validate-kinematics.py` | Proves that port matches the board, then prints the envelope it implies. |
| `dump-board-source.py` | Pulls the board's filesystem over the REPL. Read-only, no motion. |
| `probe-ik-behaviour.py` | Asks the board's IK what it does at its edges. Zero motion. |
| `fake-arm-smoke-test.py` | 49 offline checks for `datacollection/`. No hardware needed. |
| `boarddump/` | What `dump-board-source.py` retrieved, kept for diffing against the kit's `MaxArm/`. |

## Why `kinematics_reference.py` exists

The board's IK is compiled (`__espmax.mpy`), which made its limits look like something to measure.
It is not: the board's copy is byte-identical to the kit's, and the kit ships a readable Arduino
twin of the same library. Porting that twin gives the limits in closed form, exactly.

`validate-kinematics.py` is what makes it trustworthy — 336 points against the board's own
`position_to_pulses`, all agreeing bar 8 within 2.4e-5 degrees of a joint limit, where single- and
double-precision arithmetic honestly differ. Re-run it after any change to the port.

Findings: `work/STATUS-anaconda.md` section 11. Short version: `work/SUMMARY-anaconda.md`.
