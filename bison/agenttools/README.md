# bison/agenttools — checks and hardware probes

146 offline checks, no hardware, no motion, standard library only — plus two scripts that do talk
to a real board.

```
./run-tests.py              # everything offline, ~40 s
./run-tests.py motion       # one suite: geometry | motion | arm | example
./validate-on-board.py      # talks to the arm, commands NO motion
./try-moves.py --dry-run    # print the motion stages; drop --dry-run to run them
```

## The two hardware scripts

**`validate-on-board.py`** — proves the host model matches this particular board, with
`set_position()` never called. The only movement is the `go_home()` the firmware runs on every
port-open reset. Run it first; if it fails, nothing above it is worth trying. Passing as of
2026-09-22: see `work/STATUS-bison.md` §6 for the table.

**`try-moves.py`** — staged motion test, smallest risk first, prompting before each stage and
Ctrl-C safe (one Ctrl-C stops the arm, a second quits). Defaults are deliberately timid: z floor
52 mm, right half of the desk only, no suction and no descent unless `--desk` / `--nozzle` are
passed. `--speed` is the same knob as `MAXARM_TRAVEL_SPEED`; try 60 on a first run.
`test_example.py` drives every stage of it against the fake board, so it has been through its own
paces before it meets a nozzle.

This is the agent's tool, not the example — it prompts, it has a dry run, and it is meant to be
interrupted. `../main.py` is the worked example: a straight-through guided tour with no options.

| file               | what it covers                                                             |
| ------------------ | --------------------------------------------------------------------------- |
| `harness.py`       | `check()` and a runner. Deliberately not a test framework.                   |
| `fake_arm.py`      | A board in RAM: the firmware's quirks, on a fake serial port.                |
| `test_geometry.py` | Kinematics, the envelope, and agreement with the board-validated reference.  |
| `test_motion.py`   | Retargeting, zones, detours, staging, swing splitting. Pure geometry.       |
| `test_arm.py`      | End to end against the fake board, through the real transport and protocol.  |
| `test_example.py`  | Runs `main.py` and `try-moves.py` against the fake board, before either meets a nozzle. |
| `validate-on-board.py` | **Talks to the real arm. Commands no motion.** Host model vs the board. |
| `try-moves.py`     | **Moves the real arm.** Staged, prompted, Ctrl-C safe.                      |

Python modules are `snake_case` rather than the repo's usual kebab-case because they are imported
by name; the executable script keeps kebab-case.

## What the fake board reproduces

Not a robot simulator — a reproduction of the four ways this board misleads its driver:

1. `set_position()` returns `None` inside the 50 mm blind cylinder,
2. returns `False` when the IK will not solve,
3. returns **`True` while a servo sits silently clamped and nothing moves** — the only real lie,
4. droops: commanded z is not reached z near the desk, and the readback shows it *lower*, exactly
   as the owner observed (command 46, read 42).

Reachability in the fake uses the library's own `geometry`, so "the fake arm refused" means the
real one would have. The desk height, sag coefficient and tracking lag are invented — shapes of
behaviour, not measurements.

`fake_arm()` shrinks the settle times and speeds for the duration of a check (`brisk_tuning()`),
because the fake board has no servos to wait for. Margins, step sizes, thresholds and the retarget
budget are left exactly as the library ships — those are what the checks are about.

Useful handles for writing more checks:

```python
with fake_arm(config) as (arm, wire):
    wire.arm.commands            # every position ever commanded
    wire.arm.move_count          # how many were accepted
    wire.arm.is_frozen = True    # accept everything, move nothing (the silent clamp)
    wire.arm.command_delay_s     # give a move real duration, to poll it from a thread
    wire.arm.tracking_gain       # lower it to see a move that is still in flight
    wire.arm.sag_per_mm          # raise it to make the arm miss, so a nudge has work to do
    wire.nozzle.events           # ['on', 'angle:45.0', 'off', ...]
    wire.bus_servo.is_answering = False   # make position reads fail
```

## The one test that outranks the others

`test_geometry.test_matches_validated_reference` compares this library's kinematics against
`anaconda/agenttools/kinematics_reference.py`, which was checked against the live board over 336
points. **That file is the ground truth, not this one.** If they disagree, this library is wrong.
Re-run it after touching anything in `geometry.py`.
