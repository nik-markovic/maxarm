# agenttools/

Agent-side tools. Not part of the demo, and the owner is not expected to run most of them.

**`condor/maxarm/` is the working copy of the library now.** `bison/` is closed — its status
doc describes a state that was verified, and it should keep describing it. Everything here came
across so that condor can check its own library rather than reaching back into a frozen pilot.

## Offline — no hardware, no motion

| file | what it checks |
| ------------------ | ---------------------------------------------------------------- |
| `run-tests.py` | Every library suite: geometry, motion, arm, example. ~11 s. |
| `test-scene.py` | `calibration/arm.py` against a fake board. Run separately — the |
| | hyphen in its name means it cannot be imported like the others. |
| `test-calibration.py` | The camera half: `detect.py`, `mapping.py`, and the parts of |
| | `calibrate.py` that do not need a device. Also run separately. |
| `test-detect.py` | The detector alone: cubes rendered through a pinhole camera, and |
| | the owner's frame `files/cubes2.png`. Also run separately. |
| `test_geometry.py` | Kinematics against the validated reference, 1 200 poses a run. |
| `test_motion.py` | Routing: detours, staging, swing splitting. Board-free. |
| `test_arm.py` | End-to-end against the fake board, including the ways it lies. |
| `test_example.py` | `main.py` and `try-moves.py`, executed as written. |
| `fake_arm.py` | A MaxArm that exists only in RAM: board, firmware quirks, port. |
| `harness.py` | `check()` and `run()`. Deliberately not a test framework. |

```
../.venv/bin/python agenttools/run-tests.py          # every library suite
../.venv/bin/python agenttools/run-tests.py motion   # just one
../.venv/bin/python agenttools/test-scene.py         # the scene playback
../.venv/bin/python agenttools/test-calibration.py   # the camera half
../.venv/bin/python agenttools/test-detect.py        # the cube detector on its own
```

## On the real board

| file | what it does |
| ----------------------- | ------------------------------------------------------------ |
| `validate-on-board.py` | Host kinematics against the live board. **Zero commanded |
| | motion** — the only movement is the homing the firmware does |
| | on every port-open reset. Run this first after any library |
| | change; if it fails, nothing above it can be trusted. |
| `diagnose-servo-bus.py` | Why did the arm accept a move and not make it? Reads only. |
| `try-moves.py` | Stage-by-stage motion test, smallest risk first, prompted and |
| | Ctrl-C safe. `--dry-run` prints every route without a board. |
| `pulse-suction.py` | Pulses the cup on and off with no arm motion at all, for |
| | watching the pump on its own. `--vent` adds the second vent |
| | pulse `release()` no longer does. |

## Seeing what is actually sent

`MAXARM_TRACE=1` echoes every expression that reaches the board, on stderr, with timestamps and
the board's reply:

```
MAXARM_TRACE=1 ./calibration/arm.py 2>trace.txt
```

```
   2.406s > nozzle.off()
   2.409s < (no output) [3 ms]
   2.409s # suction off: tip (100, -254, 76) -> (100, -254, 78), dz +2 mm, moved 2 mm
   6.408s > print('#R#', repr(arm.set_position((100.0, -254.0, 100.0), 16)))
```

`>` sent, `<` answered, `!` refused, `#` a note. `protocol.py` is the only place anything reaches
the board and `run()` is the only way out of it, so **the trace is complete by construction**: a
command that is not in it was not sent. Gaps in the timestamps are proof that nothing was sent
during them, which is the question worth asking when the arm does something unprompted.

The `#` notes cover the two things a command log cannot show on its own: how far the tip moved
when the cup was switched (the cup is a bellows, and it only contracts against a seal — so that
number is the only evidence this firmware gives that a pick sealed), and any correction nudge,
one line each and only when one fires.

Tracing adds two position reads per suction change and nothing else. It never changes what the
arm does.

## The cube detector

**The detector itself now lives in [`calibration/detect.py`](../calibration/detect.py)** —
`calibrate.py` is built on it, and this directory is the part of the tree that gets thrown away
between pilots. The code moved unchanged.

| file | what it is |
| --------------- | ---------------------------------------------------------------- |
| `find-cube.py` | The command line over it: runs the detector on one image, prints |
| | what it found and draws the overlay. Reports where each cube's |
| | **base** sits — not its blob centre, which is ~40 mm wrong. |

```
../.venv/bin/python agenttools/find-cube.py ../files/cubes1.jpeg -o /tmp/overlay.jpg
../.venv/bin/python agenttools/find-cube.py ../files/red-cube.jpeg -c red
```

`files/cubes1.jpeg` and `files/red-cube.jpeg` are the regression fixtures: any rewrite should
still land all three cubes in `cubes1.jpeg` and put the red base at roughly (408, 382).

**Read `work/STATUS-condor-prototype.md` before building on the detector.** It records what it
gets right, where its accuracy runs out, and the two approaches that were tried and failed.
