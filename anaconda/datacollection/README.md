# anaconda/datacollection — measuring the arm's reach envelope

`trace-contour.py` produces the deliverable. The rest support it.

| File | Role |
| --- | --- |
| `trace-contour.py` | **Measures the reach boundary and writes the X,Y,Z CSV.** |
| `calibrate-readback.py` | Characterises the servos; `--mode joints` is a zero-motion safety check. |
| `test-endpoint.py` | Is this one (X,Y,Z) reachable? Useful for spot checks. |
| `maxarm_link.py` | Raw-REPL transport. Read-only toward the board. |
| `envelope_model.py` | Motion-free reachability prediction from the board's own kinematics. |
| `reach_probe.py` | Guarded motion: stepping, stall detection, no-fly enforcement, settling. |
| `calibration.py` | Measurement routines behind `calibrate-readback.py`. |

Findings and their evidence: `work/STATUS-anaconda.md`. Short version: `work/SUMMARY-anaconda.md`.
Offline tests: `anaconda/agenttools/fake-arm-smoke-test.py` (49 checks, no hardware needed).

Modules use snake_case because Python cannot import a hyphenated name; CLIs are kebab-case per
`work/GENERAL_GUIDELINES.md`.

## Producing the envelope

```
./trace-contour.py --out reach.csv                       # all planes, ~30-40 min
./trace-contour.py --out reach.csv --z 204 --z-to 204    # one plane, to rehearse
./trace-contour.py --out reach.csv --resume              # continue after Ctrl-C
```

**Plot rows where `limit_kind == 'arm'`.** Those are points where the arm refused. Everything else is a
fence we imposed — the X>=0 testing cut, the no-fly square — and drawing them would put walls in the
envelope that do not exist.

Ctrl-C is safe at any point: rows flush as they are found and the arm stops where it is.

## Why the readback matters

`ESPMax.set_position()` is not a trustworthy oracle. It returns `None` inside a 50 mm cylinder around
the base axis, `False` when the IK throws, and — the damaging case — **`True` while
`set_servo_in_range()` has silently clamped servo 2 or 3**, leaving the arm somewhere else entirely.

So every step commands a move and then reads the pose back off the bus servos. Limits are found by
**loss of progress**, not by command-vs-readback error: progress needs no calibration, and the arm's
ordinary ±2.6 mm drift never reads as an edge.

## How the tracer works

The boundary is described in **polar terms per Z plane**, because that is how the arm moves. Per
bearing from the base axis:

1. Ask the board which radii along that bearing are reachable — no motion, one round trip per ray.
2. Probe outward to find where the arm stops reaching out.
3. Probe inward to find where it stops folding in.

Probing pushes *along the bearing*, so X and Y advance together and each probe finds the furthest
point in that direction. The frontmost reach is simply the bearing at −90°; no axis is special-cased.

Both ends matter. Near the top the reachable set is a narrow **annulus**, not a disc: at the reset
pose the arm spans X −66…+66 but Y only −173…−151, and that inner edge is the arm refusing to fold
further. An earlier version described the boundary as min/max X per Y row, which cannot represent an
annulus — at X=0 it sees `x_min = 0`, calls it the mirror cut, and loses the inner limit entirely.

A probe that ends on our own overshoot cap rather than on the arm keeps extending, up to
`--max-overshoot`. Otherwise a bad prediction silently costs a boundary point.

## The structure the data shows

At fixed Z, **the radius does not depend on bearing** — measured spread across a whole fan is 1.2 mm
at z=104 and 0.5 mm at z=84, against a ±2.6 mm noise floor. So the envelope is an annulus cut by a
base-rotation fan, and reachability reduces to three functions of Z:

```
reachable(x, y, z)  <=>  r_min(z) <= hypot(x, y) <= r_max(z)
                    and  bearing within the fan, which narrows with height
                    and  not inside the +/-70 base square
```

Fitting those three functions is the next step and is **not done** — see STATUS.

## Safety, enforced in one place

`reach_probe.py` holds all of it, so no caller can forget:

- **No-fly square**, |X| and |Y| <= 70 at every Z, checked on **targets and on paths** — two legal
  poses can have an illegal straight line between them. `route_around()` detours; `escape_no_fly()`
  recovers the arm if it ever ends up inside.
- **Z >= 48** during testing (operator choice, not the arm); **Z <= 224** always, because
  `set_position()` silently clamps above 225 and reports success.
- Sag is judged against a **locally measured baseline** and re-read before it is believed, because a
  real desk contact (~4 mm) barely clears the noise floor (2.4 mm).

Run `./calibrate-readback.py --mode joints` before trusting measurements from an unfamiliar region.
It takes seconds and never moves the arm.
