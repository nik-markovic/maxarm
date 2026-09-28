# condor — camera-to-arm calibration

Spec: [`work/PILOT-condor.md`](../work/PILOT-condor.md).
State of play: [`work/STATUS-condor.md`](../work/STATUS-condor.md).
What the exploration session established, and what it could not do:
[`work/STATUS-condor-prototype.md`](../work/STATUS-condor-prototype.md).

| directory      | holds                                                             |
| -------------- | ----------------------------------------------------------------- |
| `maxarm/`      | The arm control library, copied unchanged from `bison/`            |
| `calibration/` | `arm.py` places the cubes, `calibrate.py` maps the camera to them  |
| `config/`      | Calibration output. Generated, never hand-edited                   |
| `agenttools/`  | Agent-side tools and the offline test suites                       |
| `files/`       | Test fixtures — the frames the detector was developed against      |

**Both halves are written. The arm half has run on hardware; the camera half has not run
against a live camera yet.**

```
./calibration/arm.py --dry-run     # check every target, board off
./calibration/arm.py               # place the three cubes
./calibration/calibrate.py         # photograph them and solve the mapping
```

[`calibration/arm.py`](calibration/arm.py) takes three cubes off a stack under the nozzle and
puts them on the desk at coordinates it then exports as `CUBE_POSITIONS`.
[`calibration/calibrate.py`](calibration/calibrate.py) photographs the result and solves the
pixel-to-arm mapping into `config/`. Read [`calibration/README.md`](calibration/README.md)
before running either — in particular, three cubes give an affine fit where the camera's
perspective wants four point pairs, and that section says what to do about it.

The rest of this file is the `maxarm/` library's own documentation, inherited from `bison`
along with the code. It is accurate: nothing in the library was changed.

---

## maxarm — the MaxArm control library

A Python library for driving the Hiwonder MaxArm from a PC, over the board's **stock firmware**.
Nothing is written to the board — no files, no flash, not even a variable name.
Unplug it and it is exactly as it shipped.

Library spec: [`work/PILOT-bison.md`](../work/PILOT-bison.md).
Decisions, evidence and what is still untested: [`work/STATUS-bison.md`](../work/STATUS-bison.md).

```python
from maxarm import ExclusionZone, Limits, MaxArm

MUG = ExclusionZone("mug", x_min=60.0, x_max=110.0, y_min=-160.0, y_max=-110.0)

with MaxArm(limits=Limits(x=(0.0, None)), zones=(MUG,)) as arm:
    arm.move_to(250.0, -110.0, 110.0)      # one command, ~200 mm/s
    arm.pick_at(200.0, -150.0, 52.0)       # travels high, steps the last 15 mm down
    arm.place_at(235.0, -150.0, 52.0)
    arm.home()
```

`move_to()` decides for itself whether the move needs a detour around the robot's base,
a lift and a guarded descent onto the desk, or a nudge at the end.
**That decision is the library's job, so none of it is a knob.**

[`main.py`](main.py) is the worked example: a step-by-step guided tour that configures the desk,
then walks through fast sweeps, the reach limits, both no-fly zones and a pick-and-place, printing
what it is about to do and what came back. Run it straight through — there is nothing to choose.

```
./main.py                          # the guided tour, on real hardware
../bison/agenttools/run-tests.py            # 146 offline checks, ~40 s, no board needed
../bison/agenttools/try-moves.py --dry-run  # print the route of every motion stage
../bison/agenttools/try-moves.py            # prompted stage-by-stage testing, Ctrl-C safe
```

## What you configure

Four fields, all optional. Anything left `None` takes its default.

```python
ArmConfig(
    connection=ConnectionMethod.USB,           # UART and BLE raise NotImplementedError
    device="/dev/ttyUSB0",                     # or $MAXARM_DEVICE
    limits=Limits(x=(0.0, None), z=(48.0, None)),
    zones=(ExclusionZone("mug", 80, 140, -220, -160),
           ExclusionZone("card box", 80, 140, -220, -160, z_max=70.0)),
)
```

`MaxArm(ArmConfig(...))` and `MaxArm(limits=..., device=...)` are the same thing; use whichever
reads better where you are standing.

`Limits` are **the desk, not the arm**: open on every axis except a z floor of 48 mm, which is what
stops the nozzle scraping. Lower it deliberately (44 or so) to snug onto a card. The arm's own
limits are computed in `geometry.py`, not configured, and cannot be relaxed.

**The base exclusion square (|x| ≤ 70, |y| ≤ 70, at every height) is hardcoded and cannot be
switched off.** Inside it the cup or its air hose strikes the robot's own casting. Your zones sit
on top of it.

### What you do not configure

Step sizes, speeds, settle times, margins, sag thresholds and retarget budgets are measurements,
not preferences. They live in [`maxarm/tuning.py`](maxarm/tuning.py) with the measurement written
above each one. Five answer to the environment, because they are the five a different desk could
plausibly need:

| variable                | changes                                          | default |
| ----------------------- | ------------------------------------------------ | ------- |
| `MAXARM_DEVICE`         | serial port                                      | `/dev/ttyUSB0` |
| `MAXARM_TRAVEL_SPEED`   | mm/s in free space; drop it for a cautious run   | 200     |
| `MAXARM_Z_FLOOR`        | default floor, when the desk is not this desk    | 48      |
| `MAXARM_DESK_SAG`       | mm of droop that counts as touching down         | 3.0     |
| `MAXARM_RETARGET_LIMIT` | how far a target may be moved to make it legal   | 20      |

If anything else turns out to need changing, measure it again and edit the number — do not grow
the API back.

## The API

```python
arm = MaxArm(...)             # see above
arm.connect()                 # NB: this resets the board, which homes the arm
arm.disconnect()              # or use `with MaxArm() as arm:`
```

**Moving.** Never raises for a bad target; the reason is the return value, and `MoveResult` is
falsy unless the arm arrived where it was asked.

```python
result = arm.move_to(x, y, z, on_step=None)
result.is_ok, result.status, result.residual_mm, result.position
arm.move_relative(dx=0.0, dy=0.0, dz=-5.0)      # jog, for a UI
arm.home()
arm.stop()                                       # safe from a signal handler
```

**Nozzle and pick-and-place.**

```python
arm.grip(); arm.release()            # release holds the valve open 1 s on the board
arm.set_nozzle_angle(45.0)           # cup rotation, -90..+90, waits it out
arm.pick_at(x, y, z, approach_mm=30.0, rotation_deg=0.0)
arm.place_at(x, y, z)
```

**Reading.** `get_state()` is free and never blocks — every step of every move refreshes it, so a
UI can poll it at 30 Hz *during* a move. Pass `is_fresh=True` to go to the wire instead.

```python
state = arm.get_state()              # position, joints, pulses, suction, last status
arm.get_position()                   # settled, median of several reads
arm.get_joint_angles()               # measured, from the servos
arm.get_joint_frame()                # base/shoulder/elbow/wrist/tip in 3D -- for a renderer
```

**Asking, without moving.** Pure arithmetic, no round trip, works before `connect()`.

```python
arm.is_reachable((200.0, -100.0, 60.0))
arm.check_target(target)             # -> (aim point, MoveStatus, explanation)
arm.get_exclusion_zones()
```

## Statuses

Nine, and the test for whether there should be a tenth is whether you would *do* something
different about it.

| status           | meaning                                                             |
| ---------------- | ------------------------------------------------------------------- |
| `REACHED`        | did what you asked                                                   |
| `APPROXIMATED`   | arrived at the nearest legal point; see `residual_mm`                |
| `UNREACHABLE`    | refused: no legal point near enough to aim at                        |
| `NO_FLY`         | refused: the target, or every route to it, crosses an exclusion zone |
| `NOT_CONNECTED`  | refused: call `connect()` first                                      |
| `DESK_CONTACT`   | the nozzle touched down early; the arm was backed off                |
| `BLOCKED`        | moved, then stopped advancing — obstruction, clamp, or a refusal     |
| `NOT_RESPONDING` | accepted everything and never moved — servo power, torque or mode    |
| `STOPPED`        | `stop()` was called                                                  |

## How a move actually runs

**The arm does not move in straight lines.** `set_position()` solves the IK for the endpoint and
gives all three servos the same duration, so they interpolate in *pulse* space and the tip swings.
`geometry.pulse_path()` says exactly where: about 2 mm off the straight line on a short hop, 21 mm
on a long sweep, 123 mm on a reach across the front of the base.

So a move is broken into **hops**, and there are only two ways to drive one:

| how         | when                                                                 |
| ----------- | --------------------------------------------------------------------- |
| **flown**   | one command, watched while it travels — the normal case                |
| **stepped** | 2 mm at a time, measured after each — the last 15 mm onto the desk     |

Most moves are a single flown hop. The extra hops appear only when the move needs them: an escape
if the arm starts inside a zone, a lift and a detour arc if the straight line crosses one, an
approach to the top of the desk guard band before a descent. A hop whose swing would leave safe
space is **halved until it would not** — splitting is a routing decision, so every hop that
survives it is still one command.

A flown hop is polled every 120 ms while it travels — reads only, no commands. That is what makes
`stop()` immediate: there is no stop command on this board, but retargeting the arm to the pose it
currently occupies is one.

After the last hop the arm is read back, and if it is more than 3.5 mm out the miss is **mirrored
and re-commanded**, up to twice. Re-sending the same coordinate was measured to gain nothing — as
far as the arm is concerned it is already there — so a correction has to aim past.

## Layout

| file           | owns                                                                    |
| -------------- | ----------------------------------------------------------------------- |
| `geometry.py`  | kinematics, and the limits the firmware enforces. Pure arithmetic.       |
| `zones.py`     | exclusion zones as positions **and as paths**                            |
| `tuning.py`    | the measured constants, with their provenance. Internal.                 |
| `route.py`     | retargeting, detours, staging, swing splitting. Internal, no board.      |
| `transport.py` | bytes on a wire                                                          |
| `protocol.py`  | every expression ever sent to the board, in one place                    |
| `motion.py`    | driving a route with the readback in the loop. Internal.                 |
| `maxarm.py`    | `MaxArm` — the class callers use                                         |

The first four need no hardware and no board, which is why most of the test suite is instant.

## Connection methods

Only **USB** works on stock firmware, and that is a property of the board rather than a gap here:

- **UART** (4-pin header, FTDI dongle) would carry Hiwonder's `0xAA 0x55` binary protocol, but
  stock firmware has no parser for it — `MaxArm_ctl.py` and the five board methods it calls are
  simply not on the board. Using it means replacing `main.py`, `espmax.py` and `SuctionNozzle.py`,
  and `work/GUIDELINES.md` forbids touching the board. It raises `NotImplementedError` saying so.
- **BLE** is live on stock firmware but is a one-way 3 mm jog protocol with no position readback.

If the firmware is ever swapped, write a second `BoardProtocol` and nothing above it changes.

## Three things to expect on real hardware

1. **Connecting moves the arm.** Opening the port drives DTR/RTS onto the ESP32's EN/BOOT pins and
   hard-resets the chip — measured, five opens out of five, even with both held low. The board
   re-runs `main.py`, which homes the arm and rotates the nozzle. Measured on this board: 4.7 s
   once and 15.1 s twice, the long ones being the board's BLE init timeout. The budget is 25 s.
2. **Accuracy is about ±3 mm per axis**, worse at full reach, and readback is quantised to 1 mm.
   Nothing in software improves it. Design the demo around it rather than against it.
3. **An arm that accepts commands and does not move is a real state**, seen once already. It is not
   a limit and not a bad target: servo torque, servo mode, or servo power, none of which is
   readable on this firmware. `connect()` checks what it can up front, `move_to()` reports
   `NOT_RESPONDING` for the rest, and the remedy is to power-cycle the arm at the barrel jack.
