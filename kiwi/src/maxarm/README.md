# maxarm — a Python library for the MaxArm

This library drives the MaxArm robot arm over its USB cable: move the suction cup, pick a piece up and set it
down, in millimeters on your desk, or in pixels of a calibrated camera.

You do not need to know the arm's firmware, its servos or its geometry. Say where things are, and the library
plans the path, checks every limit before anything moves, and tells you how the move ended.

```python
from maxarm import MaxArm, Space

with MaxArm() as arm:                            # connects: the arm homes itself
    arm.use_coordinates(Space.DESK)              # millimeters on the desk, z the height above it
    arm.set_clearance(14.0)                      # nothing travels sideways lower than this
    arm.grab(120.0, -200.0, 4.0)                 # the cup down onto a piece 4 mm tall, suction on, back up
    arm.drop(-60.0, -180.0, 5.0, angle=0.0)      # set it down over there, turned to face +x
    arm.home()
```

The arm's base is at x 0, y 0; the arm reaches from about 95 to 270 mm out, in front of it (toward -y) and to
either side.

## Using the library

- Create a `MaxArm()` and `connect()` it, or use it in a `with` block as above. It finds the first `/dev/ttyUSB*`
  unless you pass `device=`. Connecting resets the board, which homes the arm: allow a few seconds.
- Choose the coordinates you want to work in with `use_coordinates()` (below). The default is the arm's own.
- Before `grab()`, `drop()` or `carry()`, set a clearance: how high above the desk the cup must stay while it
  travels sideways, so it does not sweep the pieces around.
- Move:

```python
from maxarm import MoveType

arm.move(x, y, z)                        # fast
arm.move(x, y, z, MoveType.SLOW)         # slowly, so nothing gets yanked
arm.move(x, y, z, MoveType.PRECISE)      # slowly, then wait until the servos have settled
arm.move(x, y, z, angle=30.0)            # and turn the cup, or the piece it holds, to face 30 degrees
arm.down_to(x, y, height)                # land the cup there exactly, and stay
arm.grab(x, y, height)                   # land, suction on, back up
arm.drop(x, y, height, angle=90.0)       # land the held piece there, facing 90 degrees, let go, back up
arm.carry((x1, y1, h1), (x2, y2, h2), facing=180.0)   # grab and drop in one, any turn
arm.home()
arm.set_suction(True)                    # by hand, for a job grab() and drop() do not fit
```

- Check what came back. Every move returns a `MoveResult`, which is true only when the move was done:

```python
result = arm.grab(250.0, -150.0, 4.0)
if not result:
    print(result.detail)       # "292 mm from the base axis, past the 270 mm reach limit"
```

  `REFUSED` means nothing moved and `detail` says why. `STOPPED` means `stop()` or Ctrl-C cut it short.
- Ask without moving: `can_reach(x, y, z)` works even before `connect()`. `get_position()` says where the servos
  say the cup is.
- Stop at any time with `arm.stop()`; it is safe to call from another thread or a signal handler.
  `arm.stop_on_ctrl_c()` makes Ctrl-C hold the arm where it is.

## Coordinates

| space    | a point is                                   | needs, in `config/` |
| -------- | -------------------------------------------- | ------------------- |
| `ARM`    | the arm's own x, y, z                        | nothing             |
| `DESK`   | x, y, and the height above the desk          | `desk.json`, or not |
| `CAMERA` | a pixel of the camera's picture, and height  | `camera.json`       |
| `CUSTOM` | your own x, y, z                             | `custom.json`       |

Desk space is the one to start with: its x and y are the arm's, and z 0 is the cup resting on the desk. Without a
`desk.json` the library assumes a flat desk under a well-placed arm, which is usually right.

In camera space, a point is the pixel where something is seen, plus its height: the top of a tile 2 mm thick is
`(px, py, 2.0)`. The same pixel at height 0 is a different point of the desk.

`camera.json` and `arm-true.json` are written by the demo's calibration pages; `config/README.md` describes each
file. To use another folder, pass `config_dir=` to `MaxArm()`, or call `use_calibration()` to read the files again.
`Spaces(config_dir).convert(point, Space.CAMERA, Space.DESK)` converts between spaces without an arm.

## Picking things up

- `grab()` lands the cup at exactly the height you give, switches the suction on and lifts. `drop()` does the
  reverse, holding the cup's vent open until it is up, so a light piece is not lifted again.
- `carry()` checks the drop before the grip, so a refusal means nothing moved. It can turn a piece any way: a turn
  bigger than the cup's 90 degrees either way from center is split between the way there and the way back. It also
  allows for where on the piece the cup gripped, so the piece's center lands where you asked.
- `retry_deeper_mm=1.5` (for `grab()` and `carry()`) checks the grip: one that does not seal is let go and tried
  again 0.5 mm deeper (`retry_step_mm`), down to that much deeper. The result's `deeper_mm` says how deep it went;
  `NOT_GRIPPED` means no try sealed.
- `get_seal()` says whether the cup sealed when the suction last went on: `SEALED`, `LOOSE` or `UNCERTAIN`. It is
  told by the pump's pull on the arm's power supply, so it is right against the desk or a piece pressed onto it,
  but cannot tell whether a light piece came up with the cup.

## Positions are true

The firmware's own positions are several millimeters off, more so far out and high up. The library corrects every
position it sends or reads (`raw.py`), so the numbers you use are where the cup really is. A calibrated arm also
carries its own correction, `config/arm-true.json`, which the library reads again whenever it changes.

To plug in a model of your own, give `converter.use()` anything with `to_true(raw)` and `to_raw(true)`.
`converter.describe()` says which one is in use.

## Limits

Every move is checked before anything is sent:

- **Reach**: a move ends 95 to 270 mm from the base axis, and never in the 120 degrees behind the arm.
- **Firmware**: what the board would refuse or silently clamp (`geometry.py`).
- **The base**: a box round the arm's base that a straight path would cross is gone round.
- **Clearance**: nothing travels sideways lower than the clearance plus what the cup holds.
- **The desk**: nothing goes more than 5 mm into it.

## What to expect

1. **Connecting moves the arm**: the board resets and homes it, in about 4 s, sometimes 15.
2. **Fast and slow moves land 1-3 mm short**, as the servos stop short of small moves. `down_to()`, `grab()` and
   `drop()` allow for it and land where you asked; a `PRECISE` move waits until they settle, but does not correct.
3. **An arm that takes commands and does not move**: power it off and on at its barrel jack.

## Settings

A few values can be changed from the environment:

| variable                      | default      | changes                                      |
| ----------------------------- | ------------ | -------------------------------------------- |
| `MAXARM_DEVICE`               | first ttyUSB | the serial port                              |
| `MAXARM_MOVE_FAST_MM_S`       | 200          | the speed of fast moves                      |
| `MAXARM_MOVE_SLOW_MM_S`       | 10           | slow moves, and landings                     |
| `MAXARM_SUCTION_GRAB_MS`      | 1500         | the wait for a grip                          |
| `MAXARM_SUCTION_RELEASE_MS`   | 3000         | the wait for a release, when the vent is not held open |
| `MAXARM_BASE_LIMIT_HALF_MM`   | 70           | the half-width of the box round the base     |
| `MAXARM_BASE_LIMIT_HEIGHT_MM` | 200          | its height                                   |

`MAXARM_TRACE=1` prints everything sent to the board, with its answer and timing.

## Inside the library

`__init__.py` lists the modules. For how a move works, read `arm.py` and `landing.py`; for the limits and the path
round them, `path.py`; for the spaces, `spaces.py`.
