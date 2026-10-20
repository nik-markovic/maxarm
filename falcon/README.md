# falcon — the desk calibration, on elephant's library

Spec: [`work/PILOT-falcon.md`](../work/PILOT-falcon.md).
State of play: [`work/STATUS-falcon.md`](../work/STATUS-falcon.md).

The pilot adds the calibration web app, and its first page: the desk, measured by touching the cup down on it.
The library is elephant's, with two changes (below). Move, grab and drop, in whichever coordinates you work in;
the library owns every coordinate conversion, and an app only finds things and passes their coordinates.

```
../.venv/bin/python calibration/calibration.py     # the calibration app: open http://localhost:8080
```

```python
arm = MaxArm()
arm.connect()                                   # the board resets, and homes the arm
arm.use_coordinates(Space.CAMERA)               # once: a point is a pixel and a height
arm.set_clearance(TILE_HEIGHT_MM + 10)          # once: grab() and drop() need it

for tile, (px, py) in zip(find_tiles(frame, camera, reader), SLOTS):
    arm.grab(tile.px, tile.py, TILE_HEIGHT_MM, angle=tile.angle)
    arm.drop(px, py, TILE_HEIGHT_MM + 1, angle=UPRIGHT)
```

That is [`main.py`](main.py), give or take its Ctrl-C handling and a fallback for tiles the cup cannot turn far enough.

| directory      | holds                                                                              |
| -------------- | ---------------------------------------------------------------------------------- |
| `maxarm/`      | The library                                                                        |
| `calibration/` | The calibration web app: `calibration.py`, the desk page; guide in its `README.md` |
| `config/`      | `desk.json` and `camera.json`, one matrix each; per rig, so not in git             |
| `tiles/`       | The tile finder from dolphin. No coordinate code of its own                        |
| `training/`    | The tile reader's recipe, and its baseline scenes                                  |
| `files/`       | The reader, `reader.tflite`, and what it was built from                            |
| `agenttools/`  | Offline tests, the fake board and the app on it, `try-arm.py`. Removed at release  |

## What changed in the library since elephant

- **`MoveType.PRECISE` is built.** The servos have a deadband: they ignore a target a few pulses away and stop short
  of a farther one, 1-3 mm. PRECISE moves slowly, reads the servos, and sends target + shortfall until each servo is
  within a pulse of its target: 1 mm steps went from 1.4 mm median error to 0.5 on the owner's arm, for ~1.2 s more.
- **Heights are true z, on every call.** The firmware's own z (raw z) puts the cup higher than it says near the base
  and lower at full reach: a flat desk read 50 at 100 mm out and 42 at 260. Every MaxArm is built the same, so the
  library corrects every target before it is sent and every reading after it is read, in `protocol.py`:
  `true z = raw z + 0.05 x (reach - 207)`, reach = sqrt(x^2 + y^2). x and y are untouched; at 207 mm out, where a
  desk reads 44, the two agree. `MAXARM_TRACE` logs moves and reads as `(x, y, 44.0 [raw 49.0])`.
  `raw.debug_use_raw(True)` (`maxarm/raw.py`) turns it off process-wide, for testing the firmware only.
  `desk.json` is a plain plane again; one carrying the earlier elbow offset (`joint_offsets_deg`) is not used.
- **Reach is 95 to 280 mm** (was up to 270); the firmware's own ends at 286. Nearer than 95 the arm folds too far to
  be guided and the desk reads ~1.4 mm off the true z line. Both radii limit where a move ends, not the way there.
- **A default desk.** Without `desk.json`, desk space is arm space with z 44 taken off: the desk under an ideally
  placed arm, in true z (`spaces.DEFAULT_DESK_Z_MM`, `spaces.default_desk_to_arm()`). Desk coordinates no longer need
  a calibration. `Spaces.is_desk_known` (saved or default) and `is_desk_saved` replace `is_desk_calibrated`;
  `ignored_desk` says why a `desk.json` was not used. Only `config_dir=None` means no desk at all.
- **Positions are read from the servos' pulses**, with the firmware's kinematics on the host, in floats. The board's
  own read rounds to whole millimetres, and truncates negative values.
- **`arm.use_calibration(config_dir)`** reads the calibration files afresh; `None` means none at all. A calibration
  app needs it: it measures the desk with no desk calibration in the way, then tries out the file it has just
  written, on the same connection. `MaxArm(config_dir=None)` and `Spaces(None)` likewise.
- **A move of any size is sent.** Moves under 0.5 mm were skipped without a word, which hid what the desk page's
  tenth-of-a-millimetre slider (since removed) was for, and still would a desk move by a tenth. Now only a move the
  command cannot carry (two decimals) is skipped.
- Smaller: a cup turn of exactly 90 degrees is no longer refused for float error; `connect()` starts from the reset
  pose, where the board sends the arm.
- Converting between spaces needs no new call: `Spaces(config_dir).convert(point, Space.ARM, Space.DESK)`.

## The calls

```python
MaxArm(device="/dev/ttyUSB0", config_dir=Path("config"))   # both optional
arm.connect(); arm.disconnect()                           # or: with MaxArm() as arm:

arm.use_coordinates(Space.DESK)        # ARM (the default), DESK, CUSTOM, CAMERA
arm.use_calibration(config_dir)        # read config/ afresh; None for no calibration
arm.set_clearance(14.0)                # mm above the desk

arm.move(x, y, z)                      # fast: 200 mm/s
arm.move(x, y, z, MoveType.SLOW)       # 10 mm/s: nothing heavy gets yanked
arm.move(x, y, z, MoveType.PRECISE)    # slow, then corrected until it lands: ~0.5 mm, ~1.2 s more
arm.move(x, y, z, angle=30.0)          # and turn the cup -- or the piece it holds
arm.grab(x, y, height, angle=None)     # pick up the piece whose top face is there
arm.drop(x, y, height, angle=None)     # set the held piece down there, facing `angle`
arm.home()                             # the reset pose, by way of a move
arm.set_suction(True)                  # by hand, for a job grab() and drop() do not fit

arm.can_reach(x, y, z)                 # arithmetic: works before connect()
arm.get_position()                     # where the servos say the arm is
arm.stop()                             # safe from a signal handler
arm.stop_on_ctrl_c()                   # Ctrl-C holds the arm for the rest of the run
```

Every motion returns a `MoveResult`: `DONE`, `REFUSED` (nothing moved; `detail` says why) or `STOPPED`.
It is falsy unless it is `DONE`.
`move()`, `grab()`, `drop()` and `home()` all go through one internal move, which plans the whole path
and checks every limit before anything is sent.

## Coordinate spaces

Every space is defined by how it maps onto **desk space**, which is where the library plans:
millimetres, with z the height above the desk and 0 the cup snug on its surface.

| space    | a point is          | onto desk space by                        | file          |
| -------- | ------------------- | ----------------------------------------- | ------------- |
| `ARM`    | the board's x, y, z | the inverse of `desk_to_arm`, a 4x4       | `desk.json`   |
| `DESK`   | x, y, height        | nothing to do                             | `desk.json`   |
| `CUSTOM` | your own x, y, z    | `custom_to_desk`, a 4x4                   | `custom.json` |
| `CAMERA` | px, py, height      | `desk_to_pixel`, a 3x4, solved per height | `camera.json` |

In camera space the height picks the point along the pixel's ray: give the pixel where the thing *at that height* is
seen. A tile's top face is `(px, py, 4.0)`; the same pixel at height 0 is a different point of the desk.

Without `desk.json` the library does not know where the desk is: only `ARM` is available, heights are the arm's own z,
and the clearance and desk-floor checks do nothing.
`maxarm.Spaces(config_dir).convert(point, Space.CAMERA, Space.ARM)` converts between any two, without an arm.

**`desk.json` is now a plane**, written by the desk page: desk x and y are the arm's own, desk z is the height above
the plane through the touch points. `config/` starts empty: no `desk.json` until the first SAVE, and no `camera.json`
until the camera calibration exists, so `main.py` waits for that (`config/README.md`).

## What a move does

**Fast and slow are one command per leg, waited out.** Nothing is read back while the arm moves, and nothing is
corrected. A leg is usually the whole move; there are more only when the move has to climb over the clearance, go
round the arm-base limit, or is too long to stay level (see below).

**PRECISE is slow, then corrected.** After the slow legs it reads the servos (a median of five, settled) and, while
any servo is more than a pulse from the pulse the firmware sends for the target, sends the target plus the error: at
most three rounds and 3 mm per axis, every aim checked against the limits, and it stops as soon as a round gains
nothing, so a cup resting on something is not pushed on and on. The result then says the arm is at the target.

**The cup turns with the move.** `angle` is a direction in your space, degrees from its +x towards its +y. The turn is
sent in the same round trip as the leg that crosses the desk, and the move waits for whichever finishes last: the cup
swings 180 degrees in 1.2 s. The cup turns 90 degrees either way from centre; a turn beyond that is refused before
anything moves.

**grab()** lets go of anything held, then: slowly up to the travel height if below it, fast across while the cup
centres itself, slowly down onto the piece, suction on and waited for (1.5 s), slowly back up. The travel height is
`clearance + height + 5 mm`, so the piece clears the clearance by 5 mm. Its `angle` says which way the piece faces.

**drop()** is the same in reverse: the piece is turned to `angle` on the way across, and the release is waited out
(3 s, longer than a grip: the board vents the cup for a second and then shuts the valve) before the slow lift. The
travel height allows for the piece hanging under the cup. `angle=None` leaves the cup alone, so the piece turns with
the arm.

## Limits

All of them are applied to every move, before anything is sent. `can_reach()` asks the same questions.

- **Reach**: a move ends 95 to 280 mm from the base axis, and nothing goes into the back 120 degrees. The two radii
  are for where a move ends: a path may pass inside 95 (nothing goes round it, unlike the arm-base limit), and a
  move that only climbs or stays put is let out from anywhere, an arm stopped half-way say.
- **Firmware**: what the board would refuse, or silently clamp, as `geometry.py` predicts it.
- **Arm-base limit**: a box round the base, |x| and |y| up to 70 mm, from the desk to 200 mm up. A move whose
  straight line crosses it goes by a front corner instead, at half-way height.
- **Clearance**, with a calibrated desk: nothing travels sideways below `clearance + carried + 5 mm`.
- **Desk floor**, with a calibrated desk: nothing more than 5 mm into it.
- **Path dip**: the kinematics predict that a long, low move in or out dips below its ends; such a leg is halved
  until the predicted dip is 5 mm or less. A precaution: the dip has never been measured on the arm.

The arm does not move in straight lines -- it interpolates its servos -- but straight lines are still the right thing
to check: sideways, by the model its path keeps between the two ends' distances from the base, bowing outward.
Downward the model predicts a dip on long moves in or out, which the last row guards against; nothing on the arm has
been seen to sag (work/STATUS-falcon.md §5.4). See [`maxarm/path.py`](maxarm/path.py).

## What you can tune

Seven values answer to the environment, each one `MAXARM_` and the constant's name in the code:

| variable                      | default        | what it changes                                 |
| ----------------------------- | -------------- | ----------------------------------------------- |
| `MAXARM_DEVICE`               | `/dev/ttyUSB0` | the serial port                                 |
| `MAXARM_MOVE_FAST_MM_S`       | 200            | fast moves                                      |
| `MAXARM_MOVE_SLOW_MM_S`       | 10             | slow moves, and grab's and drop's ups and downs |
| `MAXARM_SUCTION_GRAB_MS`      | 1500           | the wait for a grip                             |
| `MAXARM_SUCTION_RELEASE_MS`   | 3000           | the wait for a release                          |
| `MAXARM_BASE_LIMIT_HALF_MM`   | 70             | the arm-base limit's half-width                 |
| `MAXARM_BASE_LIMIT_HEIGHT_MM` | 200            | its height                                      |

`MAXARM_TRACE=1` prints every expression sent to the board, with its answer and timing, on stderr.
Everything else is a constant in the module that uses it, with where it came from written above it.

## Running it

```
../.venv/bin/python calibration/calibration.py     # the calibration app, on the arm
../.venv/bin/python agenttools/serve-fake.py       # the same, on a fake arm
./main.py                                          # camera snapshot, then pick up the tiles
./main.py training/tiles/baseline/tiles-1.png      # a saved frame instead
../.venv/bin/python agenttools/try-arm.py          # the motion on the real arm, stage by stage, no camera
../.venv/bin/python agenttools/try-arm.py --dry-run
../.venv/bin/python agenttools/run-tests.py        # every offline check, a few seconds, no board
../.venv/bin/python training/tiles/check-reader.py # the tile finder on its baseline scenes
```

## Three things to expect on real hardware

1. **Connecting moves the arm.** Opening the port resets the board, which homes the arm and centres the cup:
   about 4 s, sometimes 15.
2. **Fast and slow moves land 1-3 mm short**: the servos' deadband. PRECISE lands within a pulse, about 0.5 mm.
3. **An arm that accepts commands and does not move** was seen once. `connect()` checks the servo supply; a move
   cannot tell, because it reads nothing back. Power-cycle the arm at the barrel jack.
