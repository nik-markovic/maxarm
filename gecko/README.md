# gecko — the camera calibration's page, on falcon's library

Spec: [`work/PILOT-gecko.md`](../work/PILOT-gecko.md).
State of play: [`work/STATUS-gecko.md`](../work/STATUS-gecko.md).

The pilot adds the calibration web app's second page: the camera. The arm is brought over each of the three cubes,
the cubes are centred under the cup and recorded, and CALIBRATE parks the arm, takes a frame and fits the camera to
the cubes in it: `config/camera.json`. The library is falcon's. Move, grab and drop, in whichever coordinates you work in; the library owns every coordinate conversion,
and an app only finds things and passes their coordinates.

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
It needs `config/camera.json`, which the camera page's CALIBRATE writes.

| directory      | holds                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------ |
| `maxarm/`      | The library                                                                                |
| `calibration/` | The calibration web app: `calibration.py`, the desk and camera pages; guide in `README.md` |
| `config/`      | The calibration files and the camera page's frames; per rig, so not in git                 |
| `tiles/`       | The tile finder from dolphin. No coordinate code of its own                                |
| `training/`    | The tile reader's recipe, and its baseline scenes                                          |
| `files/`       | The reader, `reader.tflite`, and what it was built from                                    |
| `agenttools/`  | Offline tests, the fake board and camera, the app on them. Removed at release              |

## What changed since falcon

- **The camera page** (`calibration/camera/`, guide in `calibration/README.md`): the arm over the cubes in desk
  space, a point per cube in `config/camera-points.json`, the live view, CALIBRATE and VALIDATE CALIBRATION.
- **The camera calibration**: CALIBRATE fits a square-pixel camera to the three cubes' edges and takes in the arm's
  own x/y error through where it recorded them (`calibration/camera/fit.py`); `camera.json` holds `desk_to_pixel`
  as the library reads it, the frame size, and the fit behind it. VALIDATE draws it on a new frame and says whether
  the camera has moved. The check image draws each cube whole: the edges its faces hide fainter, and an X across
  its bottom, crossing under its centre.
- **The tests page** (`calibration/tests/`), there once `camera.json` is: FIND RED CUBE finds the red cube in the
  live view, through the calibration, and brings the cup down 4 mm over its top; ARM TO DESK sends the arm to the
  point of the desk clicked in the live view, shown full size, down to 4 mm over it. The live view carries the
  calibration's marks: the reach, the arm, the cube found, the point clicked with the cup's rim round it.
- **The red cube's points refine the calibration** (the tests page, its tab now Fine Tune): the red cube set down in place after place,
  FIND RED CUBE brings the cup over it, the cube is centred under the cup, and RECORD POINT keeps where the arm says
  the cup is and where the camera, the arm parked, sees the cube (`config/camera-refine-points.json`). SAVE
  CALIBRATION fits the arm's x/y error again to all of them and the three cubes, the camera kept
  (`calibration/camera/refine.py`). The camera page suggests the cubes at x ±140 now, inside the reach.
- **RESET lifts over the cubes.** After its slow lift off the desk it goes straight up, fast, to arm z 104 -- 60 mm
  over the default desk -- before heading home, so it clears whatever the camera page put on the desk.
- **The camera is found**, not assumed to be `/dev/video1`: the owner's by its USB port if plugged in, else the first
  that is not virtual (`calibration/camera/devices.py`). `capture()` and `main.py` use it too.
- **The library is unchanged** but for three things. The reach is 270 mm again (`path.REACH_MAX_MM`), as before
  falcon: at 280 the arm stretches too far and misses more (the owner). A settled read takes its wait and its count
  of reads from the module's constants when it runs, not when it was defined: the offline checks could not shorten
  the wait before, so the desk page's took 87 s; all of them now take 10. And `Camera` knows the frame size it was
  calibrated at, if `camera.json` says (`frame_size`), and `check_frame()` refuses a frame of another size, which
  `find_tiles()` now calls first: at another size every pixel means another place.
- The library's earlier changes, since elephant: [`falcon/README.md`](../falcon/README.md).

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

Without `desk.json` the default desk stands in: arm z 44 and level, the desk under an ideally placed arm. Only
`config_dir=None` knows no desk at all: then only `ARM` is available, and the clearance and desk-floor checks do
nothing.
`maxarm.Spaces(config_dir).convert(point, Space.CAMERA, Space.ARM)` converts between any two, without an arm.

**`desk.json` is now a plane**, written by the desk page: desk x and y are the arm's own, desk z is the height above
the plane through the touch points. `config/` starts empty: no `desk.json` until the first SAVE, and no `camera.json`
until the camera calibration exists, so `main.py` waits for that (`config/README.md`).

## What a move does

**Fast and slow are one command per leg, waited out.** Nothing is read back while the arm moves, and nothing is
corrected. A leg is usually the whole move; there are more only when the move has to climb over the clearance or go
round the arm-base limit (see below).

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

- **Reach**: a move ends 95 to 270 mm from the base axis, and nothing goes into the back 120 degrees. The two radii
  are for where a move ends: a path may pass inside 95 (nothing goes round it, unlike the arm-base limit), and a
  move that only climbs or stays put is let out from anywhere, an arm stopped half-way say.
- **Firmware**: what the board would refuse, or silently clamp, as `geometry.py` predicts it.
- **Arm-base limit**: a box round the base, |x| and |y| up to 70 mm, from the desk to 200 mm up. A move whose
  straight line crosses it goes by a front corner instead, at half-way height.
- **Clearance**, with a calibrated desk: nothing travels sideways below `clearance + carried + 5 mm`.
- **Desk floor**, with a calibrated desk: nothing more than 5 mm into it.

See [`maxarm/path.py`](maxarm/path.py).

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
../.venv/bin/python agenttools/serve-fake.py       # the same, on a fake arm, the camera three drawn cubes
../.venv/bin/python agenttools/serve-fake.py --real-camera   # a fake arm, the real camera
../.venv/bin/python agenttools/refit-frame.py      # camera.json fitted afresh to the frame CALIBRATE kept
./main.py                                          # camera snapshot, then pick up the tiles
./main.py training/tiles/baseline/tiles-1.png      # a saved frame instead
../.venv/bin/python agenttools/run-tests.py        # every offline check, about 10 s, no board
../.venv/bin/python training/tiles/check-reader.py # the tile finder on its baseline scenes
```

## Three things to expect on real hardware

1. **Connecting moves the arm.** Opening the port resets the board, which homes the arm and centres the cup:
   about 4 s, sometimes 15.
2. **Fast and slow moves land 1-3 mm short**: the servos' deadband. PRECISE lands within a pulse, about 0.5 mm.
3. **An arm that accepts commands and does not move** was seen once. `connect()` checks the servo supply; a move
   cannot tell, because it reads nothing back. Power-cycle the arm at the barrel jack.
