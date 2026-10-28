# impala — hyena's tiles app, its kinks ironed out

Spec: [`work/PILOT-impala.md`](../work/PILOT-impala.md).
State of play: [`work/STATUS-impala.md`](../work/STATUS-impala.md).
The Tiles page came with hyena ([`work/PILOT-hyena.md`](../work/PILOT-hyena.md)); this pilot starts with the UI
tweaks and goes on to the errors that persist in the calibration (PILOT-impala).

The calibration web app's Tiles page. LOCATE TILES parks the arm, takes the camera's full-size
snapshot and finds the new 25 x 25 x 2 mm letter tiles in it; each tile's hand-printed letter is read, and which way
up it lies; each is placed in the work areas -- ready to pick up, too close to another, outside the pickup area, or
out of reach -- and shown so over the camera's picture, on the view from above and in a table. A click on a tile
brings the cup over its centre. PLACE carries the tiles with the letters typed, one by one, to an arc in the dropoff
area, each set down facing away from the arm. What finds the tiles, reads, places and carries them is the final
app's own library, `applib/`, which knows nothing of the web app.

```
../.venv/bin/python calibration/calibration.py     # the calibration app: open http://localhost:8080, the Tiles tab
```

```python
camera = Camera.load(Path("config/camera.json"))
arm.use_coordinates(Space.DESK)

tiles = find_tiles(frame, camera)                       # the camera's full-size frame, as the camera page took it
letters = read_letters(frame, camera, tiles)
places = place_tiles(tiles, arm.can_reach)
for tile, letter, place in zip(tiles, letters, places):
    print(tile.x, tile.y, place.status,                 # READY, TOO_CLOSE, ELSEWHERE or OUT_OF_REACH
          letter.letter if letter.is_read else "?",     # "" for a blank tile
          letter.upright)                               # where the letter's top points, desk degrees

arm.set_clearance(word.CLEARANCE_MM)
for step in word.plan_word("TEAM", tiles, letters, places, arm.can_reach):   # raises WordRefused, saying why
    arm.carry(step.start, step.end, angle=step.upright, facing=step.facing)
```

| directory      | holds                                                                                       |
| -------------- | ------------------------------------------------------------------------------------------- |
| `maxarm/`      | The arm's library                                                                           |
| `applib/`      | The final app's own libraries: `tiles/` finds the tiles in a frame, `letters/` reads them,  |
|                | `areas.py` places them, `word.py` plans a word of them                                      |
| `files/`       | The letter reader, `letters.tflite`, and what it was built from, `letters.json`             |
| `training/`    | `letters/`: the reader's recipe, drawn hand-printed tiles and the training; nothing fetched |
| `calibration/` | The calibration web app: the desk, camera, Fine Tune and Tiles pages; guide in `README.md`  |
| `config/`      | The calibration files and the camera page's frames; per rig, so not in git                  |
| `agenttools/`  | Offline tests, the fake board and camera, drawn scenes, the owner's stills. Removed at release |

## What changed since hyena

- **What came across.** hyena's library, `applib/`, calibration app, `config/`, training recipe and offline kit,
  and its stills of the owner's tiles. Not its one-off investigation of the tiles near the camera
  (`agenttools/tile-issue/`), which stays in hyena.
- **The live view is the camera's full size** (`calibration/camera/feed.py`): MJPG at 2592 x 1944 on the document
  camera, about 4 frames a second, the camera's own JPEGs sent on as they come. A snapshot is five fresh frames of
  it averaged, 1.6 s, where it was the camera reopened in YUYV, about 11 s. FIND RED CUBE and ARM TO DESK work on
  the full-size picture. `capture.py` is gone (STATUS §1).
- **Round the arm's base, a corner as low as the arm needs** (`maxarm/path.py`): a move that goes by a front corner
  of the arm-base limit used to pass it at half-way height, which servo 3 cannot hold close to the base and high;
  76 of 1328 desk points in reach were refused from the park pose, the red cube at (104.2, 3.9) among them. Now the
  corner comes down as little as it must, not under the clearance (STATUS §2).
- **The notch is 42**, 2 mm over a cube's top, on the camera and Fine Tune pages; ARM TO DESK comes down to 2 mm
  over the desk.
- **Fine Tune**: a grid click comes down to the notch; RECORD POINT, beside the
  arrows, sees the cube in the snapshot; CLICK THE DESK over the camera, FIND RED CUBE apart, a helper; the live view
  no longer draws where the servos say the cup is (STATUS §3).
- **ARM TO DESK**: the ring under the pointer is blue; the bar over the picture keeps its size, so the picture no
  longer moves as the arm does; the red cube found is cleared from the picture on entering or clicking.
- **Tiles**: LOCATE TILES offers letters only until some are typed; after PLACE the arm parks and the tiles are found
  and read again, where the tiles carried were drawn grey (STATUS §4).
- **Every page**: the cubes' wireframes are thinner and fainter; the height slider's range takes in the arm's height
  when it is above or below it, as after RESET.
- **The camera calibration places the arm by a turn and a shift** (`calibration/camera/fit.py`, `refine.py`): the
  camera is still fitted to the cubes' shapes; the arm's points are then laid on the cubes as seen, with no stretch,
  at CALIBRATE and at SAVE CALIBRATION. A stretch through three cubes fitted each cube's own error and carried it
  across the desk: the first calibration was up to 10 mm off on the left, and the reach arcs' centre about 4 mm off
  (STATUS §6). The camera page's line says how far each cube stands from its arm point.
- **FIND RED CUBE** on Fine Tune: once the arm is down over the cube, its servos are read; a dot on the cube's top
  where it was sent, the cup's rim, yellow, where the servos say it is, and the miss, X and Y, by the cube's x, y.
- **DESK GRID** on Fine Tune: desk space every 10 mm drawn on the desk over the live view, off unless turned on, the
  lines through the arm's turning axis stronger: grid paper laid along it shows whether the camera's millimetres
  match the paper's all over it.

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
arm.carry(start, end, angle=None, facing=None)   # both, (x, y, height) each: any turn, checked first
arm.home()                             # the reset pose, by way of a move
arm.set_suction(True)                  # by hand, for a job grab() and drop() do not fit

arm.can_reach(x, y, z)                 # arithmetic: works before connect()
arm.get_position()                     # where the servos say the arm is
arm.stop()                             # safe from a signal handler
arm.stop_on_ctrl_c()                   # Ctrl-C holds the arm for the rest of the run
```

Every motion returns a `MoveResult`: `DONE`, `REFUSED` (nothing moved; `detail` says why) or `STOPPED`.
It is falsy unless it is `DONE`.
`move()`, `grab()`, `drop()`, `carry()` and `home()` all go through one internal move, which plans the whole path
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
seen. A tile's top face is `(px, py, 2.0)`; the same pixel at height 0 is a different point of the desk.

Without `desk.json` the default desk stands in: arm z 44 and level, the desk under an ideally placed arm. Only
`config_dir=None` knows no desk at all: then only `ARM` is available, and the clearance and desk-floor checks do
nothing.
`maxarm.Spaces(config_dir).convert(point, Space.CAMERA, Space.ARM)` converts between any two, without an arm.

**`desk.json` is now a plane**, written by the desk page: desk x and y are the arm's own, desk z is the height above
the plane through the touch points. `config/` starts empty: no `desk.json` until the first SAVE, and no `camera.json`
until the camera calibration exists, so the Fine Tune and Tiles pages wait for that (`config/README.md`).

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
centres itself, slowly down onto the piece and corrected there as PRECISE is, suction on and waited for (1.5 s),
slowly back up. The travel height is `clearance + height + 5 mm`, so the piece clears the clearance by 5 mm. Its
`angle` says which way the piece faces.

**drop()** is the same in reverse: the piece is turned to `angle` on the way across, and the release is waited out
(3 s, longer than a grip: the board vents the cup for a second and then shuts the valve) before the slow lift. The
travel height allows for the piece hanging under the cup. `angle=None` leaves the cup alone, so the piece turns with
the arm. The cup turns 90 degrees either way from where `grab()` centred it; more is refused, the piece still held.

**carry()** is the two together, and checks the drop before the grip, so a refusal means nothing moved. Knowing
both ends, it can turn a piece any way: a turn bigger than the cup has from centre is shared, the cup turning half
of it the other way on its way over the piece, the other half after the lift.

## Limits

All of them are applied to every move, before anything is sent. `can_reach()` asks the same questions.

- **Reach**: a move ends 95 to 270 mm from the base axis, and nothing goes into the back 120 degrees. The two radii
  are for where a move ends: a path may pass inside 95 (nothing goes round it, unlike the arm-base limit), and a
  move that only climbs or stays put is let out from anywhere, an arm stopped half-way say.
- **Firmware**: what the board would refuse, or silently clamp, as `geometry.py` predicts it.
- **Arm-base limit**: a box round the base, |x| and |y| up to 70 mm, from the desk to 200 mm up. A move whose
  straight line crosses it goes by a front corner instead, at half-way height, or as little lower as the arm needs
  to hold it there, not under the clearance.
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
../.venv/bin/python agenttools/serve-fake.py --frame agenttools/tile-shots/still-1.png \
    --camera-json agenttools/tile-shots/still-camera.json  # a fake arm, the Tiles page on the owner's tiles
../.venv/bin/python agenttools/try-tiles.py FRAME.png        # the tile finder and the letters on a frame, drawn
../.venv/bin/python training/letters/train.py      # the letter reader afresh, into files/; about 20 minutes
../.venv/bin/python agenttools/refit-frame.py      # camera.json fitted afresh to the frame CALIBRATE kept
../.venv/bin/python agenttools/run-tests.py        # every offline check, about 50 s, no board
```

## Three things to expect on real hardware

1. **Connecting moves the arm.** Opening the port resets the board, which homes the arm and centres the cup:
   about 4 s, sometimes 15.
2. **Fast and slow moves land 1-3 mm short**: the servos' deadband. PRECISE lands within a pulse, about 0.5 mm.
3. **An arm that accepts commands and does not move** was seen once. `connect()` checks the servo supply; a move
   cannot tell, because it reads nothing back. Power-cycle the arm at the barrel jack.
