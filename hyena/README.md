# hyena — the 25 mm tiles, found and placed, on gecko's library and calibration

Spec: [`work/PILOT-hyena.md`](../work/PILOT-hyena.md).
State of play: [`work/STATUS-hyena.md`](../work/STATUS-hyena.md).

The pilot adds the calibration web app's Tiles page. LOCATE TILES parks the arm, takes the camera's full-size
snapshot and finds the new 25 x 25 x 2 mm letter tiles in it; each is placed in the work areas -- ready to pick up,
too close to another, outside the pickup area, or out of reach -- and shown so over the camera's picture, on the view
from above and in a table. A click on a tile brings the cup over its centre. What finds the tiles and places them is
the final app's own library, `applib/`, which knows nothing of the web app.

```
../.venv/bin/python calibration/calibration.py     # the calibration app: open http://localhost:8080, the Tiles tab
```

```python
camera = Camera.load(Path("config/camera.json"))
arm.use_coordinates(Space.DESK)

tiles = find_tiles(frame, camera)                       # the camera's full-size frame, as the camera page took it
for tile, place in zip(tiles, place_tiles(tiles, arm.can_reach)):
    print(tile.x, tile.y, tile.turn, place.status)      # READY, TOO_CLOSE, ELSEWHERE or OUT_OF_REACH
```

| directory      | holds                                                                                       |
| -------------- | ------------------------------------------------------------------------------------------- |
| `maxarm/`      | The arm's library                                                                           |
| `applib/`      | The final app's own libraries: `tiles/` finds the tiles in a frame, `areas.py` places them  |
| `calibration/` | The calibration web app: the desk, camera, Fine Tune and Tiles pages; guide in `README.md`  |
| `config/`      | The calibration files and the camera page's frames; per rig, so not in git                  |
| `agenttools/`  | Offline tests, the fake board and camera, drawn scenes, the owner's stills. Removed at release |

## What changed since gecko

- **What came across.** gecko's library, calibration app and `config/`, and its offline test kit. The Scrabble
  tiles' finder, reader and training (`tiles/`, `training/`, `files/reader.*`) and `main.py`, which picked them up,
  stay in gecko: the new tiles are another piece. So do gecko's measurement scripts. The fixed calibration the
  library's checks read moved from `training/tiles/baseline/` to `agenttools/baseline/`.
- **The tile finder** (`applib/tiles/`): the frame resampled onto the plane of the tiles' top faces, 2 mm up; every
  colour edge votes for the centre of a 25 mm square it could be a side of; a square fitted to the edges round each
  place most voted for; kept if each of its sides stands out from what lies just beyond it. No colour or brightness
  is assumed of the tiles or the desk, and tiles may touch. The letters are not read yet.
- **The work areas** (`applib/areas.py`): pickup the outer two thirds of the reach, left of the arm's centre line
  (x < 0); dropoff the inner third; tile centres 5 mm inside the arm's reach arcs. Each tile ready, too close (under
  2 mm from another), outside the pickup area, or out of reach, by the library's `can_reach()`.
- **The Tiles page** (`calibration/tiles/`, guide in `calibration/README.md`), there once `camera.json` is: LOCATE
  TILES in the camera's heading, where it stays in sight; the live view or the snapshot, the areas and the tiles
  drawn over it, full size on a click; the view from above; the table. A tile clicked anywhere: the cup over its
  centre at 14 mm, down to 6, 4 mm over its top.
- **Shared by the pages with a camera**: desk space to the picture and back (`common/shares.js`, from `tests/`), the
  marks' drawing (`common/marks.js`) and the picture full size (`common/fullview.js`). Fine Tune's own marks are
  built on them; it works as before.
- **The library is unchanged.**

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
../.venv/bin/python agenttools/serve-fake.py --frame agenttools/tile-shots/still-1.png \
    --camera-json agenttools/tile-shots/still-camera.json  # a fake arm, the Tiles page on the owner's tiles
../.venv/bin/python agenttools/try-tiles.py FRAME.png        # the tile finder on a frame, drawn
../.venv/bin/python agenttools/refit-frame.py      # camera.json fitted afresh to the frame CALIBRATE kept
../.venv/bin/python agenttools/run-tests.py        # every offline check, about 50 s, no board
```

## Three things to expect on real hardware

1. **Connecting moves the arm.** Opening the port resets the board, which homes the arm and centres the cup:
   about 4 s, sometimes 15.
2. **Fast and slow moves land 1-3 mm short**: the servos' deadband. PRECISE lands within a pulse, about 0.5 mm.
3. **An arm that accepts commands and does not move** was seen once. `connect()` checks the servo supply; a move
   cannot tell, because it reads nothing back. Power-cycle the arm at the barrel jack.
