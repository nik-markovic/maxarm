# maxarm/ — the arm's library

Moves, grab and drop, in whichever coordinates you work in; every limit checked by arithmetic before anything is
sent. Both apps drive the arm through it (`applib/motion.py`). Carried from impala unchanged, but for where it finds
`config/`: the pilot's, beside `src/`.

## The calls

```python
MaxArm(device="/dev/ttyUSB0", config_dir=Path("config"))   # both optional
arm.connect(); arm.disconnect()                           # or: with MaxArm() as arm:

arm.use_coordinates(Space.DESK)        # ARM (the default), DESK, CUSTOM, CAMERA
arm.use_calibration(config_dir)        # read config/ afresh; None for no calibration
arm.set_clearance(14.0)                # mm above the desk

arm.move(x, y, z)                      # fast: 200 mm/s
arm.move(x, y, z, MoveType.SLOW)       # 10 mm/s: nothing heavy gets yanked
arm.move(x, y, z, MoveType.PRECISE)    # slow, then read until steady: where it is, not corrected
arm.move(x, y, z, angle=30.0)          # and turn the cup -- or the piece it holds
arm.down_to(x, y, height)              # the cup down to exactly there, precisely, and stay: suction untouched
arm.grab(x, y, height, angle=None)     # the cup down to exactly there, suction on, back up
arm.drop(x, y, height, angle=None)     # set the held piece down there, facing `angle`
arm.carry(start, end, angle=None, facing=None)   # both, (x, y, height) each: any turn, checked first
arm.carry(start, end, retry_deeper_mm=1.5)        # a grip that does not seal tried again, 0.5 mm deeper each time
arm.carry(start, end, retry_deeper_mm=1.5, retry_step_mm=0.25)   # the same, 0.25 mm at a time
arm.home()                             # the reset pose, by way of a move
arm.set_suction(True)                  # by hand, for a job grab() and drop() do not fit

arm.can_reach(x, y, z)                 # arithmetic: works before connect()
arm.get_position()                     # where the servos say the arm is: their reads' mean once steady
arm.get_seal()                         # SEALED, LOOSE or UNCERTAIN when the pump last went on; None once moved
arm.stop()                             # safe from a signal handler
arm.stop_on_ctrl_c()                   # Ctrl-C holds the arm for the rest of the run
```

Every motion returns a `MoveResult`: `DONE`, `REFUSED` (nothing moved; `detail` says why), `STOPPED`, or for
`grab()` and `carry()` with `retry_deeper_mm`, `NOT_GRIPPED` (no seal down to the limit; let go and lifted).
It is falsy unless it is `DONE`. From `grab()` and `carry()` its `deeper_mm` says how much deeper than asked the cup
went for the grip.
`move()`, `down_to()`, `grab()`, `drop()`, `carry()` and `home()` all go through one internal move, which plans the whole path
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

**PRECISE is slow, then read.** After the slow legs a precise move waits until the servos are steady; the result
then says the arm is at the target, and `get_position()` gives the servos' mean. It does not correct: the servos stop
1-3 mm short. The owner's nudge, tried in it on 30 Oct and taken out on 1 Nov (it went back and forth too much), is
kept in `arm-nudge.py` (`work/STATUS-impala.md` §19.3-19.4).

**Where the arm is** is the mean of the servos' reads, which flicker by a pulse or more for as long as they are read:
from 1 s after a move's travel, the mean of the last half second against the half second before it, every servo
within 0.35 pulse -- then the mean of that second; at most 3 s after the travel; from 2 s after it, half a second of
reads (`work/STATUS-impala.md` §19).

**The cup turns with the move.** `angle` is a direction in your space, degrees from its +x towards its +y. The turn is
sent in the same round trip as the leg that crosses the desk, and the move waits for whichever finishes last: the cup
swings 180 degrees in 1.2 s. The cup turns 90 degrees either way from centre; a turn beyond that is refused before
anything moves.

**down_to()** moves the cup down to the point given, exactly that height and never lower, as precisely as the arm
lands (`landing.py`; the measurements behind
it are in work/MAXARM_MECHANICS.md): slowly up to the travel height if below it, fast across to 20 base pulses beside
the point (the side of higher pulses), the base turned onto it aimed past by what that turn falls short, then one slow command straight down,
servos 2 and 3 aimed past by what a descent falls short; read there as PRECISE is. If the base settled more than 0.75 mm to the side at the cup, and a whole pulse nearer the target was there to be had (the base stops on whole pulses, so nothing better is asked for), it lifts and lands again, once, from the other side: other gear faces, not the same miss again. It stays down, suction as it was:
`get_position()` says where it came down. What it keeps as sent is the point given, never the aimed-past command.
It is a move only: the suction, and how deep a grip should go, are not its business.

**grab()** lets go of anything held, then: `down_to()` the height given, exactly, while the cup centres itself, suction on and
waited for (1.5 s), slowly back up. The travel height is `clearance + height + 5 mm`, so the piece clears the clearance by 5 mm. Its
`angle` says which way the piece faces.

`retry_deeper_mm` (`grab()` and `carry()`, 0 to 2 mm, more is a ValueError; 0 by default, the grip not checked): a
grip whose seal reads loose or uncertain is let go, lifted, and landed again `retry_step_mm` deeper (0.5 mm unless
given; 0.1 to 2 mm, else a ValueError), down to that much under the height given. The result's `deeper_mm` is the depth that sealed, or the last tried with `NOT_GRIPPED`.
The seal it goes by is told right only against what does not move -- the desk, a piece pressed onto it; it does not say
whether a light piece came along (see the seal, below).

**drop()** is the same in reverse, `down_to()` the place: the piece is turned to `angle` on the way across, and the release is waited out
(3 s, longer than a grip: the board vents the cup for a second and then shuts the valve) before the slow lift. The
travel height allows for the piece hanging under the cup. `angle=None` leaves the cup alone, so the piece turns with
the arm. The cup turns 90 degrees either way from where `grab()` centred it; more is refused, the piece still held.

**carry()** is the two together, and checks the drop before the grip, so a refusal means nothing moved. Knowing
both ends, it can turn a piece any way: a turn bigger than the cup has from centre is shared, the cup turning half
of it the other way on its way over the piece, the other half after the lift. And it allows for the grip: the piece hangs
off the cup by however far the grip missed its centre -- where the cup came down, read before the suction, less
where it was sent -- and turns with the cup. So the cup comes down that miss, turned as the piece has turned, off
`end`: the piece's centre lands on it, give or take the drop's own miss. A plain `drop()` goes where it is sent.

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

See [`path.py`](path.py).

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

**The seal** is told by the supply the servos and the pump run off, which every position read samples too, in the
same round trip (`seal.py`). The pump drawing air pulls the supply down; sealed, it hardly does. So the supply where
the arm stands with the pump off -- the reads that end a PRECISE move, or the end of a release -- is set against
the reads in the grip's 1.5 s, from 0.75 s on: the pump's dip. The servos pull on the supply too, so a move ends
the verdict; and a seal pulls the cup down (it is a bellows), which servos holding it short of a target above
strain against, so a cup pulled down 0.5 mm is sealed whatever the dip. A board that cannot read the supply gives
UNKNOWN; nothing else changes. It holds for a cup landed on what does not move: the desk, or a piece pressed onto it. A
light piece is pulled up into the cup instead, and the supply does not tell a piece on the cup from an empty one,
there or after a lift: whether something was picked up is the caller's to check (a camera, say).

`MAXARM_TRACE=1` prints every expression sent to the board, with its answer and timing, on stderr.
Everything else is a constant in the module that uses it, with where it came from written above it.

## Three things to expect on real hardware

1. **Connecting moves the arm.** Opening the port resets the board, which homes the arm and centres the cup:
   about 4 s, sometimes 15.
2. **Fast and slow moves land 1-3 mm short**: the servos' deadband. PRECISE does not correct
   them (`work/STATUS-impala.md` §19.4).
3. **An arm that accepts commands and does not move** was seen once. `connect()` checks the servo supply; a move
   cannot tell, because it reads nothing back. Power-cycle the arm at the barrel jack.
