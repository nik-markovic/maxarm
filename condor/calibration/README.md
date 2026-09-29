# calibration/

Two halves of one measurement, per [`work/PILOT-condor.md`](../../work/PILOT-condor.md).
The arm puts three cubes at coordinates it chose; the camera photographs them; the mapping
between the two is what the demo runs on.

| file           | what it is                                                                |
| -------------- | ------------------------------------------------------------------------- |
| `arm.py`       | The arm half. Places the cubes and exports `CUBE_POSITIONS`.              |
| `calibrate.py` | The camera half. Snapshot → cubes → the fit → `../config/`.               |
| `detect.py`    | The cube detector, from the exploration session. Unchanged, moved here.   |
| `mapping.py`   | The mapping itself: solve it, apply it, save it, load it.                 |

```
./calibration/arm.py --dry-run         # print and check every target, board off
./calibration/arm.py                   # place the cubes, with the arm powered
./calibration/calibrate.py             # then: snapshot, fit, write the config
```

## arm.py

It goes to the reset pose, drops to the stacking height and stops. Build the stack directly
under the nozzle — **red on the desk, blue on top of it, green on top of that** — press Enter,
and it moves them one at a time onto the desk. Ctrl-C stops the arm where it stands.

`--dry-run` is worth reading first. It checks every coordinate against the arm's envelope and
the hardcoded base exclusion square (|x| ≤ 70, |y| ≤ 70) without opening the port, which is the
check that catches an edited coordinate before the arm does.

Each step is one command, from where the arm is to where the step says. The three descents onto
a cube are flown slower — 40 mm/s rather than 200 — and that is the only difference; no
waypoints are invented anywhere.

```
./calibration/arm.py --move red=120,-180     # afterwards: one cube somewhere else
```

That is not part of setting the scene. It exists because a fit with nothing to spare cannot
report its own error (below), and it prints the `calibrate.py` line to run next.

## The three coordinate systems

**CPCS** — camera pixel coordinates. What the detector reports.

**AACS** — application arm coordinates. Millimetres, x and y as the arm has them, and **z
measured up from the desk**: `z = 0` is the cup snug on the surface where it can seal against
it, a 4 mm tile is gripped at `z = 4`, a 40 mm cube at `z = 40`. This is what the app should
think in.

**Board z** — what `MaxArm.move_to()` takes, and neither of the above. The arm's reported z
drifts upward as it extends, so the same desk reads **48 near the base and 36 at full reach**.
`mapping.board_z(x, y, z_aacs)` converts; nothing in `maxarm/` knows AACS exists.

**They are two mappings and two objects.** `DeskPlane` is AACS ↔ board and has no camera in it
— it is the three surface heights the owner measured, and an app that already knows where it
wants to go needs nothing else, not even a calibration run:

```python
from arm import CUBE_POSITIONS
from mapping import DeskPlane

desk = DeskPlane.through(CUBE_POSITIONS.values())
arm.move_to(*desk.aacs_to_board((100.0, -254.0, 40.0)))   # board z 78, the green drop
desk.board_to_aacs(arm.get_state().position)              # and back
```

`DeskMapping` is the camera half, CPCS → AACS x and y. It carries a `DeskPlane`, so it can do
both steps in one call:

```python
from mapping import DeskMapping

mapping = DeskMapping.load(Path("../config/calibration.json"))

mapping.pixel_to_aacs(pixel)                 # where a thing lying on the desk is
mapping.pixel_to_aacs(pixel, seen_at_mm=4)   # ...seen by its top face, 4 mm up
mapping.pixel_to_board(pixel, grip_height_mm=4, seen_at_mm=4)   # where to send the cup
mapping.parallax_mm(pixel, 4.0)              # how far height moves the answer: ~9 mm
mapping.aacs_to_board(pose)                  # the plane's, for convenience
```

## What the two halves agree on

```python
from arm import CUBE_POSITIONS, CUBE_SIZE_MM

CUBE_POSITIONS    # {"green": (100.0, -254.0, 38.0), "blue": (-100.0, -254.0, 36.0),
                  #  "red": (0.0, -90.0, 48.0)}
CUBE_SIZE_MM      # 40.0
```

**x and y are the centre of each cube's base** — not its centre of mass. The base is the only
part of a 40 mm cube that touches the plane the camera is being mapped onto; its blob centre
floats 20 mm up and projects about 40 mm sideways at this camera's elevation. See
[`work/STATUS-condor-prototype.md`](../../work/STATUS-condor-prototype.md) §3, which is the most
important paragraph in that file.

**z is AACS zero at that x and y** — the board z at which the cup is snug on the *bare desk*
there, measured with the cube out of the way. It is not a pose for the cup on the cube, which is
40 mm higher: 78 at the far cubes and 88 at the near one, which is exactly what the scene
commands. The three differ by about 11 mm on one flat desk, and that difference is the reason
AACS exists — in AACS all three are zero.

Importing `arm.py` opens no port and moves nothing. The `SCENE` steps are **not** part of what
it exports — they are a script, not data.

## Height, which is not an offset

A pixel does not name a point, it names a **ray**. Where that ray lands depends on how high the
thing it shows sits: at this camera's elevation the printed top of a 4 mm tile is about **9 mm**
from the tile's own footprint. That is bigger than the arm's accuracy, so it cannot be ignored.

So the mapping is fitted on **two planes** — the cubes' base centres at AACS 0 and their top
centres at AACS 40, both out of the same frame — and interpolates between them. That
interpolation is **exact, not approximate**: a ray is a straight line, so where it crosses
`z = h` is linear in `h`. A 4 mm piece is a tenth of the way up that line, which makes it an
interpolation rather than an extrapolation, and the error at 4 mm is a tenth of the error at 40.

Two heights go into a pickup and they are different numbers:

| argument          | what it means                          | cube by its base | 4 mm tile by its top |
| ----------------- | -------------------------------------- | ---------------- | -------------------- |
| `seen_at_mm`      | how high the *feature in the pixel* is | 0                | 4                    |
| `grip_height_mm`  | how high the *cup* must be to seal     | 40               | 4                    |

A calibration solved without the cube tops cannot answer for any height but zero, and
`pixel_to_ground()` raises rather than answering 9 mm wrong.

## calibrate.py

```
./calibration/calibrate.py                          # snapshot, fit, write the config
./calibration/calibrate.py --image FRAME.jpg        # fit a frame taken earlier
./calibration/calibrate.py --dry-run                # fit and print, write nothing
./calibration/calibrate.py --keep --at red=120,-180,44
```

It takes the largest frame the camera will give, averages several of them, finds the three
cubes, pairs each base with the coordinate the arm placed it at, and solves. Output is
`../config/calibration.json`, plus the frame it used and an overlay to look at.

The cubes must be **on the desk** when the snapshot is taken, not on the stack they arrive in:
a cube standing on another cube has its base 40 mm in the air and maps to a point 40 mm from
where it appears to be. And the camera must not move relative to the arm afterwards — that is
the whole validity condition of the config.

### Three cubes and their tops are a camera, and this is the thing to know

A plane seen by a camera is a perspective transform, with eight degrees of freedom, and three
point pairs on the desk supply only six — alone, they fit an **affine** transform, which cannot
follow a camera's perspective. But each cube also gives its *top* centre, 40 mm up at the same
x and y, and points at two heights are enough for the camera itself: a 3×4 projection, eleven
freedoms, twelve equations from three cubes. `fit()` solves that when the tops are there, and
both planes come out of it perspective. On the owner's frame the cube base edges, which the
fit never sees, measure 38.6–43.1 mm through it; the affine fit had them at 28–68.

One equation to spare means the residuals are near zero and say little.

Two things in the output speak to this:

- **The base edge check.** A cube's base is 40 mm square wherever it stands, so the fit should
  make it 40 mm square. It uses corners the fit never saw, so it is an independent check.
- **The grid on `config/calibration-check.jpg`.** The arm's millimetres, x ±120 and y −260 to
  −80, drawn where the mapping thinks they are. Its squares should shrink with distance; a grid
  of parallelograms is an affine fit.

**A held-out error needs more points**, and `arm.py --move` is how to get one: move a cube to
another coordinate the arm chose, leave the camera alone, and

```
./calibration/calibrate.py --keep --at red=120,-180,44
```

fits through that frame *and* the points already in the config. At four the fit becomes a
perspective one; at five it can hold one out and report an error in millimetres, which is the
number that decides whether a side camera can drive this demo at all.

## Offline checks

```
../.venv/bin/python agenttools/test-scene.py         # arm.py against a fake board
../.venv/bin/python agenttools/test-calibration.py   # detect.py, mapping.py, calibrate.py
../.venv/bin/python agenttools/test-detect.py        # detect.py against a known camera
```

None needs a camera, an arm or a board. The mapping checks work by projecting the arm's
coordinates through a *known* perspective transform and asking the fit to find its way back,
which is the only way to tell a mapping that is right from one that merely reproduces its input.
