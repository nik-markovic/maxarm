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

That is not part of setting the scene. It exists because three cubes are one point pair short
of what this camera needs (below), and it prints the `calibrate.py` line to run next.

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

**z is the desk under that cube, and is not a pose the cup can be sent to.** The cup grips a
cube's top face, so it grapples at base + 40 — which is exactly what the scene commands, 78 at
the far cubes and 88 at the near one. Two of the three heights are below the library's own z
floor of 48; they are the desk, which is the point. `mapping.to_grapple(pixel, height_mm)` is
the call that adds the object's height back, and the height is an argument because the letter
tiles that come later are 4 mm.

The three heights differ by about 11 mm on one flat desk. That is the arm's own z reading high
as it extends ([`work/STATUS-condor.md`](../../work/STATUS-condor.md) §8.2); the mapping fits a
plane through them and interpolates it rather than pretending the desk is level in arm z.

Importing `arm.py` opens no port and moves nothing. The `SCENE` steps are **not** part of what
it exports — they are a script, not data.

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

### Three cubes is one point pair short, and this is the thing to know

A plane seen by a camera is a perspective transform. That has eight degrees of freedom and
three point pairs supply six, so a three-cube fit is **affine**: exact at the three cubes, and
wrong between and beyond them by an amount the residuals cannot show — they are zero by
construction. At this camera's 23° elevation the scale across one frame swings by 2.7×, which
is exactly what an affine fit cannot follow.

Two things in the output speak to this:

- **The base edge check.** A cube's base is 40 mm square wherever it stands, so the fit should
  make it 40 mm square. This is the only local accuracy check available from three points, and
  the far cube is where a failing fit shows first.
- **The grid on `config/calibration-check.jpg`.** The arm's millimetres drawn where the mapping
  thinks they are. A grid that shears away from the desk at the far end is the same fault, seen.

**The fix is a fourth point pair**, and `arm.py --move` is how to get one: move a cube to
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
```

Neither needs a camera, an arm or a board. The mapping checks work by projecting the arm's
coordinates through a *known* perspective transform and asking the fit to find its way back,
which is the only way to tell a mapping that is right from one that merely reproduces its input.
