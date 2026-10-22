# calibration/ — the calibration web app

Specs: [`work/PILOT-falcon.md`](../../work/PILOT-falcon.md), the desk page;
[`work/PILOT-gecko.md`](../../work/PILOT-gecko.md), the camera page.
One web app with a page per calibration: the desk first, then the camera.

```
../.venv/bin/python calibration/calibration.py                  # then open http://localhost:8080
../.venv/bin/python calibration/calibration.py --host 0.0.0.0   # on the board, for another machine's browser
../.venv/bin/python calibration/calibration.py --camera /dev/video2   # a camera other than the one it picks
../.venv/bin/python agenttools/serve-fake.py                    # the same pages on a fake arm and camera
```

It needs `numpy`, `pyserial` and OpenCV (`requirements.txt`); the web server is Python's own.
Connecting resets the board, which homes the arm: 4 s, sometimes 15.
This process is the only thing that talks to the arm, and the only one using the camera.
Ctrl-C holds the arm where it is and closes the port.

## The desk page

It finds where the desk is under the arm, as a plane, and writes it to `config/desk.json`.

1. **Touch the desk.** Click the grid to send the arm over a point. Bring the cup down with the height slider beside
   the grid, then Z+ and Z−, until it sits snug on the desk -- with SUCTION on it should seal. Wait for *arm says*
   to finish reading (it spins, greyed, for a few seconds); RECORD TOUCH POINT then keeps where the servos say the
   cup is, which is not always where it was sent.
2. **Spread the points.** At least three, spanning 200 mm in X and 150 in Y, and not in a row. Each point shows how
   far it sits above a desk -- *offsets from*: **fit**, the plane through all the points (from the fourth on);
   **saved**, the saved `desk.json`; **default**, the default desk, arm z 44 and level. Green within 1 mm, yellow
   within 2, red beyond. Touch a red one again, or delete it (×).
3. **SAVE CALIBRATION.** The plane goes to `config/desk.json`, and the file it replaces to `desk-previous.json`.
   Points are true z, as the library reports them, so a level desk is a level plane. A `desk.json` saved before the
   library corrected heights is not used: SAVE again. `config/desk-points.json` keeps the points as the servos read
   them, in raw z, so a refined formula for true z never changes a recorded touch; the page shows them in true z.
   The plane goes through every point in the list: delete touches on anything but the desk before saving.
4. **Try it.** *desk* in the table sends the cup over that point in desk space, at the point test height: 4 mm,
   down to 0. The first comes down from above, like `grab()`; the next ones glide, slowly and level, so the gap
   between cup and desk can be watched all the way. Any other move in between, and the next one comes from above
   again. TEST INPUT comes down onto any desk point, grips, holds, lets go, and stays there. Without a `desk.json`
   all of this uses the default desk.

| control                   | does                                                                            |
| ------------------------- | ------------------------------------------------------------------------------- |
| grid click                | over to that grid point (10 or 20 mm grid); X and Y fine go back to 0           |
| orange arc, red arc       | the reach limits, 95 and 280 mm: nothing is sent outside them, a move may cross |
| height slider             | 80 down to 40, 1 mm steps; yellow from 44 to 42, red below: may touch the desk  |
| 120 SAFE                  | up to 120; green while the arm is that high or higher, as after RESET           |
| Z+, Z−                    | 1 mm up or down from the height sent; the label shows that height               |
| ▼ 2 mm                    | the slider goes 2 mm lower, as often as wanted; only the firmware stops it      |
| ▲ 20 mm                   | the slider goes 20 mm higher, as often as wanted; only the firmware stops it    |
| X fine, Y fine, nudge pad | offsets of up to a grid step, 1 mm at a time. The pad follows the view's turn   |
| TRUE / RAW                | arm z as the library has it, or the firmware's own; the ? says why they differ  |
| Cup                       | the cup's servo, -90 to 90                                                      |
| RECORD TOUCH POINT        | keeps where the servos say the cup is, read afresh; the target beside it        |
| *arm* in the table        | back to the point in arm space: up, over, slowly down                           |
| RESET                     | stop, let go, lift clear of the desk and the cubes, home with the cup centred   |

**How it moves.** Below the hover height -- true z 54 in arm space, 10 mm over the default desk, or 20 mm over the
desk in desk space -- every move is slow, 10 mm/s; above it, fast. A move across the grid from low goes straight up
first and stays at the hover height. Sliders, nudges and the moves to a point end *precise*: the servos stop short
of small moves, so the arm is read and sent on until it is within a pulse, about 0.5 mm -- some 1.4 s a move. After
each move the servos are read and shown as *arm says*; until then it is greyed with a spinner, and RECORD and SAVE
wait. Near the desk a precise move may press a cup that is already on it up to 3 mm, then stops.

**True and raw.** Everything the page sends and the server works with is true z: touch points, offsets, SAVE, the
hover height. RAW only changes the arm z shown and typed in (target, *arm says*, the height column, the table's Z
and sent) to the firmware's own, converted with the library's constants; heights above the desk are the same either
way. The arm moves the same in both.

**Which space.** Everything but the desk test is arm space with no calibration at all, so no older calibration can
keep the cup off the desk being measured. The desk test is desk space through the saved `desk.json`, or the default
desk without one: what every app will use.

## The camera page

It brings the arm over each of the three cubes, keeps where the cup was over each one, and takes the frame the
camera is fitted to. **The fit -- finding the cubes in that frame and writing `config/camera.json` -- is not built
yet**: CALIBRATE stops at the frame, and the check image is the frame as it is. The GUIDE button tells the same
story as this section, for whoever is at the arm; the first visit opens it.

1. **Aim the camera** by the live view: the whole area where pieces will go, and not too far away.
2. **Set each cube down under the cup.** Click the grid -- the dashed squares are where the cubes are suggested to
   go, red on the side nearer the camera -- and the arm comes over, high enough to slide a cube under. Set the cube
   there, a corner towards the camera. Bring the cup down to the notch, 44, 4 mm over the cube's top, and centre
   the cube under the cup.
3. **RECORD** on the cube's row keeps where the servos say the cup is, read afresh, in desk space:
   `config/camera-points.json`. RECORD AGAIN replaces it.
4. **CALIBRATE**, once all three are recorded: the arm goes home, out of the picture, and the camera takes a frame
   at its largest size, averaged over five -- about 11 s on the document camera. It is kept as
   `config/camera-frame.png`.
5. **CHECK CALIBRATION** parks the arm the same way and takes a new check image, leaving the calibration's frame
   alone. Click the check image to see it full size, and click that for actual size.

| control                 | does                                                                                 |
| ----------------------- | ------------------------------------------------------------------------------------ |
| grid click              | over to that grid point, 60 mm over the desk or higher: clear of the cubes           |
| dashed squares          | where the cubes might go; solid where each was recorded, the cross where the cup was |
| height slider           | 70 down to -2 mm over the desk; yellow from the desk down 1 mm, red below            |
| 44 CUBE                 | down to 4 mm over a cube's top. The slider snaps to it from a millimetre either side |
| 80 SAFE                 | up to 80 over the desk, clear of the cubes anywhere in reach                         |
| Z+, Z−, ▲ 20 mm, ▼ 2 mm | as on the desk page                                                                  |
| RECORD, go, ×           | keep a cube's point; back over it, from above; forget it                             |
| ? GUIDE                 | how to place the camera and the cubes: `camera/guide.md`                             |
| CALIBRATE               | park the arm, take the frame; the fit is to come                                     |
| CHECK CALIBRATION       | park the arm, take a new check image                                                 |
| RESET                   | stop, let go, lift clear of the desk and of the cubes, and home                      |

**How it moves.** Desk space throughout, through the saved `desk.json` or the default desk. Below 60 mm over the
desk every move is slow: the cubes are there. A grid click from lower goes slowly straight up first, fast across
at 60, and stays there; a new height comes straight down, fast to 60 and slowly on, and ends *precise*. CALIBRATE,
CHECK and RESET all park the same way: slowly up to 10 mm over the desk if lower, fast on up to 60, then home.

**The live view** is the camera at 800 x 600, about 10 frames a second, only while the page is open and in sight.
The snapshot needs the camera at its full size, so the view pauses for it and comes back after.

**Which camera.** The owner's document camera, by the USB port it hangs off (`usb-0000:0b:00.0-3.1`), when it is
plugged in; otherwise the first camera that is not virtual: OBS's virtual camera is never picked.
`--camera /dev/videoN` picks another (`camera/devices.py`).

**The guide** is `camera/guide.md`, plain text to edit as wanted. A line starting "- " is a bullet, and an indented
line under it carries it on; `**two stars**` make bold and `*one*` italic; a blank line starts a new paragraph.
Nothing else is special. The page reads it afresh each time GUIDE is pressed.

| directory | holds                                                                                             |
| --------- | ------------------------------------------------------------------------------------------------- |
| `common/` | the web server, the arm's one thread (`worker.py`), how it moves (`motion.py`), the look, and the |
|           | view from above with the height beside it (`grid.js`, `view.css`)                                 |
| `desk/`   | the desk page (`index.html`, `desk.js`, `grid.js`) and its requests (`api.py`, `plane.py`)        |
| `camera/` | the camera page (`index.html`, `camera.js`, `grid.js`, `guide.js`, `guide.md`), its requests      |
|           | (`api.py`, `store.py`), the live view (`feed.py`), the snapshot (`capture.py`, which `main.py`    |
|           | uses too) and which camera (`devices.py`)                                                         |
