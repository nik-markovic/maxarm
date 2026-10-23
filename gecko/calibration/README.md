# calibration/ — the calibration web app

Specs: [`work/PILOT-falcon.md`](../../work/PILOT-falcon.md), the desk page;
[`work/PILOT-gecko.md`](../../work/PILOT-gecko.md), the camera and Fine Tune pages.
One web app with a page per calibration: the desk first, then the camera; then Fine Tune, which tries the camera's
calibration and refines it.

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
| orange arc, red arc       | the reach limits, 95 and 270 mm: nothing is sent outside them, a move may cross |
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

It brings the arm over each of the three cubes, keeps where the cup was over each one, takes a frame, and fits the
camera to the cubes in it: `config/camera.json`, the mapping between desk space and the camera's pixels. The GUIDE
button tells the same story as this section, for whoever is at the arm; the first visit opens it.

1. **Aim the camera** by the live view: the whole area where pieces will go, and not too far away.
2. **Set each cube down under the cup.** Click the grid -- the dashed squares are where the cubes are suggested to
   go, red on the side nearer the camera -- and the arm comes over, high enough to slide a cube under. Set the cube
   there, a corner towards the camera. Bring the cup down to the notch, 44, 4 mm over the cube's top, and centre
   the cube under the cup.
3. **RECORD** on the cube's row keeps where the servos say the cup is, read afresh, in desk space:
   `config/camera-points.json`. RECORD AGAIN replaces it.
4. **CALIBRATE**, once all three are recorded: the arm goes home, out of the picture, the camera takes a frame at
   its largest size, averaged over five -- about 11 s on the document camera -- and the camera is fitted to the
   cubes in it, about 2 s more. `config/camera.json` is written, with the frame beside it as
   `config/camera-frame.png`, and the check image shows the fit: the cubes' outlines in faint pink -- fainter
   still the edges their faces hide, and an X across each one's bottom, crossing under its centre -- the arm's
   suction cups in cyan where the calibration puts them, a dot every 50 mm of desk, and the reach. If the cubes
   cannot all be found, or do not stand where the arm recorded them, nothing is written, and the check image shows
   what was found.
5. **VALIDATE CALIBRATION** parks the arm the same way, takes a new frame and draws the saved calibration on it. If
   the cubes are still there it fits them again and says how far apart the two put the desk: more than 2 mm
   anywhere in reach, and the camera has probably moved. Click the check image to see it full size, and click that
   for actual size.

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
| CALIBRATE               | park the arm, take the frame, fit `camera.json` to the cubes in it                   |
| VALIDATE CALIBRATION    | park the arm, take a new frame, draw the calibration on it; has the camera moved?    |
| RESET                   | stop, let go, lift clear of the desk and of the cubes, and home                      |

**How it moves.** Desk space throughout, through the saved `desk.json` or the default desk. Below 60 mm over the
desk every move is slow: the cubes are there. A grid click from lower goes slowly straight up first, fast across
at 60, and stays there; a new height comes straight down, fast to 60 and slowly on, and ends *precise*. CALIBRATE,
VALIDATE and RESET all park the same way: slowly up to 10 mm over the desk if lower, fast on up to 60, then home.

**How the camera is fitted** (`camera/cubes.py`, `fit.py`). Every decision is measured in the frame itself, so
another camera, light or desk needs nothing retuned. The cubes are the blobs whose colour stands out from the
desk's, as measured in the same frame, that are whole and cube-shaped; red is the warm one. The arm's green base
can look as green as the green cube, so which blobs are the cubes, and green from blue, is settled by where the
arm says each cube stands: only the right ones fit a camera. A camera with square pixels is then fitted to every
visible edge of the three 40 mm cubes, each turned however it was set down. Where the arm says the cubes stand
gives the arm's own x/y error, a few percent, which `camera.json` takes in: a pixel names the desk point the arm
must be sent to. The arm should be out of the picture, as CALIBRATE leaves it; lens distortion is not modelled.

**The live view** is the camera at 800 x 600, about 10 frames a second, only while the page is open and in sight.
The snapshot needs the camera at its full size, so the view pauses for it and comes back after.

**Which camera.** The owner's document camera, by the USB port it hangs off (`usb-0000:0b:00.0-3.1`), when it is
plugged in; otherwise the first camera that is not virtual: OBS's virtual camera is never picked.
`--camera /dev/videoN` picks another (`camera/devices.py`).

**The guide** is `camera/guide.md`, plain text to edit as wanted. A line starting "- " is a bullet, and an indented
line under it carries it on; `**two stars**` make bold and `*one*` italic; a blank line starts a new paragraph.
Nothing else is special. The page reads it afresh each time GUIDE is pressed.

## The Fine Tune page

`tests/`. Does the camera calibration bring the arm to what the camera sees -- and, where it does not, the points
that make it better. The page is there once the camera page has written `config/camera.json`; until then its tab
is greyed and the page says so.

Left the grid and the height column, as on the camera page, with where the arm was sent and where its servos say
it is, and arrows that move the cup 1 mm, in desk space. Right the live view, large, with the calibration drawn
over it: the reach on the desk, orange and red; the arm's target as the cup's 21 mm rim at its height, faint green,
with a dashed line down to the desk under it; where its servos say the cup is, a white dot (GUIDES turns these
off). Where the servos say the cup landed more than 0.5 mm from where it was sent, its rim there is drawn in
yellow: on the real cup, the camera is right and the arm missed; elsewhere, the camera is off. Every rim is drawn
its true size, through the arm's x/y error.

1. **FIND RED CUBE.** Set the red cube down anywhere in view. The arm parks the way CALIBRATE parks it, out of the
   picture; the camera finds the red cube in the live view and the arm comes over it at 54 mm over the desk, then
   slowly down to 44: 4 mm over its top. The cup should be centred on it. The live view shows the cube as found --
   its edges in pink, fainter where hidden, an X under its centre -- and its desk x and y on its top; the grid
   shows its footprint.
2. **ARM TO DESK.** CLICK THE DESK, or a click on the live view, shows the live view as large as the window
   allows. Click a point of the desk there: a crosshair marks it at once, with the cup's 21 mm rim round it on the
   desk, in perspective, and its desk x and y. The arm comes over the point at 14 mm, then slowly down to 4 mm
   over **the desk** -- not over the top of whatever stands there, which it knocks aside. Click again while it
   moves and the newer point replaces the older one, if that is still waiting. Moving the mouse shows the desk
   point under it. Once the arm is there and read, the bar says how it landed, by its servos: within 0.5 mm, or
   how far off. RESET is in the bar; Esc or × closes it.
3. **Refine the calibration**: FIND RED CUBE in reverse. Take the three cubes away. Set the red cube down, a corner
   towards the camera, FIND RED CUBE, centre the cube under the cup as on the camera page, and **RECORD POINT**: where
   the servos say the cup is, read afresh; then the arm parks and the camera sees where the cube stands. A numbered
   point on the grid and in the table: `config/camera-refine-points.json`. Again in a new place, ten times or more,
   spread over where pieces will go -- the dashed circle on the grid and in the live view is the place in reach and in
   view furthest from every point so far. Each point, and each of the camera page's three cubes, says how far the
   camera puts the cube from where the arm said the cup was: through the saved calibration, *off now*, and through the
   one SAVE would write, *after SAVE*; green within 1 mm, yellow within 2, red beyond, the colour of the point's
   circle on the grid, as on the desk page. **SAVE CALIBRATION**, from three points on, writes `camera.json` refined
   by them and redraws the camera page's check image with it, on the calibration's own frame. With no points left, it
   puts back the three cubes' own calibration.

| control                | does                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------- |
| grid, height column    | as on the camera page: over a grid point 60 mm up; 70 to -2 over the desk; the notch  |
| ▲ ▼ ◀ ▶                | 1 mm the way the view shows, in desk space                                            |
| GUIDES                 | the reach and the arm drawn over the live view, or not                                |
| FIND RED CUBE          | park the arm, find the red cube in the live view, come over it at 54 and down to 44   |
| CLICK THE DESK         | the live view full size: a click sends the arm over the desk there, down to 4 mm      |
| RECORD POINT           | keep where the arm says the cup is, park the arm, see where the camera has the cube   |
| ×, CLEAR               | forget a point; all of them                                                           |
| SAVE CALIBRATION       | `camera.json` refined by the points; with none, the three cubes' own put back         |

**How the red cube is found** (`camera/locate.py`). The camera is the calibration's, held still: a 40 mm cube
standing on the desk has one look for each place and turn, and the cube is moved and turned until it looks like
the frame's red blob -- its silhouette onto the blob's outline, then each edge it shows onto the frame's own edges,
as CALIBRATE fits the camera. Only the warm pixels make the blob, so a red cube standing against the green or the
blue one in the picture is found all the same. Something red that a cube standing there would not cover -- a red
sheet on the desk, a cube on its side or on another -- is refused, saying so. The live view is the calibration's
picture at a smaller size (800 x 600 against 2592 x 1944 on the document camera): the camera is scaled to it, and
a live view of another shape is refused.

**How the points refine the calibration** (`camera/refine.py`). The camera stays as CALIBRATE fitted it to the three
cubes' edges. A point is where the arm said the cup was and the pixel where the camera saw the desk under the cube's
centre; through the camera that pixel is a place on the desk. How the arm's own x and y lie on the desk -- its x/y
error, which three cubes fix exactly, each with whatever error it was centred with -- is fitted again to all the
points and the three cubes at once: any turn, stretch and shift, the best for all of them together. RECORD POINT
refuses a cup that is not over a cube's top (40 to 60 mm over the desk) and a cube the camera sees more than half
a cube from where the arm said the cup was; nothing is recorded then. The points belong to the camera as
CALIBRATE fitted it: a new CALIBRATE sets them aside, in `camera-refine-points-previous.json`. VALIDATE
CALIBRATION, on the camera page, compares a new frame with the three cubes' own calibration, which `camera.json`
keeps beside the refined one, so a refinement does not read as a camera moved.

**How a click becomes a desk point.** The calibration at desk height 0, inverted: a point of the picture is a ray,
and the desk is where it meets it. The page draws with the calibration as shares of the picture's width and
height, and the app converts the click it is sent with the library's camera space; the two agree. A click above
the horizon, or out of the arm's reach, is refused before anything moves.

| directory | holds                                                                                             |
| --------- | ------------------------------------------------------------------------------------------------- |
| `common/` | the web server, the arm's one thread (`worker.py`), how it moves (`motion.py`), the look, the     |
|           | view from above with the height beside it (`grid.js`, `view.css`) and the height column of the    |
|           | pages with cubes on the desk (`height.js`)                                                        |
| `desk/`   | the desk page (`index.html`, `desk.js`, `grid.js`) and its requests (`api.py`, `plane.py`)        |
| `camera/` | the camera page (`index.html`, `camera.js`, `grid.js`, `guide.js`, `guide.md`), its requests      |
|           | (`api.py`, `store.py`), the fit (`cubes.py`, `fit.py`, `edges.py`, `model.py`), the check image   |
|           | (`check.py`), the red cube found (`locate.py`), the refinement (`refine.py`), the live view       |
|           | (`feed.py`), the snapshot (`capture.py`, which `main.py` uses too), which camera (`devices.py`)   |
| `tests/`  | Fine Tune (`index.html`, `tests.js`, `grid.js`, `view.js`, `shares.js`, `refine.js`) and its      |
|           | requests (`api.py`, `refine.py`)                                                                  |
