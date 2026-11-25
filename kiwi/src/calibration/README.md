# calibration/ — the calibration web app

Specs: [`work/PILOT-falcon.md`](../../../work/PILOT-falcon.md), the desk page;
[`work/PILOT-gecko.md`](../../../work/PILOT-gecko.md), the camera and Fine Tune pages;
[`work/PILOT-hyena.md`](../../../work/PILOT-hyena.md), the Tiles page;
[`work/PILOT-impala.md`](../../../work/PILOT-impala.md), the tweaks to all of them since;
[`work/PILOT-jackal.md`](../../../work/PILOT-jackal.md), what it shares with the /IOTCONNECT app;
[`work/PILOT-kiwi.md`](../../../work/PILOT-kiwi.md), the Chat page.
One web app with a page per calibration: the desk first, then the camera; then Fine Tune, which tries the camera's
calibration and refines it; then Tiles, which finds the 25 mm letter tiles, reads their letters, says which the
arm can pick up, and sets a word of them down; then Chat, where the language model on the board's Ara240 finds a
word of the tiles for the arm to set down.

```
../.venv/bin/python src/main.py --no-iotc               # then open http://localhost:8080
../.venv/bin/python src/main.py --no-iotc --host ::     # every address, for another machine's browser
```

The pages are served by the demo's one program, `src/main.py` (`run.sh` on the board), beside /IOTCONNECT;
`calibration.py` builds them on its arm, camera, tiles and model.

It needs `src/requirements.txt` (`src/requirements-pc.txt` on a PC); the web server is Python's own.
The pages' own files are in the pilot's [`web/`](../../web/), a directory per page; their requests are answered here,
and what the /IOTCONNECT app does the same way is in [`applib/`](../applib/): the tiles' flow, the red cube's, the
camera, the arm's moves, and what is drawn over the camera's picture.
Connecting resets the board, which homes the arm: 4 s, sometimes 15. While the arm is not connected -- not there
at the start, or lost on the way -- it is tried again every 30 s, and CONNECT tries at once.
The pages and the /IOTCONNECT commands share the arm's one thread (`applib/worker.py`): one command at a time,
whichever asks, and RESET or `stop` cuts short either's. They share the camera too: the pages' live view and the
stream are two branches of its one pipeline. Ctrl-C holds the arm where it is and closes the port.

## The desk page

It finds where the desk is under the arm, as a plane, and writes it to `config/desk.json`.

1. **Touch the desk.** Click the grid to send the arm over a point. Bring the cup down with the height slider beside
   the grid, then Z+ and Z−, until it sits snug on the desk; then SUCTION on should say SEALED. Wait for *arm says*
   to finish reading (it spins, greyed, for a few seconds); RECORD TOUCH POINT then keeps where the servos say the
   cup is, which is not always where it was sent.
2. **Spread the points.** At least three, spanning 200 mm in X and 150 in Y, and not in a row. Each point shows how
   far it sits above a desk -- *offsets from*: **fit**, the plane through all the points (from the fourth on);
   **saved**, the saved `desk.json`; **default**, the default desk, arm z 44 and level. Green within 1 mm, yellow
   within 2, red beyond. Touch a red one again, or delete it (×).
3. **SAVE CALIBRATION.** The plane goes to `config/desk.json`, and the file it replaces to `desk-previous.json`.
   Points are true, as the library reports them, so a level desk is a level plane. A `desk.json` saved before the
   library corrected heights is not used: SAVE again. `config/desk-points.json` keeps the points as the servos read
   them, raw, so a refined formula never changes a recorded touch; the page shows them true.
   The plane goes through every point in the list: delete touches on anything but the desk before saving.
   **No points: SAVE LEVEL DESK.** The same button, with no points in the list, saves the default desk instead -- level,
   arm z 44 -- once you confirm that the arm's base cups are pressed fully onto the desk and that, far out all round,
   the cup grips the desk at a reported TRUE 44 but not at 45. When true z alone already finds the desk, this beats a
   plane fitted through touches that are each a little off. One point in the list, and it is SAVE CALIBRATION again.
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
| Suction                   | the pump; on, it says SEALED, LOOSE or UNCERTAIN (only just touching), until a move |
| RECORD TOUCH POINT        | keeps where the servos say the cup is, read afresh; the target beside it        |
| SAVE LEVEL DESK           | with no points: `desk.json` as the default desk, level at 44, once confirmed    |
| *arm* in the table        | back to the point in arm space: up, over, slowly down                           |
| RESET                     | stop, let go, lift clear of the desk and the cubes, home with the cup centred   |

**How it moves.** Below the hover height -- true z 54 in arm space, 10 mm over the default desk, or 20 mm over the
desk in desk space -- every move is slow, 10 mm/s; above it, fast. A move across the grid from low goes straight up
first and stays at the hover height. Sliders, nudges and the moves to a point end *precise*: the servos stop short
of small moves by 1-3 mm, and a precise move is not corrected; it waits until the servos are steady. After each move
the servos are read and shown as *arm says* -- the mean of their reads once steady; after a precise move, the one it
ended on. Until then it is greyed with a spinner, and RECORD and SAVE wait.

**True and raw.** Everything the page sends and the server works with is true: touch points, offsets, SAVE, the
hover height. RAW only changes the arm numbers shown and typed in (target, *arm says*, the height column, the
table's X, Y, Z and sent) to the firmware's own, converted with the library's constants; heights above the desk are
the same either way. The arm moves the same in both.

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
   there, a corner towards the camera. Bring the cup down to the notch, 42, 2 mm over the cube's top, and centre
   the cube under the cup.
3. **RECORD** on the cube's row keeps where the servos say the cup is, read afresh, in desk space:
   `config/camera-points.json`. RECORD AGAIN replaces it.
4. **CALIBRATE**, once all three are recorded: the arm goes home, out of the picture, five fresh frames of the live
   view, at the camera's full size, are averaged -- under 2 s -- and the camera is fitted to the cubes in them,
   about 2 s more, to the cubes' readings by the formula alone, so no earlier correction gets in. The arm's base is
   found in the frame by its nuts, and the arm laid on the camera: its axis where the nuts put it, its turn from the
   cubes' bearings (`camera/anchor.py`). `config/camera.json` is written, with the frame
   beside it as `config/camera-frame.png`, and `config/arm-true.json`, the arm's correction the three cubes make,
   which the library converts with at once. A cube whose reach is more than 5 mm off what the other two say does
   not fit: CALIBRATE names it and leaves it out -- record it again and CALIBRATE. The check image shows the fit:
   the cubes' outlines, yellow and half see-through
   -- fainter still the edges their faces hide, and an X across each one's bottom, crossing under its centre, a
   pixel wide so that the cube's own edge shows -- the arm's
   suction cups in cyan where the calibration puts them, a dot every 50 mm of desk, and the reach. If the cubes
   cannot all be found, or one does not stand where the arm recorded it against the other two, nothing is written,
   and the check image shows what was found. A cube that does not fit is outlined in thick red, a red cross and
   arrow where the arm recorded it. What came of CALIBRATE stays in words beside the check image until the next,
   and the button shows **!** when it failed or a cube did not fit.

| control                 | does                                                                                 |
| ----------------------- | ------------------------------------------------------------------------------------ |
| grid click              | over a grid point 60 mm up, then down to the notch, 42                               |
| dashed squares          | where the cubes might go; solid where each was recorded, the cross where the cup was |
| height slider           | 70 down to -2 mm over the desk, or further to take in the arm's height, after RESET  |
|                         | say; yellow from the desk down 1 mm, red below                                       |
| 42 CUBE                 | down to 2 mm over a cube's top. The slider snaps to it from a millimetre either side |
| 80 SAFE                 | up to 80 over the desk, clear of the cubes anywhere in reach                         |
| Z+, Z−, ▲ 20 mm, ▼ 2 mm | as on the desk page                                                                  |
| RECORD, go, ×           | keep a cube's point; back over it, from above; forget it                             |
| ? GUIDE                 | how to place the camera and the cubes: `web/camera/guide.md`                         |
| CALIBRATE               | park the arm, take the frame, fit `camera.json` to the cubes in it                   |
| RESET                   | stop, let go, lift clear of the desk and of the cubes, and home                      |

**How it moves.** Desk space throughout, through the saved `desk.json` or the default desk. Below 60 mm over the
desk every move is slow: the cubes are there. A grid click from lower goes slowly straight up first, fast across
at 60, then down to the notch; a new height comes straight down, fast to 60 and slowly on, and ends *precise*.
CALIBRATE and RESET park the same way: slowly up to 10 mm over the desk if lower, fast on up to 60, then home.

**How the camera is fitted** (`applib/camera/cubes.py`, `fit.py`). Every decision is measured in the frame itself, so
another camera, light or desk needs nothing retuned. The cubes are the blobs whose colour stands out from the
desk's, as measured in the same frame, that are whole and cube-shaped; red is the warm one. The arm's green base
can look as green as the green cube, so which blobs are the cubes, and green from blue, is settled by where the
arm says each cube stands: only the right ones fit a camera. A camera with square pixels is then fitted to every
visible edge of the three 40 mm cubes, each turned however it was set down, first each cube wherever its shape
puts it -- which blob is which, and a camera near enough -- then again with each cube where the arm read the cup
over it: the arm's millimetres, as the library corrects the firmware's, are real ones, and its distances between
the cubes are the camera's measure, which a cube's small picture fixes only loosely; the cubes' 40 mm give their
shape. Each cube is then drawn where its own edges put it through that camera, and the page's line gives how far
each arm reading is from it. The stretch a turn, shift and stretch would take between the two is how the arm's x
and y measure against the camera's: information, not used. A pixel names the desk point the arm must be sent to.
The arm
should be out of the picture, as CALIBRATE leaves it; lens distortion is not modelled.

**The live view** is the camera at its full size, the size every calibration is fitted at, only while a page is
open and in sight, or a snapshot wants it. QUALITY or FPS, on the camera page, is which of the camera's modes:
QUALITY its largest YUYV, uncompressed, the sharpest picture and slower, made JPEGs for the pages; FPS its largest
MJPG, its own JPEGs passed on, faster. The choice is there only when the YUYV is at least as large as the MJPG; QUALITY at first. Changing it, after
a warning, sets the calibration and everything in its pixels aside, as `*-previous`: CALIBRATE again. Its focus is
held still, since focusing moves the picture (the C920's by up to 4%): at the focus `camera.json` was calibrated
at (`lens`), for the same camera; with none, the camera focuses itself once, until its picture keeps still, and
CALIBRATE keeps where it stopped. Zoom is at its least, the widest view; pan and tilt at 0; backlight compensation
off, so that a dark desk does not wash out what is light on it.
The camera is read through GStreamer (`applib/camera/feed.py`): a frame its driver marks damaged is dropped, and the
camera's own JPEGs go to the page as they come. The snapshot is five fresh frames of the view averaged: nothing
pauses for it. The document camera's raw format, YUYV, shows the same picture, at 2 frames a second
(`work/STATUS-impala.md` §1).

**Which camera.** The first USB camera that captures video, found at each start: the /dev/videoN moves with the
image and with what else is plugged in (`applib/camera/devices.py`). Nothing else is picked, a virtual camera neither.

**The guide** is `web/camera/guide.md`, plain text to edit as wanted. A line starting "- " is a bullet, and an indented
line under it carries it on; `**two stars**` make bold and `*one*` italic; a blank line starts a new paragraph.
Nothing else is special. The page reads it afresh each time GUIDE is pressed.

## The Fine Tune page

`tests/`. Does the camera calibration bring the arm to what the camera sees -- and, where it does not, the points
that make it better. The page is there once the camera page has written `config/camera.json`; until then its tab
is greyed and the page says so.

Left the grid and the height column, as on the camera page, with where the arm was sent and where its servos say
it is, the arrows that move the cup 1 mm, in desk space, and RECORD POINT. A grid click comes over the point 60 mm
up, then slowly down to the page's notch, 42: 2 mm over a cube's top. Right the camera: CLICK THE DESK over it, and
apart, a helper, FIND RED CUBE; the live view, large, with the calibration drawn over it: the reach on the desk,
orange and red; the arm's target as the cup's 21 mm rim at its height, faint green, with a dashed line down to the
desk under it (GUIDES turns these off). Where the servos say the cup is, is on the grid; over the picture only
once FIND RED CUBE or ARM TO DESK has brought the cup down (below). Every rim is drawn its true size.

1. **FIND RED CUBE**, a helper. Set the red cube down anywhere in view. The arm parks the way CALIBRATE parks it,
   out of the picture; the camera finds the red cube in the live view and the arm comes over it at 54 mm over the
   desk, then slowly down to 42: 2 mm over its top. The cup should be centred on it. The live view shows the cube
   as found -- its edges thin, yellow and half see-through, fainter where hidden, an X under its centre -- and its
   desk x and y on its top; the grid shows its footprint. Once the arm is down its servos are read: on the cube's
   top a small dot where the arm was sent, and the cup's 21 mm rim, green, where the servos say it is, both drawn on
   the top's plane so that
   the gap between them is the arm's own miss; the label adds it, X and Y, and so does the line under the camera.
   It only brings the cup **near** the cube: align the cube under the cup before RECORD POINT, which keeps where
   the arm is, not where the cube was found. Wherever the camera is way off, the grid and the arrows are the way
   over a cube. Its **?** says this on the page; so does ARM TO DESK's.
2. **ARM TO DESK.** CLICK THE DESK, or a click on the live view, shows the live view as large as the window
   allows, the red cube found cleared from it. Moving the mouse shows the desk point under it, the cup's rim round
   it in blue. Click a point of the desk there: a blue crosshair marks it at once, with the cup's 21 mm rim round
   it on the desk, in perspective, a dot and its desk x and y. The arm comes over the point at 14 mm, then slowly
   down to 2 mm over **the desk** -- not over the top of whatever stands there, which it knocks aside. From a cup at
   a cube's top -- 40 to 60 mm, as FIND RED CUBE or a grid click leaves it -- the arm parks first: straight across
   from there strains it. Click again while it moves and the newer point replaces the older one, if that is still
   waiting. Once the arm is down, only the blue dot is left; once its servos are read, the same marks in green
   where they say the cup is: the cup's rim at the height they say, the crosshair and the dot on the desk under it.
   **The green rim should sit on the cup in the picture**: that tests the calibration, wherever the arm came down.
   The bar says how it landed against the point, by its servos: within 0.5 mm, or how far off. RESET is in the
   bar; Esc or × closes it.
3. **Refine the calibration**: FIND RED CUBE in reverse. CALIBRATE's three cubes are points 1, 2 and 3, each with
   its cube beside its number; forgotten (×) like any other, a cube's point is gone, and its row on the camera page
   says so. Take the three cubes away. Set the red cube down, a corner
   towards the camera, click the grid where it stands -- or FIND RED CUBE -- align the cube under the cup with the
   arrows, as on the camera page, and **RECORD POINT**: where the servos say the cup is, read afresh; then the arm
   parks and the camera's snapshot, at full size, sees where the cube stands. A numbered
   point on the grid and in the table: `config/camera-refine-points.json`. A dialog then shows the snapshot zoomed
   in on the cube: the cube as found, yellow, and the edges in the picture it was fitted to, cyan; on its top a
   yellow dot over its centre and a green one where the arm read the cup. **REDO** looks for the cube again in the
   same snapshot, its contrast, gamma and sharpness tweaked at random, with the same reading -- the servos' from
   before the arm parked -- and the point takes the cube found; **UNDO** forgets the point; **KEEP**, Esc or ×
   keep it. Again in a new place, ten times or more,
   spread over where pieces will go -- the dashed circle on the grid and in the live view is the place in reach and in
   view furthest from every point so far. Each point says how far the arm's
   reading is from the cube as the camera sees it, across the desk (*XY*), and how far from 40 mm tall the camera
   sees the cube (*Z*, + taller): *now*, through the saved calibration and the arm's correction as the library
   converts with it -- a point in the saved correction is 0 -- and *without it*: how far its reading, through the
   correction of the other points, is from its cube; in the live view the point's dot is that place, its ring the cube.
   Green within 1.5 mm, yellow within 2.5, red beyond; the point's circle on the grid takes the colour of its XY now,
   as on the desk page. A point whose reach is more than 5 mm off what the others say does not fit (`anchor.py`): its
   badge is red, and SAVE names it and leaves it out -- its cube moved after RECORD, or the cup was off it; forget
   it and record it again. In Z one far off says the snapshot's edges were not all the cube's. **SAVE
   CALIBRATION**, from four points on, fits the camera again to every point's cube, on their bottom and vertical
   edges (the tops' are found inside the cube), finds the base through it in CALIBRATE's frame and lays the arm as
   CALIBRATE does -- the axis where the base is, the turn from all the points' bearings -- writes `arm-true.json`,
   the points that fit --
   their pattern over the whole desk and a mesh of what it leaves at each (`maxarm/mesh.py`) -- and `camera.json` with its report, and redraws the camera page's check image with it, on the
   calibration's own frame. With no points left, it puts back the three cubes' own calibration, no correction.
   Points not saved -- changed since the last SAVE or CALIBRATE -- are not used by the arm: SAVE then stands out,
   orange with "!", and leaving the page asks first; saved, SAVE is greyed out.
4. **DESK GRID**, a switch beside GUIDES, off unless turned on: desk space itself drawn on the desk over the live
   view, a thin pale line every 10 mm out past the reach, every 50 mm a little stronger, and the two lines through
   the arm's turning axis, x = 0 and y = 0, stronger still. Lay grid paper along it: where the camera is right, the
   paper's lines run with these all over the sheet.

| control                | does                                                                                  |
| ---------------------- | ------------------------------------------------------------------------------------- |
| grid                   | over a grid point 60 mm up, then down to the notch, 42                                |
| height column          | as on the camera page                                                                 |
| ▲ ▼ ◀ ▶                | 1 mm the way the view shows, in desk space                                            |
| GUIDES                 | the reach and the arm drawn over the live view, or not                                |
| DESK GRID              | desk space every 10 mm drawn on the desk over the live view, or not (off at first)    |
| FIND RED CUBE          | park the arm, find the red cube in the live view, come over it at 54 and down to 42   |
| CLICK THE DESK         | the live view full size: a click sends the arm over the desk there, down to 2 mm      |
| RECORD POINT           | keep where the arm says the cup is, park the arm, see the cube in the snapshot; then  |
|                        | the snapshot zoomed in on the cube, and UNDO, REDO (look again, tweaked), KEEP        |
| ×, CLEAR               | forget a point; all of them                                                           |
| SAVE CALIBRATION       | `arm-true.json`, the points that fit (pattern and mesh); `camera.json`'s frame kept   |

**How the red cube is found** (`applib/camera/locate.py`; the arm brought over it, `applib/redcube.py`; drawn,
`applib/overlay/cube.py`, as the app's `find-cube` finds and draws it). The camera is the calibration's, held still:
a 40 mm cube standing on the desk has one look for each place and turn, and the cube is moved and turned until it
looks like the frame's red blob -- its silhouette onto the blob's outline, then each edge it shows onto the frame's
own edges, as CALIBRATE fits the camera. Only the warm pixels make the blob, so a red cube standing against the
green or the blue one in the picture is found all the same. Something red that a cube standing there would not
cover -- a red sheet on the desk, a cube on its side or on another -- is refused, saying so. The live view is the
calibration's own size; a smaller one would do, the camera scaled to it, but one of another shape is refused.

**How the points refine the calibration** (`camera/refine.py`, `applib/camera/sighting.py`). The three cubes put the
camera roughly in place; the points then make it. A point is where the arm said the cup was and the edges found
round the cube in the snapshot -- the frame's own, so they hold for any camera. **The camera** -- focal length,
principal point (held loosely to the frame's middle), where it stands and which way it looks -- is fitted with
every point's cube, 40 mm, standing where the arm read the cup over it, its millimetres real ones, turned as its
edges say, the cubes' points 1 to 3 among them until forgotten: the arm's distances between the points are the
camera's measure, the cubes' 40 mm their shape. No turn or shift lies between the arm and the camera. A point's XY
is its cube, placed where its own edges put it through the camera the other points make, against its reading: the
point left out. The stretch a turn, shift and stretch would take between the readings and the cubes as seen is the
arm's measure against the camera's, shown, and checked. Z is the cube's height through the camera, made free: a
picture's own check. A point kept before the edges were has only the pixel under the cube's centre, as the camera
of its day saw it: it counts for the arm, not for the camera, and has no Z. The cubes' own calibration is only
where the fit starts. Points all in one corner do not fix a camera: SAVE refuses one whose focal length is more
than 15% from the cubes', or an arm stretched more than 12% or skewed more than 6 degrees. RECORD POINT
refuses a cup that is not over a cube's top (40 to 60 mm over the desk) and a cube the camera sees more than half
a cube from where the arm said the cup was; nothing is recorded then. The points belong to the camera as
CALIBRATE fitted it: a new CALIBRATE sets them aside, in `camera-refine-points-previous.json`, and writes its
three cubes as the new points 1, 2 and 3. A file from before the cubes were points gets them once, its points
numbered on after them.

**How a click becomes a desk point.** The calibration at desk height 0, inverted: a point of the picture is a ray,
and the desk is where it meets it. The page draws with the calibration as shares of the picture's width and
height, and the app converts the click it is sent with the library's camera space; the two agree. A click above
the horizon, or out of the arm's reach, is refused before anything moves.

## The Tiles page

`tiles/`. Where the 25 mm tiles are, their letters and which way up, which of them the arm can pick up, and a word of
them set down in the dropoff area: what the /IOTCONNECT app's `detect` and `place` do, by the same code
(`applib/tilework.py`). Everything drawn over the camera's picture -- the areas, the tiles, their letters, the word --
is drawn in Python (`applib/overlay/tiles.py`), as the app's stream shows it, and laid over the live view; the tile
picked or under the pointer, and the arm, the page draws over that.
The page is there once the camera page
has written `config/camera.json`; until then its tab is greyed and the page says so.

Left the view from above, with the work areas: **pickup**, green, the outer two thirds of the reach, both sides;
**dropoff**, blue, the inner third. The dashed lines are the line between
them and the arm's centre line. Right the camera, LOCATE TILES in its heading. Across the bottom the tiles' table.

1. **Set the tiles down**, letters up, where the camera sees them. A letter is a capital printed in black marker;
   on a letter that is another turned -- M and W -- a dot at its bottom right says which it is. Say how thick they
   are in **tile height**, beside LOCATE TILES: 2 mm to start. The tiles are looked for that high, and the grip, the
   release and the way across are counted from it.
2. **LOCATE TILES.** The arm parks the way CALIBRATE parks it, out of the picture; the camera takes its snapshot at
   full size -- under 2 s -- and the tiles are found in it, about 2 s more. The live view shows them outlined on
   it. Each tile is
   coloured by where it is, the same on the grid, over the picture and in the table:
   - **ready**, green: in the pickup area, in reach, at least 2 mm from every other tile;
   - **too close**, yellow: in the pickup area, but less than 2 mm from another, which the cup coming down might
     push: every tile is where the snapshot measured it, so none may move before its turn;
   - **outside the pickup area**, grey: in reach, but in the dropoff area or right of the centre line;
   - **out of reach**, red: the cup cannot be sent to its centre, kept 5 mm inside the arm's 95 and 270 mm arcs, or
     the arm cannot come down onto it there.

   Each tile's letter is read in the same snapshot, a second or so more. It is shown on a badge at the face's
   corner at the letter's bottom right, so the badge says which way up the letter lies; **?**, ringed in white, is a
   letter not read -- the reader under 60% sure of it -- its badge and its tile's outline red, on the grid and over
   the picture; and **–**, ringed, a blank tile. The table has the letter,
   how sure the reader is, its next best capital, *dot* when the dot was found, and *up*: where the letter's top
   points, degrees from desk +x, starred when it reads the same turned (N, S, Z upside down; O, X any way). The
   letters, left to right, are said under the picture.
3. **Try a tile's centre.** A tile clicked, on the grid, over the picture full size, or its row's *go*: the cup
   comes over its centre at 14 mm over the desk, then slowly down to 4 mm over the tile's top. It should be
   centred on it. A tile out of reach is refused, nothing moved.
4. **Set a word down.** Until letters are typed in the box beside PLACE, LOCATE TILES puts five there at random
   from the tiles ready to pick up; type any others, as many as there are ready tiles for. Letters typed stay, LOCATE
   after LOCATE; emptied, the box is offered letters again. PLACE checks them -- letters on ready tiles, read,
   each tile once, nothing found lying where they go -- and refuses, saying why, before the arm moves. Then each
   tile in turn: the cup over it, slowly down as far under the tile's top as **tile offset** beside PLACE says -- 1 mm
   to start -- read there, grip, up; across to its place with the cup at twice the tile's thickness and 4 mm, 8 mm for
   2 mm tiles, turning it on the way, slowly down to 2 mm over its top, let go -- the cup's vent held open -- up.
   The places are along the dotted arc 130 mm from the base axis, 25 mm in from the dropoff
   area's outer edge, 5 mm apart at their corners nearest the arm, centred on the arm's centre line, each letter
   turned and the word ordered for where its **reader** stands, the choice beside tile offset:
   - **back**, behind the arm: each letter's top away from the arm, the word from the arm's left to its right;
   - **front**, facing the arm: each letter's top towards it, the word from the arm's right to its left;
   - **left**, **right**, at that side of the arm: each letter turned a quarter, its top away from the reader, the
     word read top to bottom, from the far end of the arc to the near one.

   The tile height, the tile offset and the reader are the app's settings, kept in `config/appcfg.json` as soon as they
   change, and used by the /IOTCONNECT app too; the file holds only what differs from the defaults. A turn the cup
   cannot make after the grip, past 90 degrees, it makes half before. Each place is drawn, dashed, and filled once
   its tile is there. At the end -- or a refusal or failure on the way, but not RESET -- the arm parks and the tiles
   are found and read again in a new snapshot, as LOCATE TILES finds them. Where the tiles carried were stays
   drawn, dotted grey, the letter faint at its corner and a chevron to where its top pointed -- to put the tiles
   back by -- PLACE after PLACE, until LOCATE TILES; the box keeps its letters. RESET lets go of a tile held,
   wherever the arm is, and then LOCATE TILES before the next PLACE.

| control             | does                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------ |
| LOCATE TILES        | park the arm, take the snapshot at full size, find the tiles, read and place them    |
| tile height         | how thick the tiles are, mm: `appcfg.json`                                           |
| PLACE               | the tiles with the letters typed to the dropoff arc, gripped as deep as tile offset  |
|                     | says; then park, and find and read the tiles again                                   |
| tile offset, reader | how far under a tile's top the cup grips it; where the word's reader stands: both    |
|                     | `appcfg.json`                                                                        |
| Areas               | the work areas and the arm's reach drawn on the desk, or not                         |
| FULL SIZE           | the picture as large as the window allows; the tile under the pointer is said        |
| a tile, *go*        | the cup over the tile's centre, down to 4 mm over its top                            |
| RESET               | stop, let go, lift clear of the desk and of what is on it, and home                  |

**How the tiles are found** (`applib/tiles/`). The snapshot is resampled onto the plane of the tiles' top faces,
the tile height up, where every tile is the same 25 mm square, perspective gone. Every colour edge votes for the
centre of a square it could be a side of, either side of it: a tile's four sides all vote for its centre, and the
seam between two touching tiles for both of theirs. A square is fitted to the edges round each place most voted
for -- its centre to a few hundredths of a millimetre, its turn to a tenth of a degree, its size from 25 to 27.5
mm, as a face looks larger close to the camera -- and kept as a tile if each of its sides stands out from what lies just beyond
it, beyond its side face where the camera sees one. Nothing is assumed of the tiles' or the desk's colour or
brightness. A tile cut by the picture's edge, or behind the arm, is not looked for. The squares are fitted on the
board's other cores too (`applib/crew.py`).

**How a letter is read** (`applib/letters/`). The tile's face is resampled square-on from the snapshot, along its
sides. Ink is what is darker than the face round it; a face with none is blank, and a small round blob of ink off a
corner of the letter is the dot, which puts the letter's bottom right there. A small network (`files/letters.tflite`,
int8, trained on drawn hand-printed capitals by `training/letters/train.py`) looks at the face at each of its four
quarter turns and says which capital it is if that turn is upright, or that it is not; the best wins, at the dot's
turn if there is one.

**How a tile is placed** (`applib/areas.py`). By its centre, in desk mm round the arm's base axis; the areas and the
2 mm gap are the app's own, for now, and the reach the library's: `can_reach()` at the tile's top and over it.

**How a word is set down** (`applib/word.py`, the library's `carry()`). Of several ready tiles with a letter, the
one whose letter needs the least turning goes; an N, an S or an O is turned to whichever of its ways up is nearest.
The library's `carry()` picks the tile up and sets it down, and knowing both ends, shares out a turn the cup cannot
make from centre.

## The Chat page

`chat/`. A request to the language model on the board's Ara240 -- Qwen2.5-7B-Instruct behind NXP's connector,
`connector/` (the pilot's README, "The Ara240") -- which finds a word of the tiles and has the arm set it down: what
the /IOTCONNECT app's `agent` command does, by the same code (`applib/agent/`). The tiles are the Tiles page's: the
same LOCATE TILES and PLACE, with its tile offset and reader. Like it, the page is there once `config/camera.json`
is. Without the connector the page says why the model is not there; everything else runs as before.

Left the desk: the live view with the tiles drawn over it, the letters on the tiles ready to pick up, as tiles, and
the word being set down, each tile filled once the arm has set it down. Right the conversation.

1. **Wait for the model.** Its pill, in the conversation's heading, says where it is: not installed, not running,
   waiting for the Ara240, loading -- how long so far, of about 6 minutes -- or ready. Until it is ready, the box
   waits.
2. **Ask**, in the box or by a suggestion under the conversation:
   - **go** (run, start, ...): the longest word the model finds of the tiles, set down;
   - **a 7-letter word**: one of exactly that many letters, the number read from the request;
   - **a clue** -- a farm animal, something you sit on: a word that matches it, or none if the tiles spell none;
   - **spell AVNET** (place, write): that word as it is, or what letters the tiles lack for it;
   - **elaborate** (explain): the last word explained, nothing moved;
   - **look at the tiles again, then ...**: LOCATE TILES first;
   - **a word, but don't place it**: the word, nothing moved.
3. **Watch it work.** Each tool call shows as it happens: the words weighed -- for a clue those nearest it in
   meaning, else the longest the tiles spell -- with their letter count, the one chosen filled; a word of the
   visitor's struck through when the tiles lack its letters. Then the model's answer, with what it took: seconds,
   model calls, tokens read and written; under it, **what the model did**, everything that went to and from the
   model for it, as text to copy. The word is set down while the model explains it; when the arm is done, the
   conversation says so. The first request after a word is set down looks at the tiles again first.

One request at a time: another meanwhile is refused, and so is one before the model is ready. **STOP** cuts the
model short; RESET stops the arm. The model is told what happened before in one line -- the last word and what
became of it -- never the conversation itself: **NEW VISITOR** starts both again, and so do ten minutes of quiet.

| control      | does                                                                               |
| ------------ | ---------------------------------------------------------------------------------- |
| ASK, Enter   | the request to the model                                                           |
| a suggestion | that request                                                                       |
| STOP         | the model's answer cut short                                                       |
| NEW VISITOR  | the conversation, and what the model is told of it, started again                  |
| LOCATE TILES | as on the Tiles page                                                               |

**How a word is found** (`applib/agent/`). The model says what the request wants (`find_word`): the longest, a clue,
or a word of the visitor's. Code finds the words the tiles spell, each tile once, in the word list
(`files/words-en.tsv`, `dictionary.py`) -- the model cannot count letters -- and plans the word as PLACE would. For a
clue the twelve nearest it in meaning, by a small embedding model on the CPU, are put to the model, which names the
one the clue fits, or none; for a length, the everyday words of it, and it names the most vivid (`finder.py`). A word
set down before this visit is not chosen again while another fits. A tool call the model writes out as text is made
all the same (`repair.py`).

From the pilot's root:

| directory              | holds                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------- |
| `src/calibration/`     | the pages (`calibration.py`) and each page's requests: `common/` the web server and    |
|                        | the arm's controls (`arm_api.py`); `desk/` the plane and                               |
|                        | its files; `camera/` CALIBRATE, the check image (`check.py`), the                       |
|                        | refinement (`refine.py`); `tests/` Fine Tune; `tiles/` the Tiles page; `chat/` the      |
|                        | Chat page                                                                               |
| `web/`                 | the pages: `common/` the look, the view from above with the height beside it (`grid.js`,|
|                        | `view.css`, `height.js`), desk space to the picture and back (`shares.js`), the marks  |
|                        | drawn over it (`marks.js`), a layer drawn in Python laid over it (`layer.js`), the      |
|                        | picture full size (`fullview.js`); `desk/`, `camera/` (its GUIDE, `guide.md`),          |
|                        | `tests/`, `tiles/`, `chat/`, each page's `index.html`, script, styles and grid          |
| `src/applib/`          | what the /IOTCONNECT app shares with the pages: the camera (`camera/`: which camera,    |
|                        | the live view, the fit and its files, the cubes, the red cube), the arm's moves         |
|                        | (`motion.py`), the tiles' flow (`tilework.py`), the red cube's (`redcube.py`), what is  |
|                        | drawn over the picture (`overlay/`), the settings (`settings.py`); and the tiles found, |
|                        | read, placed and a word of them planned (`tiles/`, `letters/`, `areas.py`, `word.py`);  |
|                        | the language model and its tools (`agent/`)                                             |
