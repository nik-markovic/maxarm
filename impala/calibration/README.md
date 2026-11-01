# calibration/ — the calibration web app

Specs: [`work/PILOT-falcon.md`](../../work/PILOT-falcon.md), the desk page;
[`work/PILOT-gecko.md`](../../work/PILOT-gecko.md), the camera and Fine Tune pages;
[`work/PILOT-hyena.md`](../../work/PILOT-hyena.md), the Tiles page;
[`work/PILOT-impala.md`](../../work/PILOT-impala.md), the tweaks to all of them since.
One web app with a page per calibration: the desk first, then the camera; then Fine Tune, which tries the camera's
calibration and refines it; then Tiles, which finds the 25 mm letter tiles, reads their letters, says which the
arm can pick up, and sets a word of them down.

```
../.venv/bin/python calibration/calibration.py                  # then open http://localhost:8080
../.venv/bin/python calibration/calibration.py --host 0.0.0.0   # on the board, for another machine's browser
../.venv/bin/python calibration/calibration.py --camera /dev/video2   # a camera other than the one it picks
../.venv/bin/python agenttools/serve-fake.py                    # the same pages on a fake arm and camera
```

It needs `numpy`, `pyserial`, OpenCV and a TFLite interpreter (`requirements.txt`); the web server is Python's own.
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
   Points are true, as the library reports them, so a level desk is a level plane. A `desk.json` saved before the
   library corrected heights is not used: SAVE again. `config/desk-points.json` keeps the points as the servos read
   them, raw, so a refined formula never changes a recorded touch; the page shows them true.
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
of small moves, so the arm is read and each servo off its target nudged onto it, about 0.5 mm -- some 5 s a move.
After each move the servos are read and shown as *arm says* -- the mean of their reads once steady; after a precise
move, the one it ended on. Until then it is greyed with a spinner, and RECORD and SAVE wait. Near the desk a precise
move may push a cup that is already on it up to 4 mm, three times at most.

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
   about 2 s more. `config/camera.json` is written, with the frame beside it as
   `config/camera-frame.png`, and the check image shows the fit: the cubes' outlines in faint pink -- fainter
   still the edges their faces hide, and an X across each one's bottom, crossing under its centre, a pixel wide so
   that the cube's own edge shows -- the arm's
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
| height slider           | 70 down to -2 mm over the desk, or further to take in the arm's height, after RESET  |
|                         | say; yellow from the desk down 1 mm, red below                                       |
| 42 CUBE                 | down to 2 mm over a cube's top. The slider snaps to it from a millimetre either side |
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
visible edge of the three 40 mm cubes, each turned however it was set down: the cubes' 40 mm are its ruler. The
arm is tied to it by a turn, a shift and a stretch, which three points fix exactly: each cube stands where the arm
said, 0 mm off, since three points have nothing to check one another against. The stretch is how the arm's x and
y, as the library corrects the firmware's, measure against the camera's millimetres; the page's line says it as
information, not as an error. A pixel names the desk point the arm must be sent to. The arm
should be out of the picture, as CALIBRATE leaves it; lens distortion is not modelled.

**The live view** is the camera at its full size, the size every calibration is fitted at, only while a page is
open and in sight, or a snapshot wants it. QUALITY or FPS, on the camera page, is which of the camera's modes:
QUALITY its largest YUYV, uncompressed, the sharpest picture and slower, made JPEGs for the pages; FPS its largest
MJPG, its own JPEGs passed on, faster. The choice is there only when the YUYV is at least as large as the MJPG; QUALITY at first. Changing it, after
a warning, sets the calibration and everything in its pixels aside, as `*-previous`: CALIBRATE again. Its focus is
held still, since focusing moves the picture (the C920's by up to 4%): at the focus `camera.json` was calibrated
at (`lens`), for the same camera; with none, the camera focuses itself once, until its picture keeps still, and
CALIBRATE keeps where it stopped. Zoom is at its least, the widest view; pan and tilt at 0.
The camera's own JPEGs go to the page as they come. The snapshot is five fresh frames of the view averaged: nothing
pauses for it. The document camera's raw format, YUYV, shows the same picture, at 2 frames a second
(`work/STATUS-impala.md` §1).

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
it is, the arrows that move the cup 1 mm, in desk space, and RECORD POINT. A grid click comes over the point 60 mm
up, then slowly down to the page's notch, 42: 2 mm over a cube's top. Right the camera: CLICK THE DESK over it, and
apart, a helper, FIND RED CUBE; the live view, large, with the calibration drawn over it: the reach on the desk,
orange and red; the arm's target as the cup's 21 mm rim at its height, faint green, with a dashed line down to the
desk under it (GUIDES turns these off). Where the servos say the cup is, is on the grid; over the picture only
once FIND RED CUBE or ARM TO DESK has brought the cup down (below). Every rim is drawn its true size.

1. **FIND RED CUBE**, a helper. Set the red cube down anywhere in view. The arm parks the way CALIBRATE parks it,
   out of the picture; the camera finds the red cube in the live view and the arm comes over it at 54 mm over the
   desk, then slowly down to 42: 2 mm over its top. The cup should be centred on it. The live view shows the cube
   as found -- its edges thin pink, fainter where hidden, an X under its centre -- and its desk x and y on its top;
   the grid shows its footprint. Once the arm is down its servos are read: on the cube's top a small dot where the
   arm was sent, and the cup's 21 mm rim, yellow, where the servos say it is, both drawn on the top's plane so that
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
   point on the grid and in the table: `config/camera-refine-points.json`. Again in a new place, ten times or more,
   spread over where pieces will go -- the dashed circle on the grid and in the live view is the place in reach and in
   view furthest from every point so far. Each point says how far the arm's
   reading is from the cube as the camera sees it, across the desk (*XY*), and how far from 40 mm tall the camera
   sees the cube (*Z*, + taller): through the three cubes' calibration, *off at CALIBRATE* -- 0 for its three
   cubes, and for a later point how well CALIBRATE foresaw it -- and through the one SAVE would write, *after
   SAVE*, the arm laid on all the points at once: 0 for any three, from the fourth on how far a point's reading is
   from where the others put it. Green within 1 mm, yellow within 2, red beyond; the point's circle on the grid
   takes the colour of its XY, as on the desk page. One point far off while the others are not: in XY the cup was
   off the cube's centre or its edges were found wrong, in Z the snapshot's edges were not all the cube's; forget
   it and record it again. **SAVE CALIBRATION**, from four points on, writes `camera.json` with the camera fitted
   to the points' pictures and the arm laid on them, and redraws the camera page's check image with it, on the
   calibration's own frame; its line says how the camera measures the arm's distances. With no points left, it
   puts back the three cubes' own calibration.
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
| RECORD POINT           | keep where the arm says the cup is, park the arm, see the cube in the snapshot        |
| ×, CLEAR               | forget a point; all of them                                                           |
| SAVE CALIBRATION       | `camera.json`: the camera from the points' pictures, the arm laid on all the points   |

**How the red cube is found** (`camera/locate.py`). The camera is the calibration's, held still: a 40 mm cube
standing on the desk has one look for each place and turn, and the cube is moved and turned until it looks like
the frame's red blob -- its silhouette onto the blob's outline, then each edge it shows onto the frame's own edges,
as CALIBRATE fits the camera. Only the warm pixels make the blob, so a red cube standing against the green or the
blue one in the picture is found all the same. Something red that a cube standing there would not cover -- a red
sheet on the desk, a cube on its side or on another -- is refused, saying so. The live view is the calibration's
own size; a smaller one would do, the camera scaled to it, but one of another shape is refused.

**How the points refine the calibration** (`camera/refine.py`, `camera/sighting.py`). The three cubes put the
camera roughly in place; the points then make it, in two parts kept apart. A point is where the arm said the cup
was and the edges found round the cube in the snapshot -- the frame's own, so they hold for any camera. Through a
camera, those edges make a cube standing somewhere on the desk: moved and turned, its base on the desk, until its
edges lie along them. **The camera** -- focal length, principal point (held loosely to the frame's middle), where
it stands and which way it looks -- is fitted to the pictures alone: every point's cube a 40 mm cube, wherever its
edges put it, as CALIBRATE fits its three, the cubes' points 1 to 3 among them until forgotten. The arm has no say
in it. **The arm** is then laid on the cubes as that camera sees them by a turn, a shift and a stretch, the least
squares of how far each reading is from its cube: six numbers, which three points fit exactly, so a point's XY is
its reading against the others' from the fourth point on, and the stretch is the arm's measure against the
camera's, shown, not counted against any point. Z is the cube's height through the camera, made free: a
picture's own check. A point kept before the edges were has only the pixel under the cube's centre, as the camera
of its day saw it: it counts for the arm, not for the camera, and has no Z. The cubes' own calibration is only
where the fit starts. Points all in one corner do not fix a camera: SAVE refuses one whose focal length is more
than 15% from the cubes', or an arm stretched more than 12% or skewed more than 6 degrees. RECORD POINT
refuses a cup that is not over a cube's top (40 to 60 mm over the desk) and a cube the camera sees more than half
a cube from where the arm said the cup was; nothing is recorded then. The points belong to the camera as
CALIBRATE fitted it: a new CALIBRATE sets them aside, in `camera-refine-points-previous.json`, and writes its
three cubes as the new points 1, 2 and 3. A file from before the cubes were points gets them once, its points
numbered on after them. VALIDATE
CALIBRATION, on the camera page, compares a new frame with the three cubes' own calibration, which `camera.json`
keeps beside the refined one, so a refinement does not read as a camera moved.

**How a click becomes a desk point.** The calibration at desk height 0, inverted: a point of the picture is a ray,
and the desk is where it meets it. The page draws with the calibration as shares of the picture's width and
height, and the app converts the click it is sent with the library's camera space; the two agree. A click above
the horizon, or out of the arm's reach, is refused before anything moves.

## The Tiles page

`tiles/`. Where the 25 mm tiles are, their letters and which way up, which of them the arm can pick up, and a word of
them set down in the dropoff area.
The page is there once the camera page
has written `config/camera.json`; until then its tab is greyed and the page says so.

Left the view from above, with the work areas: **pickup**, green, the outer two thirds of the reach, both sides;
**dropoff**, blue, the inner third. The dashed lines are the line between
them and the arm's centre line. Right the camera, LOCATE TILES in its heading. Across the bottom the tiles' table.

1. **Set the tiles down**, letters up, where the camera sees them. A letter is a capital printed in black marker;
   on a letter that is another turned -- M and W -- a dot at its bottom right says which it is.
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
   letter not read -- the reader under 60% sure of it -- and **–**, ringed, a blank tile. The table has the letter,
   how sure the reader is, its next best capital, *dot* when the dot was found, and *up*: where the letter's top
   points, degrees from desk +x, starred when it reads the same turned (N, S, Z upside down; O, X any way). The
   letters, left to right, are said under the picture.
3. **Try a tile's centre.** A tile clicked, on the grid, over the picture full size, or its row's *go*: the cup
   comes over its centre at 14 mm over the desk, then slowly down to 6, 4 mm over the tile's top. It should be
   centred on it. A tile out of reach is refused, nothing moved.
4. **Set a word down.** Until letters are typed in the box beside PLACE, LOCATE TILES puts five there at random
   from the tiles ready to pick up; type any others, as many as there are ready tiles for. Letters typed stay, LOCATE
   after LOCATE; emptied, the box is offered letters again. PLACE checks them -- letters on ready tiles, read,
   each tile once, nothing found lying where they go -- and refuses, saying why, before the arm moves. Then each
   tile in turn: the cup over it, slowly down to the height in the grip box beside PLACE -- 1 mm to start, under the
   tile's top -- corrected there, grip, up; across to its place at 15 mm, turning it on the way, slowly down to 5
   mm, 3 mm over its top, corrected, let go, up. The places are along the dotted arc 130 mm from the base axis, 25
   mm in from the dropoff area's outer edge, 5 mm apart at their corners nearest the arm, centred on the arm's
   centre line: the word reads from the arm's left to its right, each letter's top away from the arm. A turn the cup
   cannot make after the grip, past 90 degrees, it makes half before. Each place is drawn, dashed, and filled once
   its tile is there. At the end -- or a refusal or failure on the way, but not RESET -- the arm parks and the tiles
   are found and read again in a new snapshot, as LOCATE TILES finds them. Where the tiles carried were stays
   drawn, dotted grey, the letter faint at its corner and a chevron to where its top pointed -- to put the tiles
   back by -- PLACE after PLACE, until LOCATE TILES; the box keeps its letters. RESET lets go of a tile held,
   wherever the arm is, and then LOCATE TILES before the next PLACE.

| control             | does                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------ |
| LOCATE TILES        | park the arm, take the snapshot at full size, find the tiles, read and place them    |
| PLACE               | the tiles with the letters typed to the dropoff arc, gripped at the grip box's z;   |
|                     | then park, and find and read the tiles again                                         |
| Areas               | the work areas and the arm's reach drawn on the desk, or not                         |
| FULL SIZE           | the picture as large as the window allows; the tile under the pointer is said        |
| a tile, *go*        | the cup over the tile's centre, down to 4 mm over its top                            |
| RESET               | stop, let go, lift clear of the desk and of what is on it, and home                  |

**How the tiles are found** (`applib/tiles/`). The snapshot is resampled onto the plane of the tiles' top faces,
2 mm up, where every tile is the same 25 mm square, perspective gone. Every colour edge votes for the centre of a
square it could be a side of, either side of it: a tile's four sides all vote for its centre, and the seam between
two touching tiles for both of theirs. A square is fitted to the edges round each place most voted for -- its
centre to a few hundredths of a millimetre, its turn to a tenth of a degree, its size from 25 to 27.5 mm, as a face
looks larger close to the camera -- and kept as a tile if each of its sides stands out from what lies just beyond
it. Nothing is assumed of the tiles' or the desk's colour or
brightness. A tile cut by the picture's edge, or behind the arm, is not looked for.

**How a letter is read** (`applib/letters/`). The tile's face is resampled square-on from the snapshot, along its
sides. Ink is what is darker than the face round it; a face with none is blank, and a small round blob of ink off a
corner of the letter is the dot, which puts the letter's bottom right there. A small network (`files/letters.tflite`,
int8, trained on drawn hand-printed capitals by `training/letters/train.py`) looks at the face at each of its four
quarter turns and says which capital it is if that turn is upright, or that it is not; the best wins, at the dot's
turn if there is one.

**How a tile is placed** (`applib/areas.py`). By its centre, in desk mm round the arm's base axis; the areas and the
2 mm are the app's own, for now, and the reach the library's: `can_reach()` at the tile's top and over it.

**How a word is set down** (`applib/word.py`, the library's `carry()`). Of several ready tiles with a letter, the
one whose letter needs the least turning goes; an N, an S or an O is turned to whichever of its ways up is nearest.
The library's `carry()` picks the tile up and sets it down, and knowing both ends, shares out a turn the cup cannot
make from centre.

| directory | holds                                                                                             |
| --------- | ------------------------------------------------------------------------------------------------- |
| `common/` | the web server, the arm's one thread (`worker.py`), how it moves (`motion.py`), the look, the     |
|           | view from above with the height beside it (`grid.js`, `view.css`), the height column of the       |
|           | pages with cubes on the desk (`height.js`), and for the pages with a camera: desk space to the    |
|           | picture and back (`shares.js`), the marks drawn over it (`marks.js`), the picture full size       |
|           | (`fullview.js`)                                                                                   |
| `desk/`   | the desk page (`index.html`, `desk.js`, `grid.js`) and its requests (`api.py`, `plane.py`)        |
| `camera/` | the camera page (`index.html`, `camera.js`, `grid.js`, `guide.js`, `guide.md`), its requests      |
|           | (`api.py`, `store.py`), the fit (`cubes.py`, `fit.py`, `edges.py`, `model.py`), the check image   |
|           | (`check.py`), the red cube found (`locate.py`), the refinement (`refine.py`), the live view and   |
|           | its snapshots (`feed.py`), which camera (`devices.py`)                                            |
| `tests/`  | Fine Tune (`index.html`, `tests.js`, `grid.js`, `view.js`, `refine.js`) and its requests          |
|           | (`api.py`, `refine.py`)                                                                           |
| `tiles/`  | Tiles (`index.html`, `tiles.js`, `grid.js`, `view.js`) and its requests (`api.py`); the tiles     |
|           | found, read and placed by `applib/`, outside the app                                              |
