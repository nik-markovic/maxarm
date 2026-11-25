# calibration/ — the calibration pages

One web app with a page per step: Desk, Camera, Fine Tune, Tiles and Chat. It is served by the demo's one program,
`src/main.py`, beside /IOTCONNECT; `calibration.py` builds the pages on the program's arm worker, camera, tiles and
model. The web server is Python's own (`common/server.py`): it serves `web/` and answers `/api/<page>/...`, each
page's requests in its own package, listed in its `api.py` docstring.

What the person at the arm reads is in the pages and their `help.md`, not here.

## Shared by every page

- **The arm.** Every move goes through the arm's one thread (`applib/worker.py`), shared with /IOTCONNECT: one
  command at a time, whichever asks. Connecting resets the board, which homes the arm; a failed or lost arm is tried
  again every `motion.RECONNECT_S`, and CONNECT tries at once. `common/arm_api.py` has the controls the pages share:
  grid moves, height, nudges, suction, RESET.
- **Arm says.** After a move the servos are read until steady, and their mean is what the page shows as *arm says*.
  The servos stop 1-3 mm short of small moves and nothing corrects that, so every RECORD keeps what they say, never
  the target. RECORD and SAVE wait for the reading.
- **Help.** HELP opens `web/<page>/help.md`, rendered by `web/common/help.js`, a small subset of markdown described
  at its top. Fine Tune's two checks have their own, `help-touch.md` and `help-find.md`. The camera page opens its
  help on the first visit. The intro line under the header, and its dismissal, are in `web/common/page.js`.
- **Messages.** A request that fails answers a plain sentence naming the cause, for a toast or the page; the
  technical line, when there is one, goes to the console.
- **Order.** Fine Tune, Tiles and Chat wait for `config/camera.json`: their tabs are grayed until it exists.

## Desk

The arm moved by hand, in its own coordinates, to get to know it. It has no requests of its own: only the shared
arm controls, and the cup's height over the desk in the state (`arm_api.desk_height()`). The desk is the library's
default, level at `DEFAULT_DESK_Z_MM`, which is right when the base's suction cups hold the base flat; a
`config/desk.json` put there by hand is used instead.

- TRUE/RAW changes only the numbers shown; everything sent is true. The page converts with the library's constants
  from the limits (`true_z`, `true_xy`).

## Camera (`camera/`)

RECORD keeps where the cup is over each of the three cubes (`camera-points.json`). CALIBRATE parks the arm out of
the picture, averages five fresh frames, finds the cubes (`applib/camera/cubes.py`) and fits the camera to them
(`applib/camera/fit.py`). It then finds the arm's base in the frame by its nuts and lays the arm on the camera
(`anchor.py`): its axis where the nuts are, its turn from the cubes' bearings. It writes `camera.json`,
`camera-frame.png` and `arm-true.json`; `check.py` draws the check picture.

- Every decision is measured in the frame itself, so another camera, light or desk needs nothing retuned.
- The arm's green base can look as green as the green cube, so which blob is which cube is settled by where the
  arm recorded each.
- The camera is fitted to every visible edge of the cubes, each standing where the arm read the cup over it: the
  arm's distances, in true mm, are the camera's measure, and the cubes' size their shape. How the arm's x and y
  measure against the camera (`arm_stretch`) is reported, not used. A cube more than `anchor.DOES_NOT_FIT_MM` off
  where the other two put it is left out. Lens distortion is not modeled.
- QUALITY is the camera's largest YUYV mode, FPS its largest MJPG. Changing it sets the calibration aside, as
  `*-previous`. Focus is held still at the focus `camera.json` was made at, since focusing moves the picture; zoom
  is at its widest and backlight compensation off. The camera is the first USB capture device, found at each start
  (`applib/camera/devices.py`).
- A new CALIBRATE sets Fine Tune's points aside (`camera-refine-points-previous.json`) and writes its three cubes
  as points 1, 2 and 3.

## Fine Tune (`tests/`)

Points that make the calibration good all over the desk, and two checks to find where it is not.

- **RECORD POINT** keeps where the servos say the cup is over the red cube, parks the arm, and finds the cube's
  edges in a snapshot (`applib/camera/sighting.py`). `record.py` draws the zoomed picture for the dialog; REDO
  looks again in the same snapshot with its contrast, gamma and sharpness tweaked. A cup not over a cube's top, or a
  cube seen more than half a cube away, is refused.
- **SAVE CALIBRATION** (`refine.py`, `camera/refine.py`) fits the whole camera again to every point's bottom and
  vertical edges (a top's edges are found inside the cube), each cube standing where the arm read it. It lays the
  arm again by the base in CALIBRATE's frame, writes `arm-true.json` (the pattern and mesh in `maxarm/mesh.py`,
  exact at the points) and `camera.json`, and redraws the check picture. It needs `MIN_POINTS`, and refuses points
  that do not lay the arm on the camera. Points not saved are not used by the arm.
- **The points' numbers.** *Off now* is through the saved correction, which is exact at its points, so a saved
  point reads near 0. *Without it* is the point left out of the fit: how well the calibration does between points.
- **ARM TO DESK.** A click on the picture is a ray, and the desk point is where it meets height 0. The page draws
  with the calibration as shares of the picture (`web/common/shares.js`); the server converts with the library's
  camera space; the two agree. From a cube's height the arm parks first, since straight across from there strains
  it.
- **FIND RED CUBE** is the app's `find-cube`: `applib/camera/locate.py` moves and turns a model cube until its
  silhouette and edges match the red blob, and refuses red that a standing cube would not cover;
  `applib/redcube.py` brings the cup over it.

## Tiles (`tiles/`)

The app's `detect` and `place`, by the same code (`applib/tilework.py`). The overlay is drawn in Python
(`applib/overlay/tiles.py`); the page draws the tile under the pointer and the arm over it.

- **Tiles** (`applib/tiles/`). The snapshot is resampled onto the plane of the tiles' tops, where every tile is the
  same square. Every color edge votes for the center of a square it could be a side of; a square is fitted round
  each well-voted place and kept if its sides stand out from what lies beyond. Nothing is assumed of the tiles' or
  the desk's color. The fits run on the board's other cores (`applib/crew.py`).
- **Letters** (`applib/letters/`). Each face is resampled square-on. Ink is what is darker than the face round it;
  a small round blob off the letter's corner is the dot that tells M from W. A small network (`files/letters.tflite`,
  from `training/letters/`) looks at the face at each quarter turn; the best wins.
- **Placing** (`applib/areas.py`, `applib/word.py`). The areas and the gap between tiles are the app's; the reach is
  the library's `can_reach()`. Of several tiles with a letter, the one that needs the least turning goes; the
  library's `carry()` shares out a turn the cup cannot make from center.
- The tile height, tile offset and reader are the app's settings (`applib/settings.py`), kept in `appcfg.json` and
  used by /IOTCONNECT too.

## Chat (`chat/`)

The app's `agent` command, by the same service (`applib/agent/`). The tiles are the Tiles page's.

- The model only says what a request wants (`find_word`): the longest word, one for a clue, or the visitor's own.
  Code finds the words the tiles spell (`dictionary.py`), since the model cannot count letters, and plans the word
  as PLACE would. For a clue or a length, a short list is put to the model, which names one, or none
  (`finder.py`). A tool call the model writes out as text is made all the same (`repair.py`).
- The model is told what happened before in one line, the last word and what became of it, never the conversation.
  NEW VISITOR, or `chat.VISIT_QUIET_S` without a request, starts that again.
- One request at a time. Without the connector, or before the model is ready, the page says where the model is.

## Where things are

| path                   | holds                                                                               |
| ---------------------- | ----------------------------------------------------------------------------------- |
| `calibration.py`       | the pages, built on the program's arm, camera, tiles and model                      |
| `common/`              | the web server; the arm controls every page shares (`arm_api.py`)                   |
| `camera/`              | CALIBRATE, the arm laid on the camera (`anchor.py`), the check picture, the refit   |
| `tests/`               | Fine Tune: points, SAVE, the two checks                                             |
| `tiles/`, `chat/`      | the Tiles and Chat pages                                                            |
| `web/common/`          | the look (`page.css`, `page.js`), the help, the view from above and the height      |
|                        | beside it (`grid.js`, `view.css`, `height.js`), desk space to the picture and back  |
|                        | (`shares.js`), marks over it (`marks.js`), the Python-drawn layer (`layer.js`), the |
|                        | picture full size (`fullview.js`)                                                   |
| `web/<page>/`          | each page's `index.html`, script, styles, grid and `help.md`                        |
