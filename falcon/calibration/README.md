# calibration/ — the calibration web app

Spec: [`work/PILOT-falcon.md`](../../work/PILOT-falcon.md). One web app with a page per calibration; the desk is first.

```
../.venv/bin/python calibration/calibration.py                  # then open http://localhost:8080
../.venv/bin/python calibration/calibration.py --host 0.0.0.0   # on the board, for another machine's browser
../.venv/bin/python agenttools/serve-fake.py                    # the same page on a fake arm, nothing plugged in
```

It needs `numpy` and `pyserial` (`requirements.txt`); the web server is Python's own.
Connecting resets the board, which homes the arm: 4 s, sometimes 15.
This process is the only thing that talks to the arm. Ctrl-C holds the arm where it is and closes the port.

## The desk page

It finds where the desk is under the arm, as a plane, and writes it to `config/desk.json`.

1. **Touch the desk.** Click the grid to send the arm over a point. Bring the cup down with HEIGHT, then the fine
   sliders, until it sits snug on the desk -- with SUCTION on it should seal. RECORD keeps where the servos say the
   cup is, which is not always where it was sent.
2. **Spread the points.** At least three, spanning 200 mm in X and 150 in Y, and not in a row. From the fourth on,
   each point shows how far it sits off the plane through all of them: green within 1 mm, yellow within 2, red
   beyond. Touch a red one again, or delete it (×).
3. **SAVE.** The plane goes to `config/desk.json`, and the file it replaces to `desk-previous.json`. Points are
   true z, as the library reports them, so a level desk is a level plane. A `desk.json` saved before the library
   corrected heights is not used: SAVE again. Points recorded then are converted to true z when the page loads.
4. **Try it.** *desk* in the table sends the cup over that point in desk space, at the point test height: 4 mm,
   down to 0. The first comes down from above, like `grab()`; the next ones glide, slowly and level, so the gap
   between cup and desk can be watched all the way. Any other move in between, and the next one comes from above
   again. TEST INPUT comes down onto any desk point, grips, holds, lets go, and stays there.

| control                   | does                                                                           |
| ------------------------- | ------------------------------------------------------------------------------ |
| grid click                | over to that grid point plus X and Y fine; from low down, straight up first    |
| HEIGHT, Z fine, Z 0.1     | z is their sum: 34 to 209, then -4 to 4, then -1.0 to 1.0. Down to 30 as it is |
| LOW REACH                 | HEIGHT down to 12, the firmware's floor at full reach. The desk lifts the arm  |
| X fine, Y fine, nudge pad | offsets of up to a grid step, 1 mm at a time. The pad follows the view's turn  |
| Cup                       | the cup's servo, -90 to 90                                                     |
| RECORD                    | keeps where the servos say the cup is, read afresh; the target beside it       |
| *arm* in the table        | back to the point in arm space: up, over, slowly down                          |
| RESET                     | stop, let go, lift clear, and home with the cup centred                        |

**How it moves.** Below the hover height -- arm z 80, or 20 mm over the desk in desk space -- every move is slow,
10 mm/s; above it, fast. A move across the grid from low goes straight up first and stays at the hover height.
Sliders, nudges and the moves to a point end *precise*: the servos stop short of small moves, so the arm is read
and sent on until it is within a pulse, about 0.5 mm -- some 1.4 s a move. After each move the servos are read and
shown as *arm says*. Near the desk a precise move may press a cup that is already on it up to 3 mm, then stops.

**Which space.** Everything but the desk test is arm space with no calibration at all, so no older calibration can
keep the cup off the desk being measured. The desk test is desk space through the saved `desk.json`, which is what
every app will use.

| directory | holds                                                                                      |
| --------- | ------------------------------------------------------------------------------------------ |
| `common/` | the web server, the arm's one thread (`worker.py`), how it moves (`motion.py`), the look   |
| `desk/`   | the desk page (`index.html`, `desk.js`, `grid.js`) and its requests (`api.py`, `plane.py`) |
| `camera/` | `capture.py`, the camera snapshot `main.py` uses. The camera page is to come               |
