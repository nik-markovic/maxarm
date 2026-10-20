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
   library corrected heights is not used: SAVE again. Points recorded then are converted to true z when the page
   loads.
4. **Try it.** *desk* in the table sends the cup over that point in desk space, at the point test height: 4 mm,
   down to 0. The first comes down from above, like `grab()`; the next ones glide, slowly and level, so the gap
   between cup and desk can be watched all the way. Any other move in between, and the next one comes from above
   again. TEST INPUT comes down onto any desk point, grips, holds, lets go, and stays there. Without a `desk.json`
   all of this uses the default desk.

| control                   | does                                                                            |
| ------------------------- | ------------------------------------------------------------------------------- |
| grid click                | over to that grid point (10 or 20 mm grid); X and Y fine go back to 0           |
| height slider             | 80 down to 40, 1 mm steps; yellow from 44 to 42, red below: may touch the desk  |
| 120 SAFE                  | up to 120; green while the arm is that high or higher, as after RESET           |
| Z+, Z−                    | 1 mm up or down from the height sent; the label shows that height               |
| ▼ 2 mm                    | the slider goes 2 mm lower, as often as wanted; only the firmware stops it      |
| X fine, Y fine, nudge pad | offsets of up to a grid step, 1 mm at a time. The pad follows the view's turn   |
| TRUE / RAW                | arm z as the library has it, or the firmware's own; the ? says why they differ  |
| Cup                       | the cup's servo, -90 to 90                                                      |
| RECORD TOUCH POINT        | keeps where the servos say the cup is, read afresh; the target beside it        |
| *arm* in the table        | back to the point in arm space: up, over, slowly down                           |
| RESET                     | stop, let go, lift clear, and home with the cup centred                         |

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

| directory | holds                                                                                      |
| --------- | ------------------------------------------------------------------------------------------ |
| `common/` | the web server, the arm's one thread (`worker.py`), how it moves (`motion.py`), the look   |
| `desk/`   | the desk page (`index.html`, `desk.js`, `grid.js`) and its requests (`api.py`, `plane.py`) |
| `camera/` | `capture.py`, the camera snapshot `main.py` uses. The camera page is to come               |
