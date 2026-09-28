# config/

Output of `calibration/calibrate.py`, read by the demo.

| file                     | what it is                                                       |
| ------------------------ | ---------------------------------------------------------------- |
| `calibration.json`       | The mapping. `DeskMapping.load()` reads it.                      |
| `calibration-frame.jpg`  | The snapshot it was solved from, so it can be re-fitted offline. |
| `calibration-check.jpg`  | The same frame with the detections and the arm's grid drawn on.  |

Nothing here is hand-written. If a file in this directory is edited by hand, the calibration
that produced it no longer describes the desk, so re-run the calibration instead.

`calibration.json` holds two pixel-to-arm transforms — one for the desk and one for the plane a
cube-height above it, which together turn a pixel into a position for something of any height —
the AACS zero plane in the arm's own board z, and every point pair it was fitted through, so a
later run can add a point rather than starting again (`calibrate.py --keep`). Its `about` and
`height` fields say in one line each what the numbers mean; nothing else needs this directory's
README to read it.

The mapping is only valid while the camera and the arm hold still relative to each other.
Anything bumped means recalibrating — see `work/STATUS-condor-prototype.md`.

**Look at `calibration-check.jpg` after every calibration.** The magenta wireframes say the
detector found the right things; the cyan grid is where the mapping thinks the arm's
millimetres are, and it is the fastest way to see a fit that has gone wrong.
