# config/

The calibration the tile finder reads. condor's `calibration/calibrate.py` made the first one from
three cubes; dolphin's `calibration/refine.py` refines it with grid paper.

| file                        | what it is                                                        |
| --------------------------- | ----------------------------------------------------------------- |
| `calibration.json`          | The mapping, grid-refined. `DeskMapping.load()` reads it.          |
| `calibration-condor.json`   | condor's, as it came. `refine.py` always starts from this one.     |
| `calibration-frame.png`     | condor's cube snapshot, lossless, so a re-fit matches.             |
| `calibration-check.jpg`     | condor's check drawing of that frame, through condor's mapping.    |
| `grid-check.jpg`            | The grid frame with the fitted lattice and the new mapping drawn on. |

Nothing here is hand-written. If a file in this directory is edited by hand, the calibration
that produced it no longer describes the desk, so re-run the calibration instead:
`./calibration/refine.py files/grid-1.png`.

`calibration.json` keeps everything condor's held -- two pixel-to-arm planes, the AACS zero plane
in the arm's own board z, the cube points -- so anything that reads condor's format still gets arm
coordinates. It adds two things, kept apart on purpose:

- `desk_camera`: 3x4, true desk millimetres (x, y, height) to pixels, with square pixels, from the
  grid paper. What vision works in.
- `desk_to_arm`: 3x3 affine, where the arm must be sent to reach a desk point, through the three
  cubes. The arm's own error; its x travels about 7% short.

The stored pixel-to-arm planes are the two composed. Its `about`, `height` and `desk` fields say
in one line each what the numbers mean.

The mapping is only valid while the camera and the arm hold still relative to each other.
Anything bumped means recalibrating: condor's cubes first, then the grid paper.

**Look at `grid-check.jpg` after every refinement.** Red is every tenth fitted grid line, 50 mm
apart, and should lie on the paper's own lines across the whole sheet -- a red line drifting off
the paper's towards one side is a lattice fit that has gone wrong. Cyan is a 20 mm grid in desk
millimetres along the arm's axes: where the desk frame is and which way it runs.
