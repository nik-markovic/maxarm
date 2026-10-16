# config/

The calibration, as the library reads it: one matrix per file, and a line saying what it means.

| file          | holds                                                 | written by                            |
| ------------- | ----------------------------------------------------- | ------------------------------------- |
| `desk.json`   | `desk_to_arm`, 4x4: desk space to the arm's own x, y, z | the tilt calibration (next pilot)     |
| `camera.json` | `desk_to_pixel`, 3x4: desk space to camera pixels     | the camera calibration                |
| `custom.json` | `custom_to_desk`, 4x4: a frame of your own, if wanted | you; there is none here               |

Desk space is millimetres on the desk with z the height above its surface, 0 being the cup snug on it.
`maxarm/spaces.py` explains how each matrix is used.

**Where these two came from.** Neither calibration has been redone yet, so both are dolphin's
(`dolphin/config/calibration.json`), converted by `agenttools/convert-calibration.py`:

- `desk.json` is condor's three measured desk heights -- the bare desk under each cube, which is the tilt -- composed
  with dolphin's grid-paper x/y correction (the arm's x travels about 7% short). It stands in for the tilt
  calibration until there is one.
- `camera.json` is dolphin's grid-paper camera: square pixels, true millimetres, 2592x1944 frames.

The conversion was checked against dolphin's own `pixel_to_board()` over a grid of pixels at four heights: they
agree to 3e-13 mm. The cube points, the two plane transforms and the AACS plane are gone; nothing reads them.

The calibration holds only while the camera and the arm stay where they were. A camera calibrated at one frame size
is wrong at another -- nothing checks that yet.
