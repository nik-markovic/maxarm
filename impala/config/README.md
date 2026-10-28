# config/

The calibration, as the library reads it: one matrix per file, and a line saying what it means; and what the
calibration pages keep to get there. Every file here but this one belongs to one rig, so none of them is in git
(`../.gitignore`).

| file                        | holds                                                   | written by                  |
| --------------------------- | ------------------------------------------------------- | --------------------------- |
| `desk.json`                 | `desk_to_arm`, 4x4: desk space to the arm's own x, y, z | `calibration/`, desk page   |
| `camera.json`               | `desk_to_pixel`, 3x4: desk space to camera pixels       | `calibration/`, camera page |
| `custom.json`               | `custom_to_desk`, 4x4: a frame of your own, if wanted   | you; there is none here     |
| `desk-points.json`          | the desk page's touch points, as the servos read them   | `calibration/`, desk page   |
| `desk-previous.json`        | the `desk.json` the last SAVE replaced                  | `calibration/`, desk page   |
| `camera-points.json`        | where the cup was over each cube's centre, desk space   | `calibration/`, camera page |
| `camera-frame.png`          | the frame CALIBRATE took: cubes in view, arm parked     | `calibration/`, camera page |
| `camera-check.jpg`          | the latest check image, as the camera page shows it     | `calibration/`, camera page |
| `camera-refine-points.json` | the red cube's points, which refine `camera.json`       | `calibration/`, tests page  |

Desk space is millimetres on the desk with z the height above its surface, 0 being the cup snug on it.
`maxarm/spaces.py` explains how each matrix is used.

## desk.json

Written by the desk page's SAVE. Arm space is the library's: true x, y and z (`maxarm/raw.py`).
There the desk is a plane: arm x and y are desk x and y, and the surface is at arm z = `tilt_x` x + `tilt_y` y +
`offset`. So

```
desk_to_arm = [[1,      0,      0, 0     ],
               [0,      1,      0, 0     ],
               [tilt_x, tilt_y, 1, offset],
               [0,      0,      0, 1     ]]
```

The file also lists the touch points it was fitted through, in arm space; the library ignores them.

**Without one, the default desk is used:** arm z 44, level -- where the desk is under an ideally placed arm, in true
z (`DEFAULT_DESK_Z_MM` in `maxarm/spaces.py`). Desk space works from the start; `desk.json` is for a desk that is not
quite level or not quite at 44. A `desk.json` with `joint_offsets_deg` (before 2026-10-20) is ignored the same way.

## camera-points.json

Written by the camera page's RECORD: for each cube, where the servos said the cup was when the cube stood centred
under it, in desk space -- x and y the arm's own, z the cup's height over the desk, about 44 at the notch -- and
where the cup had been sent. A cube not recorded is `null`. Under each x and y the cube's base is at desk height 0
and its top at 40: those, not the cup's height, are what the fit will use.

## camera.json

Written by the camera page's CALIBRATE, from the three cubes in `camera-frame.png`, standing where
`camera-points.json` says, and again by the tests page's SAVE CALIBRATION, refined by the red cube's points in
`camera-refine-points.json`. The library reads `desk_to_pixel`: desk (x, y, z) to camera pixels,

```
pixel ~ desk_to_pixel @ [x, y, z, 1]        # divide the first two by the third
```

z the height above the desk, so a pixel and a height name a point: the top of a 4 mm tile is that tile's pixel at
z 4. It maps desk space as the desk page defines it -- x and y the arm's own, which the library corrects to true
millimetres -- with the arm placed on the desk where its points put it: a pixel names the desk point the arm must be
sent to.

The rest is the fit behind it, which the camera page reads back to draw and to check by:

| key             | what it is                                                                                 |
| --------------- | ------------------------------------------------------------------------------------------ |
| `frame_size`    | width, height: the only frame size `desk_to_pixel` holds for                               |
| `true_to_pixel` | 3x4: the camera alone, in true mm (x, y, z up), square pixels, laid along the arm's x, y   |
| `desk_to_true`  | 2x2: the arm's x, y to true mm -- the board's mirror (a turn and stretch in older files)   |
| `cubes`         | each cube's desk x, y, its turn on the desk, degrees, and the pixel under its centre       |
| `edge_px`       | how far the frame's edges sat from the fitted cubes': median and 90%, pixels               |
| `refined`       | if refined: how many red cube points, and how far off they all are, median and worst, mm   |
| `cubes_alone`   | if refined: `true_to_pixel`, `desk_to_true` and `cubes` as the three cubes alone gave them |

`desk_to_pixel` is `true_to_pixel` with `desk_to_true` applied to x and y, scaled. The library's `Camera` refuses a
frame of another size (`check_frame()`).

CALIBRATE places the arm on the three cubes by a turn and a shift, the best for the three; a refinement places it
again, on the three cubes and every red cube point at once, and keeps the camera: `true_to_pixel` is the same
camera, its true millimetres turned and shifted onto the arm's, so that `desk_to_true` is the mirror alone.

The calibration holds only while the camera and the arm stay where they were. VALIDATE CALIBRATION says whether the
camera has moved.

## camera-refine-points.json

Written by the tests page's RECORD POINT: for each point, where the servos said the cup was when the red cube stood
centred under it, in desk space as in `camera-points.json`, where the cup had been sent, and where the camera then
saw the desk under the cube's centre, the arm parked: a pixel of a frame of `frame_size`, the live view's. Pixels,
not millimetres, so a refinement changes none of them. They belong to the camera as CALIBRATE fitted it: the next
CALIBRATE moves them to `camera-refine-points-previous.json`.
