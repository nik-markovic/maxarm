# config/

The calibration, as the library reads it: one matrix per file, and a line saying what it means; and what the
calibration pages keep to get there. Every file here but this one belongs to one rig, so none of them is in git
(`../.gitignore`).

| file                 | holds                                                       | written by                     |
| -------------------- | ----------------------------------------------------------- | ------------------------------ |
| `desk.json`          | `desk_to_arm`, 4x4: desk space to the arm's own x, y, z     | `calibration/`, desk page      |
| `camera.json`        | `desk_to_pixel`, 3x4: desk space to camera pixels           | the camera page's fit, to come |
| `custom.json`        | `custom_to_desk`, 4x4: a frame of your own, if wanted       | you; there is none here        |
| `desk-points.json`   | the desk page's touch points, as the servos read them       | `calibration/`, desk page      |
| `desk-previous.json` | the `desk.json` the last SAVE replaced                      | `calibration/`, desk page      |
| `camera-points.json` | where the cup was over each cube's centre, desk space       | `calibration/`, camera page    |
| `camera-frame.png`   | the frame CALIBRATE took: the cubes in view, the arm parked | `calibration/`, camera page    |
| `camera-check.jpg`   | the latest check image, as the camera page shows it         | `calibration/`, camera page    |

Desk space is millimetres on the desk with z the height above its surface, 0 being the cup snug on it.
`maxarm/spaces.py` explains how each matrix is used.

## desk.json

Written by the desk page's SAVE. Arm space is the library's: x and y the firmware's, z true z (`maxarm/raw.py`).
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

None yet: the camera page takes the frame (`camera-frame.png`), and fitting the camera to it is still to come;
`main.py` cannot run until it has written one. It will map desk space as the desk page defines it -- x and y the
arm's own -- so the arm's x/y error (x travels ~7% short, dolphin §5a) is the camera calibration's to absorb.

The calibration holds only while the camera and the arm stay where they were. A camera calibrated at one frame size
is wrong at another -- nothing checks that yet.
