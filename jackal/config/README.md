# config/

The calibration, as the library reads it: one matrix per file, and a line saying what it means; what the
calibration pages keep to get there; the app's settings; and the device's identity for /IOTCONNECT. Every file here
but this one belongs to one rig, so none of them is in git (`../.gitignore`).

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
| `camera-refine-points.json` | the points which refine `camera.json`                   | `calibration/`, both pages  |
| `camera-view.json`          | QUALITY or FPS: the camera's mode for the live view     | `calibration/`, camera page |
| `appcfg.json`               | the app's settings that differ from their defaults      | the Tiles page; the app     |
| `iotcDeviceConfig.json`     | the device, as /IOTCONNECT knows it                     | you: downloaded             |
| `device-cert.pem`           | the device's certificate                                | you, or `scripts/install.sh`|
| `device-pkey.pem`           | its private key                                         | you, or `scripts/install.sh`|

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
`camera-points.json` says, and again by the tests page's SAVE CALIBRATION, the camera fitted to the points in
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
| `lens`          | the camera as it calls itself, its focus (`focus_absolute`) held still, and its mode       |
| `frame_size`    | width, height: the only frame size `desk_to_pixel` holds for                               |
| `true_to_pixel` | 3x4: the camera alone, in true mm (x, y, z up), square pixels, along the arm's x, y        |
| `desk_to_true`  | 2x2: the arm's x, y to true mm -- the board's mirror                                       |
| `arm_stretch`   | 2x2: the stretch a turn, shift and stretch would take: the arm's measure, not used         |
| `cubes`         | each cube's desk x, y, its turn on the desk, degrees, and the pixel under its centre       |
| `edge_px`       | how far the frame's edges sat from the fitted cubes': median and 90%, pixels               |
| `refined`       | if refined: how many points, how far each arm reading is from where the others put it      |
|                 | (`off_mm`), and their cubes' tops from 40 mm (`height_mm`): median and worst, mm           |
| `cubes_alone`   | if refined: the matrices and `cubes` as the three cubes alone gave them                    |

`desk_to_pixel` is `true_to_pixel` with `desk_to_true` applied to x and y, scaled. The library's `Camera` refuses a
frame of another size (`check_frame()`).

The camera is fitted with each cube standing where the arm read the cup over it, its millimetres taken as real:
CALIBRATE's three cubes, a refinement's points. The arm's distances between them are the camera's measure, the
cubes' 40 mm their shape; no turn or shift lies between the two. The mirror is in `desk_to_true`; `arm_stretch` is
how the arm's x, y measure against the cubes where their own edges put them.
A refinement fits the whole camera again -- focal length, principal point, where it stands and which way it
looks -- with every point's cube, 40 mm, where the arm read it. The three cubes take part as points 1 to 3, unless
forgotten; their own calibration is only where the fit starts.

The calibration holds only while the camera and the arm stay where they were. VALIDATE CALIBRATION says whether the
camera has moved.

## camera-refine-points.json

Written by CALIBRATE, its three cubes as points 1, 2 and 3, and the tests page's RECORD POINT, the red cube's:
for each point, where the servos said the cup was when the cube stood centred under it, in desk space as in
`camera-points.json`, where the cup had been sent, and how the camera then saw the cube, the arm parked, in a frame
of `frame_size`: `pixel`, the desk under the cube's centre, and `edges`, the edges found round the cube -- for each,
the `corners` it joins and the pixels `found` on it, x, y after x, y -- numbered as the cube is turned by
`turn_deg` (true mm). SAVE fits the camera to the edges; a point with none, from before they were kept, counts by
its pixel. `cube` names the camera page's cube a point is, seen in `camera-frame.png`; `null` for the red cube's
points. A point forgotten on the tests page is gone, a cube's too. Pixels, not millimetres, so a refinement changes
none of them. They belong to the camera as CALIBRATE fitted it: the next CALIBRATE moves them to
`camera-refine-points-previous.json`. A file from before the cubes were points gets them once, the next time the
app reads it, its own points numbered on after them.

## appcfg.json

The app's own settings, which the calibration app's Tiles page and the /IOTCONNECT app share (`src/applib/settings.py`):

| setting   | default | what it is                                                                           |
| --------- | ------- | ------------------------------------------------------------------------------------ |
| `grip_mm` | 1.0     | the desk height the cup grips a tile at, mm: 0 to 6                                   |
| `reader`  | `back`  | where the person reading a word stands, round the arm: `back`, `front`, `left`, `right` |

Only what differs from the defaults is written; a setting set back to its default is taken out, and with none left
the file goes. Either app writes it as soon as a setting changes -- the Tiles page's grip box and reader, the app's
`reader-location` command -- and reads it again when it has changed.

## iotcDeviceConfig.json, device-cert.pem, device-pkey.pem

The device for the /IOTCONNECT app (`src/main.py`): `iotcDeviceConfig.json` as downloaded from the device's page in
/IOTCONNECT, and its certificate and private key under these two names. Either the certificate /IOTCONNECT made, from
the device's credentials zip, or one `scripts/install.sh` makes here when there is no key yet, pasted into the device
when it is created.
