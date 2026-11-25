# config/

Everything here but this file belongs to one rig, so none of it is in git (`../.gitignore`): the calibration as the
library reads it, what the calibration pages keep to get there, the app's settings, and the device's identity for
/IOTCONNECT. A newer package unpacked over the old leaves this folder alone.

| file                        | holds                                                   | written by                   |
| --------------------------- | ------------------------------------------------------- | ---------------------------- |
| `desk.json`                 | `desk_to_arm`, 4x4: desk space to the arm's own x, y, z | you, if wanted               |
| `camera.json`               | `desk_to_pixel`, 3x4: desk space to camera pixels       | Camera page, Fine Tune       |
| `camera-points.json`        | where the cup was over each of the three cubes          | Camera page                  |
| `camera-frame.png`          | the frame CALIBRATE took, the arm parked                | Camera page                  |
| `camera-check.jpg`          | the latest check picture                                | Camera page, Fine Tune       |
| `camera-refine-points.json` | Fine Tune's points                                      | Camera page, Fine Tune       |
| `camera-view.json`          | QUALITY or FPS: the camera's mode                       | Camera page                  |
| `arm-true.json`             | the arm's raw-to-true correction the points make        | Camera page, Fine Tune       |
| `debug/`                    | each point's snapshot and its cube as found             | Fine Tune                    |
| `appcfg.json`               | the app's settings that differ from their defaults      | Tiles page; the app          |
| `custom.json`               | `custom_to_desk`, 4x4: a frame of your own, if wanted   | you                          |
| `iotcDeviceConfig.json`     | the device, as /IOTCONNECT knows it                     | you: downloaded              |
| `device-cert.pem`           | the device's certificate                                | you, or `scripts/install.sh` |
| `device-pkey.pem`           | its private key                                         | you, or `scripts/install.sh` |

What a page sets aside, a calibration for another camera mode say, is kept as `*-previous`. Desk space is
millimeters on the desk, z the height above it, 0 the cup resting on it; `maxarm/spaces.py` says how each matrix
is used.

## desk.json

The desk is a plane in the arm's true x, y, z: desk x and y are the arm's, and the surface is at
arm z = `tilt_x` x + `tilt_y` y + `offset`:

```
desk_to_arm = [[1,      0,      0, 0     ],
               [0,      1,      0, 0     ],
               [tilt_x, tilt_y, 1, offset],
               [0,      0,      0, 1     ]]
```

Without the file the library uses the default desk, level at arm z `DEFAULT_DESK_Z_MM` (`maxarm/spaces.py`), which
is right for a level desk under a well-placed arm, held flat by its base's suction cups. Nothing in the demo writes
the file: it is for a desk that is not flat.

## camera.json

```
pixel ~ desk_to_pixel @ [x, y, z, 1]        # divide the first two by the third
```

z is the height above the desk, so a pixel and a height name a point. Desk x and y are the arm's, so a pixel names
the desk point the arm must be sent to. The rest is the fit behind it, which the pages read back:

| key             | what it is                                                                              |
| --------------- | --------------------------------------------------------------------------------------- |
| `lens`          | the camera as it calls itself, the focus it was held at, and its mode                   |
| `frame_size`    | the only frame size `desk_to_pixel` holds for                                           |
| `true_to_pixel` | 3x4: the camera alone, in true mm along the arm's x and y, square pixels                |
| `desk_to_true`  | 2x2: the arm's x, y to true mm                                                          |
| `arm_stretch`   | 2x2: how the arm's x, y measure against the camera's; reported, not used                |
| `cubes`         | each of the three cubes: desk x, y, its turn on the desk, and the pixel under its center |
| `edge_px`       | how far the frame's edges sat from the fitted cubes', pixels: median and 90%            |
| `anchor`        | how the arm is laid on the camera: its axis by the base's nuts, its turn by the points  |
| `refined`       | after Fine Tune's SAVE: how many points, and how far each is off through the others     |
| `cubes_alone`   | after Fine Tune's SAVE: the fit from the three cubes alone                              |

`desk_to_pixel` is `true_to_pixel` with `desk_to_true` applied to x and y. The library's `Camera` refuses a frame of
another size. The calibration holds only while the camera and the arm stay where they were.

## camera-points.json, camera-refine-points.json

For each point: where the servos said the cup was over the cube, in desk space, and where the cup had been sent.
Fine Tune's points also keep how the camera saw the cube, the arm parked: `pixel`, under the cube's center, and
`edges`, the edges found round it, each with the `corners` it joins and the pixels `found` on it. `cube` names
which of the three cubes a point is; `null` for the red cube's. They are pixels, not millimeters, so a new SAVE
changes none of them.

## arm-true.json

The arm's correction on top of the library's formula (`maxarm/converter.py`), read again whenever the file changes:
per point, the formula's bearing and reach, and how much further out and further round the camera saw the cube.
`maxarm/mesh.py` makes of them one pattern over the whole desk plus what it leaves at each point. No file: the
formula alone.

## appcfg.json

The settings the Tiles page and the /IOTCONNECT app share (`src/applib/settings.py`). Only what differs from the
defaults is written; either side reads it again when it changes.

| setting          | default | what it is                                                                     |
| ---------------- | ------- | ------------------------------------------------------------------------------ |
| `tile_height_mm` | 2.0     | how thick the tiles are, mm: 1 to 8                                            |
| `tile_offset_mm` | 1.0     | how far under a tile's top the cup grips it, mm: 1 over the top to 3 in        |
| `reader`         | `back`  | where the person reading a word stands: `back`, `front`, `left`, `right`       |

## iotcDeviceConfig.json, device-cert.pem, device-pkey.pem

The device for /IOTCONNECT: `iotcDeviceConfig.json` as downloaded from the device's page, and its certificate and
private key under these two names. The certificate is either the one /IOTCONNECT made, from the device's
credentials zip, or the one `scripts/install.sh` makes here when there is no key yet.
