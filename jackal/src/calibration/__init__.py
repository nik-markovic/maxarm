"""The calibration apps, served by `calibration.py` as one web app with a page each.

    ../.venv/bin/python src/calibration/calibration.py

| directory | holds                                                                     |
| --------- | ------------------------------------------------------------------------- |
| `common/` | the web server, the arm's one thread, the arm's controls                  |
| `desk/`   | the desk calibration: touch points, the plane, `config/desk.json`         |
| `camera/` | the camera calibration: the cubes' points, the fit that writes            |
|           | `config/camera.json`, the check image, the refinement                     |
| `tests/`  | Fine Tune: the red cube found and the arm brought over it, the arm sent   |
|           | to a point of the desk clicked in the live view, the points that refine   |
| `tiles/`  | the 25 mm tiles found in the snapshot, read, and a word of them set down  |

Each page's HTML, scripts and styles are in the pilot's `web/`, a directory per
page. What the /IOTCONNECT app shares with these pages is in `applib/`.

A package, rather than a plain directory, so that `calibration` imported from
`src/` is this and not `calibration.py`.
"""
