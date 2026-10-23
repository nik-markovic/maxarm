"""The calibration apps, served by `calibration.py` as one web app with a page each.

    ../.venv/bin/python calibration/calibration.py

| directory | holds                                                                     |
| --------- | ------------------------------------------------------------------------- |
| `common/` | the web server, the arm's one thread and how it moves, the shared look    |
| `desk/`   | the desk calibration: touch points, the plane, `config/desk.json`         |
| `camera/` | the camera calibration: the cubes' points, the live view, the snapshot    |
|           | (which main.py uses too), and the fit that writes `config/camera.json`    |
| `tests/`  | the tests of it: the red cube found and the arm brought over it, and the  |
|           | arm sent to a point of the desk clicked in the live view                  |

A package, rather than a plain directory, so that `calibration` imported from
the pilot root is this and not `calibration.py`: main.py imports `calibration.camera`.
"""
