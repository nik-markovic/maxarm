"""The calibration apps, served by `calibration.py` as one web app with a page each.

    ../.venv/bin/python calibration/calibration.py

| directory | holds                                                                    |
| --------- | ------------------------------------------------------------------------ |
| `common/` | the web server, the arm's one thread and how it moves, the shared look   |
| `desk/`   | the desk calibration: touch points, the plane, `config/desk.json`        |
| `camera/` | the camera snapshot, which main.py uses too; the camera page is to come  |

A package, rather than a plain directory, so that `calibration` imported from
the pilot root is this and not `calibration.py`: main.py imports `calibration.camera`.
"""
