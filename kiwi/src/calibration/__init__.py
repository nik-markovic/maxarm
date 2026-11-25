"""The calibration pages, served by `calibration.py` as one web app with a page each, from `../main.py` beside the
/IOTCONNECT app (`--no-webapp` without them).

| directory | holds                                                                     |
| --------- | ------------------------------------------------------------------------- |
| `common/` | the web server, the arm's controls                                        |
| `camera/` | the camera calibration: the cubes' points, the fit that writes            |
|           | `config/camera.json`, the check image, the refinement                     |
| `tests/`  | Fine Tune: the red cube found and the arm brought over it, the arm sent   |
|           | to a point of the desk clicked in the live view, the points that refine   |
| `tiles/`  | the 25 mm tiles found in the snapshot, read, and a word of them set down  |
| `chat/`   | a request to the language model on the Ara240, which sets a word down     |

Each page's HTML, scripts and styles are in `web/`, a directory per
page. What the /IOTCONNECT app shares with these pages is in `applib/`.

A package, rather than a plain directory, so that `calibration` imported from
`src/` is this and not `calibration.py`.
"""
