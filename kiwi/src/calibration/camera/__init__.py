"""The camera page: the arm over each cube, a point per cube, and the camera fitted to them.

| module      | what it owns                                                          |
| ----------- | --------------------------------------------------------------------- |
| `api`       | the page's requests: the arm's moves, the points, CALIBRATE           |
| `check`     | the check image: the calibration drawn on a frame                     |
| `anchor`    | the arm laid on the camera by its base; the mesh; what does not fit   |
| `refine`    | SAVE: the arm's correction made again with every point                |

What the /IOTCONNECT app needs of the camera too is `applib/camera/`: which camera,
the live view, the cubes in a frame, the fit and `camera.json`, the red cube found.
The page itself -- `index.html`, `camera.css`, `camera.js`, `grid.js` and `help.md`,
its HELP -- is in `web/camera/`.
"""
