"""The camera page: the arm over each cube, a point per cube, and the camera fitted to them.

| module      | what it owns                                                          |
| ----------- | --------------------------------------------------------------------- |
| `api`       | the page's requests: the arm's moves, the points, CALIBRATE, VALIDATE |
| `check`     | the check image: the calibration drawn on a frame                     |
| `refine`    | the camera fitted again to the red cube's points, for SAVE            |

What the /IOTCONNECT app needs of the camera too is `applib/camera/`: which camera,
the live view, the cubes in a frame, the fit and `camera.json`, the red cube found.
The page itself -- `index.html`, `camera.css`, `camera.js`, `grid.js`, `guide.js` and
`guide.md`, the GUIDE's plain text -- is in `web/camera/`.
"""
