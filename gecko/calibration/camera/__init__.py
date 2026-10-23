"""The camera calibration: the arm over each cube, a point per cube, and the camera fitted to them.

| module      | what it owns                                                          |
| ----------- | --------------------------------------------------------------------- |
| `api`       | the page's requests: the arm's moves, the points, CALIBRATE, VALIDATE |
| `store`     | the files in `config/`, `camera.json` among them                      |
| `cubes`     | the cubes in a frame: coloured, cube-shaped blobs, and their names    |
| `fit`       | the camera that sees the cubes where the arm says they stand          |
| `model`     | the pinhole camera, the 40 mm cube, and the least-squares solver      |
| `check`     | the check image: the calibration drawn on a frame                     |
| `edges`     | where a cube drawn through the camera meets the frame's edges         |
| `locate`    | the red cube, found by the calibrated camera                          |
| `refine`    | the arm's x/y error fitted again, to the red cube's points            |
| `feed`      | the live view, and the snapshot it pauses for                         |
| `capture`   | the snapshot: the still frame at full size, which main.py uses too    |
| `devices`   | which camera: the owner's if it is plugged in, never a virtual one    |
| `index.html`, `camera.css`, `camera.js`, `grid.js`, `guide.js` | the page              |
| `guide.md`  | the page's GUIDE: plain text, bullets and stars, to edit as wanted    |
"""
