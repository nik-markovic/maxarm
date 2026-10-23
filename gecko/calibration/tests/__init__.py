"""The tests: whether the camera calibration brings the arm to what the camera sees; and the red cube's points that
refine it.

| module      | what it owns                                                                     |
| ----------- | -------------------------------------------------------------------------------- |
| `api`       | the page's requests: FIND RED CUBE, ARM TO DESK, the points, SAVE CALIBRATION;   |
|             | the calibration to draw by                                                       |
| `refine`    | the points as the page shows them, and the calibration SAVE writes               |
| `index.html`, `tests.css`, `tests.js`, `grid.js` | the page                                    |
| `view.js`   | the live view's marks, drawn over it, and the live view full size to click       |
| `shares.js` | desk space to the live view and back, through the calibration                    |
| `refine.js` | the points' table, what SAVE would change, and a place for the next point        |

The red cube is found by `camera/locate.py`, with the camera as the calibration
has it, and the calibration refined by `camera/refine.py`; the arm's moves over
the cubes are the camera page's.
"""
