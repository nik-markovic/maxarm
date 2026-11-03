"""The Fine Tune page: whether the camera calibration brings the arm to what the camera sees; and the red cube's
points that refine it.

| module      | what it owns                                                                     |
| ----------- | -------------------------------------------------------------------------------- |
| `api`       | the page's requests: FIND RED CUBE, ARM TO DESK, the points, SAVE CALIBRATION;   |
|             | the calibration to draw by, and the red cube found, drawn                        |
| `refine`    | the points as the page shows them, and the calibration SAVE writes               |

The red cube is found and the arm brought over it by `applib/redcube.py`, which the
/IOTCONNECT app's `find-cube` runs too, and drawn by `applib/overlay/cube.py`; the
calibration is refined by `camera/refine.py`. The page is in `web/tests/`.
"""
