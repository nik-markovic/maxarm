# base-cups/ -- how the 9 mm at the arm's base was measured (STATUS-gecko §4.5-4.6)

One-off analysis scripts behind STATUS-gecko §4.5-4.6. They read `config/` and write
scratch files to `/tmp/gecko-cal/` (create it first). The rough cup centres and visible angles in `rims.py` are
pixels placed by hand for the owner's camera pose of 2026-10-22: check them against `config/camera-frame.png`.

1. `rims.py`: the three visible suction-cup rims, edge points along rays from rough centres.
2. `rim-circles.py`: each rim fitted as a circle on the desk through `camera.json`; the cups' square and its centre.
   The centre printed there, desk (-7.7, 4.6) on the 11:43 calibration, is hard-coded in the two below.
3. `per-cube.py`: each cube's reach and bearing seen from that centre, in the base plate's own frame.
4. `backlash.py`: how far the arm's axis lands from the cups' centre if blue's bearing is corrected for play.
