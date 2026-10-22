"""The camera calibration: the arm over each cube, a point per cube, the frame they are fitted from.

| module      | what it owns                                                          |
| ----------- | --------------------------------------------------------------------- |
| `api`       | the page's requests: the arm's moves, the points, CALIBRATE and CHECK |
| `store`     | the files in `config/`                                                |
| `feed`      | the live view, and the snapshot it pauses for                         |
| `capture`   | the snapshot: the still frame at full size, which main.py uses too    |
| `devices`   | which camera: the owner's if it is plugged in, never a virtual one    |
| `index.html`, `camera.css`, `camera.js`, `grid.js`, `guide.js` | the page              |
| `guide.md`  | the page's GUIDE: plain text, bullets and stars, to edit as wanted    |

Fitting `camera.json` from the frame, and drawing the cubes' outlines on the
check image, are still to come.
"""
