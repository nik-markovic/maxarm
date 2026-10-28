"""The owner's points from above, each miss after the 4-number fit drawn 20 times longer; the camera's foot and the
arm's base marked. Both sessions side by side. No arm."""
import sys

import cv2
import numpy as np

sys.path.insert(0, '.')
from loo import predict  # noqa: E402
from points import load  # noqa: E402

panels = []
for cfg, title in (("/var/b/shared/work/maxarm/work/config-old", "gecko, 23 Oct evening (19 points)"),
                   ("/var/b/shared/work/maxarm/impala/config", "hyena, 27 Oct (17 points)")):
    fit, rows = load(cfg)
    arm = np.array([(r['x'], r['y']) for r in rows])
    cam = np.array([r['cam'] for r in rows])
    # misses in the arm's frame: fit the camera onto the arm (the inverse direction) with the 4-number map
    mapped = predict(arm, cam, arm, "similarity")
    A = np.column_stack((arm, np.ones(len(arm))))
    linear = np.linalg.lstsq(A, mapped, rcond=None)[0][:2].T        # the fit's own 2x2, mirror included
    miss = (cam - mapped) @ np.linalg.inv(linear).T                   # in the arm's mm and directions
    S, W, H = 1.6, 900, 760                     # px per mm, canvas
    origin = np.array([W / 2, 160])

    def at(p):
        return tuple(np.round(origin + np.array([p[0], -p[1]]) * S).astype(int))   # desk y up the page: arm at top

    img = np.full((H, W, 3), 255, np.uint8)
    for r_ in (95, 150, 200, 270):
        cv2.circle(img, at((0, 0)), int(r_ * S), (220, 220, 220), 1, cv2.LINE_AA)
    cv2.rectangle(img, at((-60, 60)), at((60, -60)), (180, 180, 180), 1)
    foot = fit.camera_position[:2]
    cv2.drawMarker(img, at(foot), (0, 120, 0), cv2.MARKER_DIAMOND, 16, 2)
    cv2.putText(img, "camera", at(foot + (8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1)
    for r, p, m in zip(rows, arm, miss):
        cv2.circle(img, at(p), 3, (0, 0, 0), -1)
        cv2.arrowedLine(img, at(p), at(p + 20 * m), (0, 0, 230), 2, cv2.LINE_AA, tipLength=0.15)
        cv2.putText(img, f"{r['name']} {np.hypot(*m):.1f}", at(p + (4, 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (60, 60, 60), 1)
    cv2.putText(img, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(img, "red: where the camera saw the cube minus where the arm said, x20, after a turn+scale+shift fit",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(img, "arm's base at top (desk y up the page, x to the right); circles 95, 150, 200, 270 mm",
                (10, H - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    panels.append(img)
cv2.imwrite("residual-map.png", np.hstack(panels))
