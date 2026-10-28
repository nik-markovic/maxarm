"""Each place predicted from a fit to the others, per model: affine (camera.json's 6 numbers) against perspective
(8 numbers), on both sessions. No arm."""
import sys

import cv2
import numpy as np

sys.path.insert(0, '.')
from points import load  # noqa: E402


def predict(src_train, dst_train, src_test, kind):
    if kind == "affine":
        A = np.column_stack((src_train, np.ones(len(src_train))))
        sol = np.linalg.lstsq(A, dst_train, rcond=None)[0]
        return np.column_stack((src_test, np.ones(len(src_test)))) @ sol
    if kind == "similarity":
        # complex-number least squares, with the board's mirror
        s = (src_train * [-1, 1]) @ [1, 1j]
        d = dst_train @ [1, 1j]
        A = np.column_stack((s, np.ones(len(s))))
        a, b = np.linalg.lstsq(A, d, rcond=None)[0]
        t = (src_test * [-1, 1]) @ [1, 1j]
        z = a * t + b
        return np.column_stack((z.real, z.imag))
    H, _ = cv2.findHomography(src_train, dst_train, 0)
    h = np.column_stack((src_test, np.ones(len(src_test)))) @ H.T
    return h[:, :2] / h[:, 2:]


if __name__ == "__main__":
    for cfg in sys.argv[1:]:
        fit, rows = load(cfg)
        arm = np.array([(r['x'], r['y']) for r in rows])
        cam = np.array([r['cam'] for r in rows])
        places = {}
        for k, (x, y) in enumerate(arm):
            places.setdefault((round(x, 1), round(y, 1)), []).append(k)
        print(cfg, f"{len(rows)} points, {len(places)} places")
        for kind in ("similarity", "affine", "homography"):
            own = predict(arm, cam, arm, kind)
            e = np.linalg.norm(own - cam, axis=1)
            loo = np.zeros(len(rows))
            for g in places.values():
                rest = [k for k in range(len(rows)) if k not in g]
                loo[g] = np.linalg.norm(predict(arm[rest], cam[rest], arm[g], kind) - cam[g], axis=1)
            print(f"   {kind:11s} fitted: typ {np.median(e):5.2f} worst {e.max():5.2f} | each from the others: typ "
                  f"{np.median(loo):5.2f}  90% {np.percentile(loo, 90):5.2f}  worst {loo.max():5.2f}")
