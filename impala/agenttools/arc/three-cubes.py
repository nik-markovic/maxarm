"""The first calibration, three cubes: how far it puts each later red cube point, by the map fitted to the three.
affine: camera.json's (exact at the three, any stretch); similarity / rigid: turn (and scale) and shift only,
least squares over the three. No arm."""
import sys

import numpy as np

sys.path.insert(0, '.')
from loo import predict  # noqa: E402
from points import fit_affine, load  # noqa: E402

for cfg in sys.argv[1:]:
    fit, rows = load(cfg)
    arm = np.array([(r['x'], r['y']) for r in rows])
    cam = np.array([r['cam'] for r in rows])
    cubes = [0, 1, 2]
    later = [k for k in range(3, len(rows)) if not any(np.allclose(arm[k], arm[c]) for c in cubes)]
    print(cfg, f"{len(later)} later points at new places")
    for kind in ("affine", "similarity", "rigid"):
        if kind == "rigid":
            mapped, prm = fit_affine(arm[cubes], cam[cubes], "rigid")
            A = np.column_stack((arm[cubes], np.ones(3)))
            sol = np.linalg.lstsq(A, mapped, rcond=None)[0]
            out = np.column_stack((arm[later], np.ones(len(later)))) @ sol
            at_cubes = np.linalg.norm(mapped - cam[cubes], axis=1)
        else:
            out = predict(arm[cubes], cam[cubes], arm[later], kind)
            at_cubes = np.linalg.norm(predict(arm[cubes], cam[cubes], arm[cubes], kind) - cam[cubes], axis=1)
        e = np.linalg.norm(out - cam[later], axis=1)
        worst = later[int(np.argmax(e))]
        print(f"   {kind:10s} at the cubes {np.round(at_cubes, 2)}  later points: typ {np.median(e):5.2f}  "
              f"worst {e.max():5.2f} (point {rows[worst]['name']})")
