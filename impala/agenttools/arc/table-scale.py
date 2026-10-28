"""The owner's latest point table (PILOT-impala, after the new SAVE): the arm's x, y against where the camera saw the
cube, desk mm. Fitted: a turn and a shift (what SAVE does), then also one scale. Does the camera read distances
longer than the arm? Each point also predicted from the others. No arm.
"""
import numpy as np

rows = np.array([
    (-140.3, -219.8, -140.0, -220.4), (140.5, -219.6, 139.3, -218.2), (139.4, 0.2, 141.6, -1.3),
    (-232.5, 68.5, -234.4, 70.8), (-230.2, -90.5, -234.1, -88.5), (-140.3, -219.8, -139.8, -220.7),
    (140.5, -219.6, 139.8, -218.8), (10.2, -150.1, 10.7, -150.0), (-109.8, -90.7, -111.6, -90.1),
    (190.0, -110.7, 193.2, -108.9), (-150.4, -20.9, -149.8, -21.7), (-48.7, -230.2, -48.0, -232.6),
    (90.4, -110.0, 91.3, -109.8), (49.5, -230.7, 49.7, -230.8), (180.1, 0.3, 181.2, -1.6),
    (-129.3, -160.9, -128.8, -163.2), (130.2, -159.8, 129.4, -158.6)])
arm, seen = rows[:, :2], rows[:, 2:]


def fit(a, b, scaled):
    """b ~ s R a + t, least squares; s = 1 unless scaled."""
    am, bm = a.mean(axis=0), b.mean(axis=0)
    u, sv, vt = np.linalg.svd((b - bm).T @ (a - am))
    rot = u @ np.diag([1, np.linalg.det(u @ vt)]) @ vt
    s = (sv * [1, np.linalg.det(u @ vt)]).sum() / ((a - am) ** 2).sum() if scaled else 1.0
    return s, rot, bm - s * rot @ am


for scaled in (False, True):
    s, rot, t = fit(arm, seen, scaled)
    off = np.linalg.norm(seen - (s * arm @ rot.T + t), axis=1)
    alone = []
    for i in range(len(rows)):
        others = np.arange(len(rows)) != i
        s_i, r_i, t_i = fit(arm[others], seen[others], scaled)
        alone.append(np.linalg.norm(seen[i] - (s_i * r_i @ arm[i] + t_i)))
    label = "turn, shift and one scale" if scaled else "turn and shift (SAVE's)"
    print(f"{label}: camera's distances x {s:.4f} the arm's; off median {np.median(off):.2f}, worst {off.max():.2f} mm;"
          f" each from the others median {np.median(alone):.2f}, worst {np.max(alone):.2f} mm")
