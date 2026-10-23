import sys, math, numpy as np
sys.path[:0] = ['/var/b/shared/work/maxarm/gecko', '/var/b/shared/work/maxarm/gecko/calibration']
from pathlib import Path
from camera import store
fit = store.load_camera(Path('/var/b/shared/work/maxarm/gecko/config'))
A = fit.desk_to_true
names = ('blue', 'green', 'red')
arm = {c: np.array([fit.cubes[c]['x'], fit.cubes[c]['y']]) for c in names}
true = {c: A @ arm[c] for c in names}                    # where the camera sees each cube
cups_axis = A @ np.array([-7.7, 4.6])
def turned(p, degrees):                                   # the arm's point with its bearing changed, reach kept
    t, r = math.atan2(p[0], -p[1]) + math.radians(degrees), math.hypot(*p)
    return np.array([r * math.sin(t), -r * math.cos(t)])
def affine(src, dst):
    X = np.zeros((6, 6)); b = np.concatenate(dst)
    for k, (x, y) in enumerate(src):
        X[2 * k] = (x, y, 1, 0, 0, 0); X[2 * k + 1] = (0, 0, 0, x, y, 1)
    q = np.linalg.solve(X, b); return np.array([[q[0], q[1], q[2]], [q[3], q[4], q[5]]])
for play in (0.0, 2.0, 3.0, 3.5, 3.9, 4.5):
    src = [turned(arm['blue'], play), arm['green'], arm['red']]       # blue as if approached the same way as green and red
    T = affine(src, [true[c] for c in names])
    axis = T @ np.array([0.0, 0.0, 1.0])
    L = T[:, :2]
    sx, sy = np.linalg.norm(L @ (np.linalg.inv(A) @ np.array([1.0, 0]))), 0
    print(f'play {play:3.1f} deg: the arm axis lands {np.linalg.norm(axis - cups_axis):4.1f} mm from the cups\' centre;'
          f' stretch eigenvalues {np.round(np.sqrt(np.linalg.eigvalsh(L.T @ L)), 4)}')
