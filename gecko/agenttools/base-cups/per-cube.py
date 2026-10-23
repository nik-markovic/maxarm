import sys, math, numpy as np
sys.path[:0] = ['/var/b/shared/work/maxarm/gecko', '/var/b/shared/work/maxarm/gecko/calibration']
from pathlib import Path
from camera import store
fit = store.load_camera(Path('/var/b/shared/work/maxarm/gecko/config'))
A = fit.desk_to_true
cups = {'front': np.array([55.8, -54.2]), 'right': np.array([-65.8, -54.7]), 'left': np.array([50.1, 64.2])}   # desk, 3-cube mapping
cups_true = {k: A @ v for k, v in cups.items()}
axis = (cups_true['left'] + cups_true['right']) / 2
# the plate's own axes, true mm: arm +x runs from the right cup to the front cup; arm +y from the front cup to the left cup
ex = cups_true['front'] - cups_true['right']; ex /= np.linalg.norm(ex)
ey = cups_true['left'] - cups_true['front']; ey /= np.linalg.norm(ey)
print(f"plate's sides {math.degrees(math.acos(abs(ex @ ey))):.1f} deg apart (90 if square)")
def plate(p):                          # true mm -> mm along the plate's arm-x and arm-y, from the axis
    d = p - axis; return np.array([d @ ex, d @ ey])
print('cube   arm says (x, y)        plate says (x, y)       reach arm/plate     bearing arm/plate')
for c in ('blue', 'green', 'red'):
    a = np.array([fit.cubes[c]['x'], fit.cubes[c]['y']]); t = plate(A @ a)
    ba, bt = math.degrees(math.atan2(a[0], -a[1])), math.degrees(math.atan2(t[0], -t[1]))
    print(f'{c:6s} ({a[0]:7.1f}, {a[1]:7.1f})   ({t[0]:7.1f}, {t[1]:7.1f})     {np.linalg.norm(a):6.1f} / {np.linalg.norm(t):6.1f}     {ba:7.2f} / {bt:7.2f}  ({bt - ba:+.2f})')
