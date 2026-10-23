import sys, math, json, numpy as np, cv2
sys.path[:0] = ['/var/b/shared/work/maxarm/gecko', '/var/b/shared/work/maxarm/gecko/calibration']
from pathlib import Path
from camera import store
config = Path('/var/b/shared/work/maxarm/gecko/config')
img = cv2.imread(str(config / 'camera-frame.png'))
fit = store.load_camera(config)
M = fit.desk_to_pixel
gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
def sample(image, pts):
    return cv2.remap(image, pts[:, 0].astype(np.float32).reshape(-1, 1), pts[:, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).ravel()
# rough ellipse per cup (centre, semi-axes x, y) and the image angles where the rim shows (0 = +x, 90 = down)
cups = {'front': ((933, 585), (114, 71), (-15, 200)), 'right': ((1385, 360), (85, 50), (-40, 185)), 'left': ((600, 360), (75, 42), (10, 175))}
found = {}
for name, ((cx, cy), (a, b), (t0, t1)) in cups.items():
    pts = []
    for t in np.radians(np.arange(t0, t1 + 0.1, 3)):
        r_e = 1.0 / math.sqrt((math.cos(t) / a) ** 2 + (math.sin(t) / b) ** 2)
        rs = np.linspace(0.75 * r_e, 1.3 * r_e, 120)
        ray = np.column_stack((cx + rs * math.cos(t), cy + rs * math.sin(t)))
        prof = sample(gray, ray)
        d = np.diff(prof)
        k = int(np.argmin(d))                     # brightest-to-darker step, going outward
        if d[k] < -2.0:
            pts.append(ray[k] + (ray[k + 1] - ray[k]) / 2)
    found[name] = np.array(pts)
    e = cv2.fitEllipse(found[name].astype(np.float32))
    print(f'{name}: {len(pts)} rim points; image ellipse centre ({e[0][0]:.1f}, {e[0][1]:.1f}), axes {e[1][0]:.1f} x {e[1][1]:.1f}, angle {e[2]:.1f}')
np.save('/tmp/gecko-cal/rim-points.npy', found, allow_pickle=True)
out = img[250:700, 450:1550].copy()
for name, pts in found.items():
    for p in pts:
        cv2.circle(out, (int(p[0] - 450), int(p[1] - 250)), 2, (0, 0, 255), -1)
cv2.imwrite('/tmp/gecko-cal/rims.jpg', out)
