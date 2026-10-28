"""The four frames of formats.py compared: grain on plain paper, how sharp the cube's outline is against it, colour;
and a 2 x 2 close-up of one corner of the cube, 4 pixels a pixel, for the eye. No arm.

    ../.venv/bin/python agenttools/live-res/format-numbers.py /tmp/formats
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "calibration")]
from camera import store  # noqa: E402
from camera.locate import find_red_cube  # noqa: E402

folder = Path(sys.argv[1])
names = ["yuyv-single", "mjpg-single", "yuyv-avg5", "mjpg-avg5"]
frames = {name: cv2.imread(str(folder / f"{name}.png")) for name in names}
fit = store.load_camera(ROOT / "config")
found = find_red_cube(frames["yuyv-avg5"], fit)
mapped = np.column_stack((found.corners, np.ones(8))) @ fit.desk_to_pixel.T
corners = mapped[:, :2] / mapped[:, 2:]

# Plain paper: a patch of the sheet left of the cube, its slow shading taken out.
x0, y0 = int(corners[:, 0].min()) - 420, int(corners[:, 1].mean()) - 60
paper = (slice(y0, y0 + 120), slice(x0, x0 + 300))
# The cube's outline against the paper: profiles across its left vertical edge, 0-4, from outside to inside.
a, b = corners[0], corners[4]
along = (b - a) / np.linalg.norm(b - a)
inward = np.array([along[1], -along[0]])
if inward @ (corners.mean(axis=0) - (a + b) / 2) < 0:
    inward = -inward
offsets = np.arange(-12, 12.01, 0.25)


def edge_width(gray):
    """10-90% rise across the outline, px, averaged along the middle of the edge."""
    profiles = []
    for share in np.linspace(0.25, 0.75, 21):
        base = a + share * (b - a)
        points = (base + offsets[:, None] * inward).astype(np.float32)
        values = cv2.remap(gray, points[:, 0].reshape(1, -1), points[:, 1].reshape(1, -1), cv2.INTER_LINEAR)[0]
        profiles.append(values)
    profile = np.mean(profiles, axis=0)
    low, high = profile[:8].mean(), profile[-8:].mean()
    level = (profile - low) / (high - low)
    if level[-1] < level[0]:
        level = 1 - level
    cross = [np.interp(t, level, offsets) if np.all(np.diff(level) > -0.2) else np.nan for t in (0.1, 0.9)]
    return abs(cross[1] - cross[0])


print(f"{'frame':12s} {'grain on paper':>15s} {'outline 10-90%':>15s}   paper B,G,R      cube top B,G,R")
for name in names:
    frame = frames[name]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    patch = gray[paper]
    grain = float((patch - cv2.GaussianBlur(patch, (0, 0), 4)).std())
    # Saturation of a colour edge matters to the finder: the red against the paper, through the red channel's
    # opposite, green.
    width = edge_width(frame[..., 1].astype(np.float32))
    top = corners[4:8].mean(axis=0).astype(int)
    cube_top = frame[top[1] - 15:top[1] + 15, top[0] - 15:top[0] + 15].reshape(-1, 3).mean(axis=0)
    paper_colour = frame[paper].reshape(-1, 3).mean(axis=0)
    print(f"{name:12s} {grain:12.2f} lv {width:12.2f} px   {paper_colour.round(0)}  {cube_top.round(0)}")

# The close-up: the cube's left top corner and its far edge, the same patch of each frame, 4 pixels a pixel.
corner = corners[4].astype(int)
box = (slice(corner[1] - 60, corner[1] + 90), slice(corner[0] - 60, corner[0] + 140))
panels = []
for name in names:
    panel = cv2.resize(frames[name][box], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cv2.putText(panel, name, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(panel, name, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    panels.append(panel)
grid = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
cv2.imwrite(str(folder / "side-by-side-corner.png"), grid)
print("written", folder / "side-by-side-corner.png", grid.shape[1], "x", grid.shape[0])
