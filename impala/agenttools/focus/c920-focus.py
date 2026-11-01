"""Does focusing move the picture, and by how much on the desk? Does a smaller mode crop it? The camera opened as the
app opens it (MJPG, largest size, autofocus on); no arm. Each frame is matched to the reference by an affine warp
(ECC), and the warp is said as how far it moves desk points in reach, mm, through config/camera.json.

    ../.venv/bin/python agenttools/focus/c920-focus.py OUT_DIR
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "calibration"))
from camera.devices import find_camera  # noqa: E402
from camera.feed import device_index  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
device = find_camera()
desk_to_pixel = np.array(json.loads((ROOT / "config" / "camera.json").read_text())["desk_to_pixel"])
plane = desk_to_pixel[:, [0, 1, 3]]
DESK = np.array([(r * np.sin(b), -r * np.cos(b)) for r in (100, 180, 260)
                 for b in np.radians(np.arange(-110, 111, 20))])


def control(name: str, value=None):
    if value is None:
        text = subprocess.run(["v4l2-ctl", "-d", device, "-C", name], capture_output=True, text=True).stdout
        return int(text.split(":")[1])
    subprocess.run(["v4l2-ctl", "-d", device, "-c", f"{name}={value}"], check=True)


def open_camera(width=10000, height=10000):
    camera = cv2.VideoCapture(device_index(device), cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    return camera


def frame_after(camera, seconds: float, averaged: int = 3) -> np.ndarray:
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        camera.read()
    return np.mean([camera.read()[1].astype(np.float64) for _ in range(averaged)], axis=0).round().astype(np.uint8)


def warp_to(reference: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """2x3 affine W: reference pixel x is seen at W [x, 1] in `frame` (same size)."""
    grey = [cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0), 1.5).astype(np.float32)
            for image in (reference, frame)]
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    cv2.findTransformECC(grey[0], grey[1], warp, cv2.MOTION_AFFINE, criteria, None, 5)
    return warp.astype(np.float64)


def desk_moved_mm(warp: np.ndarray) -> np.ndarray:
    """How far each desk point in reach moves on the desk, mm, if its pixel moves by `warp`."""
    pixels = np.c_[DESK, np.ones(len(DESK))] @ plane.T
    pixels = pixels[:, :2] / pixels[:, 2:]
    moved = np.c_[pixels, np.ones(len(DESK))] @ warp.T
    back = np.c_[moved, np.ones(len(DESK))] @ np.linalg.inv(plane).T
    return np.linalg.norm(back[:, :2] / back[:, 2:] - DESK, axis=1)


def say(label: str, warp: np.ndarray) -> None:
    scale = np.sqrt(abs(np.linalg.det(warp[:, :2])))
    mm = desk_moved_mm(warp)
    print(f"{label:28s} scale {scale:.5f}  shift {warp[0, 2]:+6.2f} {warp[1, 2]:+6.2f} px  "
          f"desk moved median {np.median(mm):.2f} mm, worst {mm.max():.2f}")


camera = open_camera()
reference = frame_after(camera, 5.0)
size = reference.shape[1::-1]
print(f"{device}, {size[0]}x{size[1]} MJPG; autofocus on, settled 5 s")
cv2.imwrite(str(out / "reference-af.png"), reference)
again = frame_after(camera, 3.0)
say("autofocus on, 3 s later", warp_to(reference, again))

control("focus_automatic_continuous", 0)
chosen = control("focus_absolute")
held = frame_after(camera, 1.5)
print(f"autofocus off: focus_absolute reads {chosen}")
say("autofocus off, held", warp_to(reference, held))
for value in range(0, 251, 25):
    control("focus_absolute", value)
    frame = frame_after(camera, 1.5)
    cv2.imwrite(str(out / f"focus-{value:03d}.png"), frame)
    say(f"focus {value}", warp_to(reference, frame))

for cycle in range(5):
    control("focus_automatic_continuous", 0)
    control("focus_absolute", (250, 100, 250, 50, 150)[cycle])
    frame_after(camera, 1.0, 1)
    control("focus_automatic_continuous", 1)
    frame = frame_after(camera, 6.0)
    control("focus_automatic_continuous", 0)
    print(f"autofocus hunt {cycle + 1} from {(250, 100, 250, 50, 150)[cycle]}: settled at focus_absolute "
          f"{control('focus_absolute')}")
    cv2.imwrite(str(out / f"hunt-{cycle + 1}.png"), frame)
    say("  against the reference", warp_to(reference, frame))
control("focus_automatic_continuous", 1)
camera.release()

for width, height in ((1280, 720), (640, 480)):
    small = open_camera(width, height)
    frame = frame_after(small, 5.0)
    small.release()
    got = frame.shape[1::-1]
    cv2.imwrite(str(out / f"mode-{got[0]}x{got[1]}.png"), frame)
    if got[0] * size[1] == got[1] * size[0]:
        say(f"mode {got[0]}x{got[1]}, scaled up", warp_to(reference, cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)))
    else:
        grey = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in (reference, frame)]
        orb = cv2.ORB_create(4000)
        (kr, dr), (kf, df) = (orb.detectAndCompute(image, None) for image in grey)
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(dr, df)
        source = np.float32([kr[m.queryIdx].pt for m in matches])
        target = np.float32([kf[m.trainIdx].pt for m in matches])
        warp, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        corners = np.array([[0, 0, 1], [size[0], size[1], 1]]) @ warp.T
        print(f"mode {got[0]}x{got[1]}: the reference's corners land at {corners.round(0).tolist()} "
              f"({int(inliers.sum())} matches): scale {np.hypot(*warp[:, 0]):.4f}")
print(f"focus left on autofocus: {control('focus_automatic_continuous')}")
