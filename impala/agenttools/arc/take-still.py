"""A snapshot from the camera as the app takes it (5 fresh MJPG frames averaged), written to a PNG. No arm."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from calibration.camera.feed import Feed  # noqa: E402

out = Path(sys.argv[1])
feed = Feed()
feed.start()
try:
    frame, described = feed.snapshot()
finally:
    feed.close()
cv2.imwrite(str(out), frame)
print(described, "->", out)
