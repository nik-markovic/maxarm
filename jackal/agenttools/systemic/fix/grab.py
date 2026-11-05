#!/usr/bin/env python3
"""Frames from the running calibration app's live view (GET /api/camera/feed, multipart JPEG): read-only, the arm
untouched. Skips the first few, averages the next ones, writes the average as PNG and the last frame as JPEG.

    grab.py out_stem [count] [skip]
"""
import sys
import time
import urllib.request

import cv2
import numpy as np

stem = sys.argv[1]
count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
skip = int(sys.argv[3]) if len(sys.argv) > 3 else 3
stream = urllib.request.urlopen("http://127.0.0.1:8080/api/camera/feed", timeout=30)
buffer, frames, seen, started = b"", [], 0, time.time()
while len(frames) < count and time.time() - started < 120:
    buffer += stream.read(65536)
    while True:
        start = buffer.find(b"\xff\xd8")
        end = buffer.find(b"\xff\xd9", start + 2)
        if start < 0 or end < 0:
            break
        jpeg, buffer = buffer[start:end + 2], buffer[end + 2:]
        seen += 1
        if seen > skip:
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame.astype(np.float32))
                last = jpeg
stream.close()
average = np.mean(frames, axis=0)
cv2.imwrite(stem + ".png", np.clip(average + 0.5, 0, 255).astype(np.uint8))
open(stem + "-last.jpg", "wb").write(last)
spread = np.mean([np.abs(f - average).mean() for f in frames])
print(f"{len(frames)} frames of {frames[0].shape[1]} x {frames[0].shape[0]} in {time.time() - started:.1f} s; "
      f"mean change from the average {spread:.2f} levels")
