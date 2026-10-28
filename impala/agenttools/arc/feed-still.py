"""A snapshot read from the running app's live view (GET /api/camera/feed): 5 fresh frames averaged, to a PNG.

No arm, and the app keeps its camera. Usage: feed-still.py OUT.png [skip=3] [count=5]
"""
import sys
import urllib.request

import cv2
import numpy as np

out = sys.argv[1]
skip = int(sys.argv[2]) if len(sys.argv) > 2 else 3
count = int(sys.argv[3]) if len(sys.argv) > 3 else 5

stream = urllib.request.urlopen("http://127.0.0.1:8080/api/camera/feed", timeout=30)
buffer, frames = b"", []
while len(frames) < skip + count:
    buffer += stream.read(65536)
    while True:
        start = buffer.find(b"\xff\xd8")
        end = buffer.find(b"\xff\xd9", start + 2)
        if start < 0 or end < 0:
            break
        frames.append(buffer[start:end + 2])
        buffer = buffer[end + 2:]
stream.close()
images = [cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) for jpeg in frames[skip:skip + count]]
average = (sum(image.astype(np.float64) for image in images) / len(images)).round().astype(np.uint8)
cv2.imwrite(out, average)
print(f"{len(images)} frames {average.shape[1]}x{average.shape[0]} averaged -> {out}")
