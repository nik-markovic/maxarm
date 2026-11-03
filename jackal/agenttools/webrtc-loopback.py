#!/usr/bin/env python3
"""The stream's WebRTC track, sent to a second peer in the same process: no KVS, no cloud.

    ../.venv/bin/python agenttools/webrtc-loopback.py

`webrtc.FrameTrack` fed frames at the stream's size, H.264 pinned as for a
viewer, encoded by aiortc; the other peer decodes them. Says how many frames came
through, their size, and what was negotiated -- that the encoding path works, not
how a browser on another network fares.
"""

import asyncio
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from aiortc import RTCPeerConnection                          # noqa: E402

from applib.stream import stream_size                         # noqa: E402
from applib.webrtc import FrameTrack, pin_h264                # noqa: E402

SIZE = stream_size((2592, 1944))
FRAMES = 25
FPS = 5.0


async def main() -> int:
    sender, receiver = RTCPeerConnection(), RTCPeerConnection()
    track = FrameTrack()
    sender.addTrack(track)
    received = []

    @receiver.on("track")
    def on_track(incoming) -> None:
        async def take() -> None:
            while True:
                try:
                    frame = await incoming.recv()
                except Exception:
                    return
                received.append((frame.width, frame.height, frame.to_ndarray(format="bgr24")))
        asyncio.ensure_future(take())

    receiver.addTransceiver("video", direction="recvonly")
    await receiver.setLocalDescription(await receiver.createOffer())
    pin_h264(sender)
    await sender.setRemoteDescription(receiver.localDescription)
    await sender.setLocalDescription(await sender.createAnswer())
    await receiver.setRemoteDescription(sender.localDescription)
    codecs = {line.split()[1] for line in sender.localDescription.sdp.splitlines() if line.startswith("a=rtpmap")}
    for index in range(FRAMES):
        frame = np.full((SIZE[1], SIZE[0], 3), (40, 90, 30), np.uint8)
        frame[:, (index * 37) % SIZE[0]:][:, :40] = (255, 255, 255)
        track.offer(frame, time.monotonic())
        await asyncio.sleep(1.0 / FPS)
    await asyncio.sleep(1.0)
    await sender.close()
    await receiver.close()
    sizes = sorted({(width, height) for width, height, _ in received})
    print(f"offered {FRAMES} frames {SIZE[0]} x {SIZE[1]} at {FPS:g} fps; received {len(received)}, sizes {sizes}; "
          f"answer's codecs {sorted(codecs)}")
    return 0 if received else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
