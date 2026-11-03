#!/usr/bin/env python3
"""The stream, from the camera to a second WebRTC peer in the same process: no KVS, no cloud, no arm.

    ../.venv/bin/python agenttools/webrtc-loopback.py [CONFIG_DIR]

The camera's pipeline with the stream's branch (`stream.py`), its access units
on a `webrtc.H264Track`, H.264 pinned as for a viewer; the other peer decodes
them. Nobody watches for the first seconds, then the peer does. Says how many
frames came through, their size, the CPU the process took while nobody watched
and while the peer did, and what was negotiated -- that the path works, not how
a browser on another network fares. CONFIG_DIR, config/ by default, needs a camera.json; the
camera's mode is its camera-view.json's.
"""

import asyncio
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from aiortc import RTCPeerConnection                          # noqa: E402

from applib.camera.feed import Feed                           # noqa: E402
from applib.camera.store import CameraFile                    # noqa: E402
from applib.overlay.stream import StreamOverlay               # noqa: E402
from applib.stream import Stream, stream_size                 # noqa: E402
from applib.webrtc import H264Track, pin_h264                 # noqa: E402

IDLE_S = 8.0
WATCHED_S = 10.0


async def main(config_dir: Path) -> int:
    fit, problem = CameraFile(config_dir).get()
    if problem:
        print(problem)
        return 1
    loop = asyncio.get_running_loop()
    overlay = StreamOverlay(fit, stream_size(fit.frame_size), lambda: {}, lambda: None)
    overlay.set_resting("webrtc-loopback")
    stream = Stream(overlay)
    feed = Feed(None, config_dir)
    feed.add_stream(stream)
    keyframes = []
    track = H264Track(lambda: keyframes.append(time.monotonic()))
    units = []
    stream.connect_encoded(lambda unit: (units.append(len(unit)),
                                         loop.call_soon_threadsafe(track.offer, unit, time.monotonic())))
    feed.start()
    started = time.monotonic()
    while feed.state()["status"] != "live" and time.monotonic() - started < 20:
        await asyncio.sleep(0.2)
    print(f"camera: {feed.state()['detail']}; H.264 {'in hardware' if stream.is_hardware else 'by x264'}")
    cpu, wall = time.process_time(), time.monotonic()
    await asyncio.sleep(IDLE_S)
    idle_cpu = 100 * (time.process_time() - cpu) / (time.monotonic() - wall)
    idle_units = len(units)

    sender, receiver = RTCPeerConnection(), RTCPeerConnection()
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
                received.append((frame.width, frame.height))
        asyncio.ensure_future(take())

    receiver.addTransceiver("video", direction="recvonly")
    await receiver.setLocalDescription(await receiver.createOffer())
    pin_h264(sender)
    await sender.setRemoteDescription(receiver.localDescription)
    await sender.setLocalDescription(await sender.createAnswer())
    await receiver.setRemoteDescription(sender.localDescription)
    stream.set_watched(True)
    cpu, wall = time.process_time(), time.monotonic()
    await asyncio.sleep(WATCHED_S)
    watched_cpu = 100 * (time.process_time() - cpu) / (time.monotonic() - wall)
    stream.set_watched(False)
    await sender.close()
    await receiver.close()
    feed.close()
    codecs = {line.split()[1] for line in sender.localDescription.sdp.splitlines() if line.startswith("a=rtpmap")}
    sent = units[idle_units:]
    print(f"nobody watching: {idle_units} access units, CPU {idle_cpu:.1f}% of a core")
    print(f"watched {WATCHED_S:g} s: {len(sent)} access units, mean {sum(sent) / max(len(sent), 1) / 1024:.0f} KB; "
          f"received {len(received)} frames, sizes {sorted(set(received))}; keyframes asked {len(keyframes)}; "
          f"CPU {watched_cpu:.1f}% of a core")
    print(f"answer's codecs {sorted(codecs)}")
    return 0 if received else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "config")))
