#!/usr/bin/env python3
"""The MaxArm tiles demo: /IOTCONNECT commands in, telemetry out, and the camera's picture with the tiles drawn over
it, live over KVS WebRTC.

    ../.venv/bin/python src/main.py               # the arm, the camera and the cloud
    ../.venv/bin/python src/main.py --no-iotc     # the arm and the camera; commands typed at the console only

Commands, from the device's Commands tab in /IOTCONNECT or typed at the console:

    detect                  find the tiles and read their letters; the stream shows them
    place TEAM              detect, then set those letters down along the dropoff arc for the reader
    find-cube               find the red cube and bring the cup over it
    reader-location left    where the person reading the words stands: back, front, left or right

Calibrate first: the calibration app (`src/calibration/calibration.py`) writes
config/, and this app uses what it chose there -- the camera's mode and focus, the
grip height, the reader. The two cannot run at once: each holds the arm and the
camera. The device's identity is in config/ too: `iotcDeviceConfig.json`, as
downloaded from /IOTCONNECT, and `device-cert.pem` and `device-pkey.pem`.

The layout, and the rule is one-way, main.py -> app/ -> applib/ -> maxarm/:

    main.py        this file: the composition root, the only place anything is wired
    app/           what the app does, command by command. Read this.
    applib/        how: the tiles, the camera, the overlay, the cloud, WebRTC
    maxarm/        the arm
    calibration/   the calibration app, its pages in ../web/

The threads:

    camera     the live view, at the calibration's frame size             applib/camera/feed.py
    stream     each frame, the overlay burnt in, to WebRTC while watched   applib/stream.py
    webrtc     asyncio: the signalling channel, every viewer, encoding      applib/webrtc.py
    iotc       the telemetry publisher, and paho's own MQTT thread          applib/iotc.py
    command    two: one moves the arm at a time (app/app.py)               applib/commands.py
    main       the console, or a wait for Ctrl-C

/IOTCONNECT is connected first, before the arm and the camera: the cloud is
required, so a wrong certificate or a device the back end does not know stops the
app there, with the SDK's own message. --no-iotc is the way to run without it.
"""

import argparse
import signal
import sys
from pathlib import Path
from threading import Event

from maxarm import MaxArm
from maxarm.spaces import CONFIG_DIR

from app.app import TilesApp
from applib import iotc, webrtc
from applib.camera.feed import Feed
from applib.camera.store import CameraFile
from applib.commands import CommandService
from applib.console import Console
from applib.motion import Motion
from applib.overlay.stream import StreamOverlay
from applib.redcube import RedCube
from applib.settings import Settings
from applib.stream import Stream, stream_size
from applib.telemetry import TelemetryState
from applib.tilework import TileWork

VERSION = "0.1.0"                       # the `version` telemetry
DEVICE_CONFIG = CONFIG_DIR / "iotcDeviceConfig.json"
DEVICE_CERT = CONFIG_DIR / "device-cert.pem"
DEVICE_KEY = CONFIG_DIR / "device-pkey.pem"
CALIBRATION_APP = Path("src/calibration/calibration.py")
SHUTDOWN_WAIT_S = 10.0                  # for a command to give the arm up once told to stop


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-iotc", action="store_true",
                        help="do not connect to /IOTCONNECT: no telemetry, no stream, commands at the console")
    args = parser.parse_args()

    camera_file = CameraFile(CONFIG_DIR)
    fit, problem = camera_file.get()
    if problem:
        raise SystemExit(f"{problem}. Calibrate first: {CALIBRATION_APP}, the desk and camera pages.")
    settings = Settings(CONFIG_DIR)
    feed = Feed(None, CONFIG_DIR)
    work = TileWork(feed, CONFIG_DIR)
    cube = RedCube(feed)
    overlay = StreamOverlay(fit, stream_size(fit.frame_size), work.scene, lambda: cube.found)
    telemetry = TelemetryState()
    motion = Motion(MaxArm(config_dir=None), CONFIG_DIR)
    app = TilesApp(motion, work, cube, settings, camera_file, overlay, telemetry)
    service = CommandService(app.on_command, on_result=lambda result: overlay.show_answer(result.message,
                                                                                         result.is_ok))
    streamer = build_streamer(args, overlay)
    stream = Stream(feed, overlay, streamer.on_frame if streamer else _drop,
                    streamer.is_watched if streamer else _nobody)
    cloud = build_cloud(args, service, telemetry, overlay, streamer)

    signal.signal(signal.SIGTERM, _interrupt)           # stopped as Ctrl-C stops it: the arm held, the port closed
    try:
        if cloud is not None:
            cloud.connect()
        print("arm        : connecting, which resets the board and homes the arm (4 s, sometimes 15)", flush=True)
        connect_arm(motion)
        feed.start()
        stream.start()
        if cloud is not None:
            cloud.start()
        print(f"camera     : {fit.frame_size[0]} x {fit.frame_size[1]}, as calibrated; stream "
              f"{stream.size[0]} x {stream.size[1]}", flush=True)
        if sys.stdin.isatty():
            Console(service).run()
        else:
            print("no console: commands from /IOTCONNECT only; Ctrl-C or SIGTERM to stop", flush=True)
            Event().wait()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        shut_down(app, motion, feed, stream, service, streamer, cloud)
    return 0


def build_streamer(args, overlay: StreamOverlay):
    """The WebRTC streamer, with no channel yet -- `iotc.py` starts it once the cloud names one -- or None."""
    if args.no_iotc:
        overlay.set_stream_status("stream: off")
        return None
    if not webrtc.IS_WEBRTC_AVAILABLE:
        print("WebRTC     : off, aiortc is not installed (src/requirements.txt)", flush=True)
        overlay.set_stream_status("stream: unavailable")
        return None
    return webrtc.WebRtcStreamer(on_status=overlay.set_stream_status)


def build_cloud(args, service: CommandService, telemetry: TelemetryState, overlay: StreamOverlay, streamer):
    """The /IOTCONNECT client, or None with --no-iotc. Unguarded: a missing config or key stops the app at
    `connect()`, with the SDK's own message."""
    if args.no_iotc:
        overlay.set_cloud_status("cloud: off")
        return None
    if not iotc.IS_SDK_AVAILABLE:
        raise SystemExit("/IOTCONNECT: the SDK is not installed (src/requirements.txt), or run with --no-iotc.")
    print(f"/IOTCONNECT: {DEVICE_CONFIG}, certificate {DEVICE_CERT.name}", flush=True)
    return iotc.IotcClient(DEVICE_CONFIG, DEVICE_CERT, DEVICE_KEY, service, telemetry, VERSION, streaming=streamer,
                           on_status=overlay.set_cloud_status, on_stream_status=overlay.set_stream_status)


def connect_arm(motion: Motion) -> None:
    """The arm connected and homed, or the app stopped, saying why."""
    try:
        motion.connect()
    except Exception as error:
        raise SystemExit(f"the arm did not connect: {error}") from None


def shut_down(app: TilesApp, motion: Motion, feed: Feed, stream: Stream, service: CommandService, streamer,
              cloud) -> None:
    """Everything stopped in turn; the arm held where it is and, once no command uses it, its port closed."""
    motion.stopping.set()
    motion.arm.stop()
    if cloud is not None:
        cloud.stop()
    if streamer is not None:
        streamer.stop()
    stream.stop()
    service.stop()
    feed.close()
    stream.join(timeout=5.0)
    if motion.is_connected and app.wait_idle(SHUTDOWN_WAIT_S):
        motion.arm.disconnect()


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def _drop(frame) -> None:
    """No stream: frames go nowhere."""


def _nobody() -> bool:
    return False


if __name__ == "__main__":
    sys.exit(main())
