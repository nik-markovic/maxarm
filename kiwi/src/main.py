"""The MaxArm tiles demo, one program: the calibration pages, /IOTCONNECT commands in and telemetry out, and the
camera's picture with the tiles drawn over it, live over KVS WebRTC.

    ../.venv/bin/python src/main.py                   # the pages on http://localhost:8080, the cloud, the console
    ../.venv/bin/python src/main.py --host ::         # the pages for another machine's browser (run.sh does this)
    ../.venv/bin/python src/main.py --no-iotc         # without the cloud: the pages and the console
    ../.venv/bin/python src/main.py --no-webapp       # without the pages: the cloud and the console

Commands, from the device's Commands tab in /IOTCONNECT or typed at the console:

    detect                  find the tiles and read their letters; the stream shows them
    place TEAM              detect, then set those letters down along the dropoff arc for the reader
    find-cube               find the red cube and bring the cup over it
    reader-location left    where the person reading the words stands: back, front, left or right
    agent SENTENCE          the sentence to the language model on the Ara240, which may run the commands above;
                            its answer is the `answer` telemetry. Needs connector/ (connector/install.sh)
    stop                    the model's request and the arm's work cut short; the cup lets go, lifts and goes home

The pages calibrate the desk and the camera and try them (`calibration/`); the
commands use what they chose in config/ -- the camera's mode and focus, the
tile height, the tile offset, the reader. Both drive the one arm, one command at
a time whichever asks, and see the one camera: the pages' live view and the
stream are two branches of its pipeline, so neither takes it from the other.

/IOTCONNECT starts only with what it needs in config/: `iotcDeviceConfig.json`,
as downloaded from /IOTCONNECT, `device-cert.pem` and `device-pkey.pem`, and the
camera calibration, which sizes the stream and places its overlay. Without one,
the rest runs and says what is missing: calibrate, or put the files there, then
start again. Once started, a connection that fails is tried again.

The layout, and the rule is one-way, main.py -> app/, calibration/ -> applib/ -> maxarm/:

    main.py        this file: the composition root, the only place anything is wired
    app/           what the app does, command by command. Read this.
    calibration/   the calibration pages' requests, their files in ../web/
    applib/        how: the tiles, the camera, the overlay, the arm's one thread, the cloud, WebRTC
    maxarm/        the arm

The threads:

    arm        the worker: every command that moves the arm, a page's or the app's    applib/worker.py
    arm-retry  connects the arm again RECONNECT_S after it fails or is lost          applib/worker.py
    camera     the camera's GStreamer pipeline; the newest frame, for commands   applib/camera/feed.py
    GStreamer  its own: the stream's branch -- the overlay, H.264 -- while watched   applib/stream.py
    pages      the web server, a thread per request                              calibration/common/server.py
    webrtc     asyncio: the signaling channel, every viewer                      applib/webrtc.py
    iotc       the telemetry publisher, and paho's own MQTT thread          applib/iotc.py
    command    two: each waits for the arm's worker (app/app.py)            applib/commands.py
    connector  where the language model is: installed, loading, ready        applib/agent/connector.py
    agent      a request answered, and a word the model chose set down       applib/agent/service.py
    main       the console, or a wait for Ctrl-C
"""

import argparse
import signal
import sys
from threading import Event
from typing import Optional

from maxarm import MaxArm
from maxarm.spaces import CONFIG_DIR

from app.app import TilesApp
from applib import agent, iotc, webrtc
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
from applib.worker import Worker
from calibration import calibration

VERSION = "0.1.0"                       # the `version` telemetry
DEVICE_CONFIG = CONFIG_DIR / "iotcDeviceConfig.json"
DEVICE_CERT = CONFIG_DIR / "device-cert.pem"
DEVICE_KEY = CONFIG_DIR / "device-pkey.pem"
UNSTREAMED_SIZE = (1280, 720)           # the overlay's, with no stream to draw it for


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-iotc", action="store_true",
                        help="do not connect to /IOTCONNECT: no telemetry, no stream")
    parser.add_argument("--no-webapp", action="store_true", help="do not serve the calibration pages")
    parser.add_argument("--host", default="127.0.0.1",
                        help="the pages' address: :: (or 0.0.0.0, IPv4 only) to reach them from another machine")
    parser.add_argument("--port", type=int, default=8080, help="the pages' port")
    args = parser.parse_args()

    camera_file = CameraFile(CONFIG_DIR)
    settings = Settings(CONFIG_DIR)
    feed = Feed(CONFIG_DIR)
    worker = Worker(Motion(MaxArm(config_dir=None), CONFIG_DIR))
    work = TileWork(feed, CONFIG_DIR)
    cube = RedCube(feed)
    cloud_off = None if args.no_iotc else why_no_cloud(camera_file)
    is_cloud_on = not args.no_iotc and cloud_off is None
    fit, _ = camera_file.get()
    overlay = StreamOverlay(camera_file, stream_size(fit.frame_size) if is_cloud_on else UNSTREAMED_SIZE,
                            work.scene, lambda: cube.found)
    telemetry = TelemetryState()
    app = TilesApp(worker, work, cube, settings, camera_file, overlay, telemetry)
    model = build_agent(app, telemetry, overlay)
    app.set_agent(model)
    service = CommandService(app.on_command, on_result=lambda result: overlay.show_answer(result.message,
                                                                                         result.is_ok))
    streamer = build_streamer(overlay, feed) if is_cloud_on else None
    cloud = build_cloud(service, telemetry, overlay, streamer) if is_cloud_on else None
    if cloud_off:
        print(f"/IOTCONNECT: off, {cloud_off}; then start again", flush=True)

    signal.signal(signal.SIGTERM, _interrupt)           # stopped as Ctrl-C stops it: the arm held, the port closed
    try:
        print("arm        : connecting, which resets the board and homes the arm (4 s, sometimes 15)", flush=True)
        worker.start()
        feed.start()
        if cloud is not None:
            cloud.start()
        if not args.no_webapp:
            calibration.serve(args.host, args.port, worker, feed, work, settings, cube, model, CONFIG_DIR)
        if sys.stdin.isatty():
            Console(service).run()
        else:
            print("no console: Ctrl-C or SIGTERM to stop", flush=True)
            Event().wait()
    except KeyboardInterrupt:
        print("\nstopping: holding the arm where it is, and closing the port", flush=True)
    finally:
        shut_down(worker, feed, service, streamer, cloud)
    return 0


def why_no_cloud(camera_file: CameraFile) -> Optional[str]:
    """What /IOTCONNECT lacks to start, or None: checked once, at the start."""
    if not iotc.IS_SDK_AVAILABLE:
        return "its SDK is not installed (src/requirements.txt)"
    for path, what in ((DEVICE_CONFIG, "download it from the device's page in /IOTCONNECT"),
                       (DEVICE_CERT, "install.sh makes it, or copy in the one /IOTCONNECT made"),
                       (DEVICE_KEY, "install.sh makes it, or copy in the one /IOTCONNECT made")):
        if not path.exists():
            return f"there is no {path}: {what}"
    _, problem = camera_file.get()
    if problem:
        return f"{problem}, which the stream is sized and drawn by"
    return None


def build_streamer(overlay: StreamOverlay, feed: Feed):
    """The stream, a branch of the camera's pipeline, and the WebRTC streamer it feeds, with no channel yet --
    `iotc.py` starts it once the cloud names one; or None without aiortc, and no stream."""
    if not webrtc.IS_WEBRTC_AVAILABLE:
        print("WebRTC     : off, aiortc is not installed (src/requirements.txt)", flush=True)
        return None
    stream = Stream(overlay)
    feed.add_stream(stream)
    streamer = webrtc.WebRtcStreamer(on_watched=stream.set_watched, on_keyframe_wanted=stream.request_keyframe)
    stream.connect_encoded(streamer.on_encoded)
    print(f"stream     : {stream.size[0]} x {stream.size[1]}, H.264 "
          f"{'in hardware' if stream.is_hardware else 'by x264'}, while watched", flush=True)
    return streamer


def build_agent(app: TilesApp, telemetry: TelemetryState, overlay: StreamOverlay):
    """The language model on the Ara240, the Chat page's and the `agent` command's; or None without strands-agents.
    Nothing is waited for here: the connector is optional and loads its model for minutes; its own thread watches
    it, and says so as it changes."""
    if not agent.IS_STRANDS_AVAILABLE:
        print("agent      : off, strands-agents is not installed (src/requirements.txt)", flush=True)
        telemetry.set(agent_status="off")
        return None

    def on_change(status: dict) -> None:
        print(f"agent      : {agent.describe_status(status)}", flush=True)
        telemetry.set(agent_status=status["state"])
        telemetry.wake()

    def on_say(text: str, is_ok: bool) -> None:                 # a word set down, or not, after the answer
        telemetry.set_once(answer=text)
        overlay.show_answer(text, is_ok)

    service = agent.AgentService(agent.AraConnector(on_change=on_change), app.agent_desk(), on_say=on_say)
    overlay.watch_agent(service.activity)
    return service


def build_cloud(service: CommandService, telemetry: TelemetryState, overlay: StreamOverlay, streamer):
    print(f"/IOTCONNECT: {DEVICE_CONFIG}, certificate {DEVICE_CERT.name}", flush=True)
    return iotc.IotcClient(DEVICE_CONFIG, DEVICE_CERT, DEVICE_KEY, service, telemetry, VERSION, streaming=streamer,
                           on_status=overlay.set_cloud_status)


def shut_down(worker: Worker, feed: Feed, service: CommandService, streamer, cloud) -> None:
    """Everything stopped in turn; the arm held where it is and, once no command uses it, its port closed."""
    worker.shutdown()
    if cloud is not None:
        cloud.stop()
    if streamer is not None:
        streamer.stop()
    service.stop()
    feed.close()


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    sys.exit(main())
