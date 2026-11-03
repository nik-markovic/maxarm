"""/IOTCONNECT: telemetry out, commands in, and the stream's channel handed to WebRTC (the property-alarm demo's
`iotc.py`, `work/refs/iotc-eiq-imx95-property-alarm`, cut to what this app uses).

Everything cloud-shaped is in this file, on one thread, and the rest of the app
does not import it:

- The app never waits on the network: the command threads write into
  `telemetry.TelemetryState`, and this file's thread reads it and sends.
- The SDK calls `_on_c2d` on paho's MQTT thread, where anything slow stalls all
  MQTT traffic, the acknowledgement included. So it only looks the command up,
  hands it to `CommandService` and returns; the command runs on a command thread,
  and is acknowledged when it is done, with the same sentence the console prints.

What flows each way:

    D2C   every 10 s, and at once when a command asks: sdk_version, version, and
          the tiles' letters and those in trouble (`tilework.TileWork.summary()`)
    C2D   the device template's commands (`files/device-template.json`), each
          acknowledged with its answer
    KVS   the signalling channel and the AWS credentials `webrtc.py` signs with

The cloud is required: `connect()` raises, and `main.py` lets it stop the app,
with the SDK's own message. Once connected, a network that comes and goes is
retried rather than fatal.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from pathlib import Path
from threading import Event, Thread
from time import monotonic
from typing import TYPE_CHECKING, Callable, Optional

from . import commands
from .commands import Command, CommandService
from .telemetry import TelemetryState

if TYPE_CHECKING:                       # for the type only: webrtc.py pulls in aiortc, which may not be there
    from .webrtc import WebRtcStreamer

logger = logging.getLogger(__name__)

try:
    from avnet.iotconnect.sdk.lite import Callbacks, Client, ClientSettings, DeviceConfig
    from avnet.iotconnect.sdk.lite import __version__ as SDK_VERSION
    from avnet.iotconnect.sdk.sdklib.mqtt import C2dAck
    IS_SDK_AVAILABLE = True
except ImportError:                     # the app runs without the cloud (--no-iotc) with no SDK installed
    SDK_VERSION = "not installed"
    IS_SDK_AVAILABLE = False

# The device template's command names, and the verbs they run. The only place those names appear.
C2D_VERBS = {
    "detect": commands.DETECT,
    "place": commands.PLACE,
    "find-cube": commands.FIND_CUBE,
    "reader-location": commands.READER_LOCATION,
}

TELEMETRY_INTERVAL_S = 10.0
MIN_SEND_GAP_S = 1.0        # a burst of wakes must not turn into an MQTT flood
RECONNECT_WAIT_S = 30.0     # after a failure that took the client down entirely
CREDENTIALS_MARGIN_S = 120  # AWS credentials are renewed this long before they expire
MAX_ACK_CHARS = 200         # an acknowledgement is shown as a tooltip: a sentence, not a page


class IotcClient:
    def __init__(self, config_path: Path, cert_path: Path, key_path: Path, service: CommandService,
                 telemetry: TelemetryState, app_version: str, streaming: "Optional[WebRtcStreamer]" = None,
                 on_status: Optional[Callable[[str], None]] = None,
                 on_stream_status: Optional[Callable[[str], None]] = None) -> None:
        self.config_path = config_path
        self.cert_path = cert_path
        self.key_path = key_path
        self.service = service
        self.telemetry = telemetry
        self.app_version = app_version
        self.streaming = streaming      # started here: only the cloud knows the channel
        self.on_status = on_status or (lambda status: None)
        self.on_stream_status = on_stream_status or (lambda status: None)
        self._client: Optional["Client"] = None
        self._kvs = None
        self._last_sent = 0.0
        self._stop = Event()
        self._thread = Thread(target=self._run, name="iotc", daemon=True)

    # --- lifecycle --------------------------------------------------------------------------------------------

    def connect(self) -> None:
        """Connect now, on the caller's thread, and raise if it cannot: nothing is caught here."""
        self.on_status("cloud: connecting")
        self._connect()

    def start(self) -> None:
        """Send telemetry from here on, on this client's thread. `connect()` first."""
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.telemetry.wake()           # so the publisher is not left in its wait
        if self._thread.is_alive():
            self._thread.join(timeout=10.0)
        if self._client is not None:
            self._client.disconnect()

    def _run(self) -> None:
        """The publisher. A connection that drops is tried again; only the first one is fatal."""
        while not self._stop.is_set():
            try:
                if self._client is None:
                    self._connect()
                self._publish_loop()
            except Exception as error:  # a network that comes and goes must not end the app
                logger.exception("iotconnect client failed")
                print(f"[iotc] {type(error).__name__}: {error} -- trying again in {RECONNECT_WAIT_S:.0f} s",
                      flush=True)
                self.on_status("cloud: offline")
                self._client = None
                self._stop.wait(RECONNECT_WAIT_S)

    # --- connecting -------------------------------------------------------------------------------------------

    def _connect(self) -> None:
        config = DeviceConfig.from_iotc_device_config_json_file(
            device_config_json_path=str(self.config_path),
            device_cert_path=str(self.cert_path),
            device_pkey_path=str(self.key_path),
        )
        self._client = Client(
            config=config,
            callbacks=Callbacks(command_cb=self._on_c2d, ota_cb=self._on_ota, disconnected_cb=self._on_disconnect),
            settings=ClientSettings(verbose=False),
        )
        self._client.connect()
        if not self._client.is_connected():
            raise ConnectionError("could not connect to /IOTCONNECT")
        print(f"[iotc] connected as {self._client.get_duid()}, SDK {SDK_VERSION}", flush=True)
        self.on_status("cloud: online")
        self._prepare_kvs()

    def _prepare_kvs(self) -> None:
        """The stream's channel and credentials, fetched at connect and handed to the streamer.

        The channel stays open the whole run, whatever the back end's streaming flag
        says: it costs nothing while nobody watches, and a viewer pressing play must be
        answered at once (the property-alarm demo's reasoning).
        """
        self._kvs = self._client.get_kvs_client()
        if self._kvs is None:
            print("[iotc] KVS disabled for this device (enable Video Streaming in the template)", flush=True)
            self.on_stream_status("stream: not enabled")
            return
        self._kvs.obtain_credentials()
        channel_arn = self._kvs.get_signaling_channel_arn()
        print(f"[iotc] KVS channel {channel_arn}, credentials good for {self._kvs.get_secs_to_expiry() / 60:.0f} "
              f"min", flush=True)
        if self.streaming is None:
            self.on_stream_status("stream: off")
        elif not channel_arn:
            print("[iotc] no signalling channel: enable Video Streaming -> WebRTC in the template", flush=True)
            self.on_stream_status("stream: no channel")
        else:
            # A callable, not the credentials themselves: they are renewed on this thread, and the streamer signs
            # each new connection with whatever is current.
            self.streaming.start(channel_arn,
                                 lambda: self._kvs.get_credentials(refresh_if_secs_to_expiry=CREDENTIALS_MARGIN_S))

    def _refresh_credentials(self) -> None:
        """The KVS credentials last an hour; they are renewed two minutes early."""
        if self._kvs is not None and self._kvs.get_secs_to_expiry() < CREDENTIALS_MARGIN_S:
            print("[iotc] renewing the KVS credentials", flush=True)
            self._kvs.obtain_credentials()

    # --- telemetry out ----------------------------------------------------------------------------------------

    def _publish_loop(self) -> None:
        """Send every `TELEMETRY_INTERVAL_S`, or as soon as something calls `telemetry.wake()`."""
        while not self._stop.is_set():
            is_woken = self.telemetry.wait_for_wake(TELEMETRY_INTERVAL_S)
            if self._stop.is_set():
                return
            if is_woken:
                self._stop.wait(max(0.0, MIN_SEND_GAP_S - (monotonic() - self._last_sent)))
            if not self._client.is_connected():
                self.on_status("cloud: reconnecting")
                self._client.connect()
                self.on_status("cloud: online" if self._client.is_connected() else "cloud: offline")
            self._refresh_credentials()
            values = {**self.telemetry.collect(), "sdk_version": SDK_VERSION, "version": self.app_version}
            self._client.send_telemetry(values)
            self._last_sent = monotonic()

    # --- commands in ------------------------------------------------------------------------------------------

    def _on_c2d(self, message) -> None:
        """MQTT thread: look the command up, hand it to a command thread, return. Nothing may escape from here: an
        exception on paho's thread stops MQTT for good."""
        try:
            verb = C2D_VERBS.get(message.command_name)
            print(f"[iotc] c2d {message.command_raw!r} -> {verb or 'not a command of this app'}", flush=True)
            if verb is None:
                self._send_ack(message, False, f"{message.command_name} is not a command of this app")
                return
            command = Command(verb, " ".join(message.command_args), message.command_raw)
            received = monotonic()
            future = self.service.submit_command(command, source="c2d")
            future.add_done_callback(lambda done: self._on_command_done(message, done, received))
        except Exception as error:
            logger.exception("could not take a c2d command")
            print(f"[iotc] could not take {getattr(message, 'command_raw', message)!r}: {error}", flush=True)

    def _on_command_done(self, message, future: Future, received: float) -> None:
        """Command thread: acknowledge the command with its answer."""
        try:
            result = future.result()
            is_ok, text = result.is_ok, result.message
        except Exception as error:
            logger.exception("c2d %s failed", message.command_name)
            is_ok, text = False, f"Failed: {type(error).__name__}"
        self._send_ack(message, is_ok, text)
        print(f"[iotc] ack {message.command_name} {'ok' if is_ok else 'failed'} in {monotonic() - received:.1f} s",
              flush=True)

    def _send_ack(self, message, is_ok: bool, text: str) -> None:
        # No ack id: the template says the command needs none. No client: the command outlived a reconnect.
        if message.ack_id is None or self._client is None:
            return
        status = C2dAck.CMD_SUCCESS_WITH_ACK if is_ok else C2dAck.CMD_FAILED
        self._client.send_command_ack(message, status, shorten(text, MAX_ACK_CHARS))

    def _on_ota(self, message) -> None:
        """No updates over the air: said so rather than left hanging."""
        print(f"[iotc] OTA offered, version {message.version}: not done by this app", flush=True)
        self._client.send_ota_ack(message, C2dAck.OTA_DOWNLOAD_FAILED, "OTA is not done by this app")

    def _on_disconnect(self, reason: str, is_from_server: bool) -> None:
        print(f"[iotc] disconnected{' by the server' if is_from_server else ''}: {reason}", flush=True)
        self.on_status("cloud: offline")


def shorten(text: str, limit: int) -> str:
    """`text` on one line, cut to `limit` characters with an ellipsis when it does not fit."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 3].rstrip() + "..."
