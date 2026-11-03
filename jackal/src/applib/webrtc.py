"""KVS WebRTC: the stream, live in a browser -- the /IOTCONNECT dashboard's video tab (the property-alarm demo's
`webrtc.py`, `work/refs/iotc-eiq-imx95-property-alarm`).

The device is the master on an Amazon Kinesis Video Streams signalling channel.
/IOTCONNECT hands over the channel ARN and temporary AWS credentials (`iotc.py`);
one signed WebSocket to the channel stays open the whole run, and every viewer
pressing play arrives on it as an SDP offer and gets a peer connection of its own.

    /IOTCONNECT ---> channel ARN + AWS credentials ---> this file
    stream.py   ---> H.264 access units, the overlay drawn in ---> this file ---> RTP ---> viewer

The stream is encoded once, in `stream.py`, whoever watches; aiortc only cuts
each access unit into RTP packets (`H264Track` hands it an `av.Packet`, not a
frame). So aiortc cannot ask for a keyframe or a bitrate itself: a keyframe is
asked of the stream when a viewer joins and when one falls behind, and the
stream is told when anyone watches at all, so that it is made only then.

Each access unit is stamped with the wall-clock time it came, at 90 kHz, so
media time and the viewer's clock cannot drift apart whatever the frame rate
does.
"""

from __future__ import annotations

import asyncio
import json
import logging
from base64 import b64decode, b64encode
from fractions import Fraction
from threading import Thread
from time import monotonic
from typing import Callable, Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

try:
    import av
    import boto3
    import websockets
    from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCRtpSender, RTCSessionDescription
    from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
    from aiortc.sdp import candidate_from_sdp
    from botocore.auth import SigV4QueryAuth
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials
    IS_WEBRTC_AVAILABLE = True
except ImportError:                     # the app runs without the stream, as it runs without the cloud
    IS_WEBRTC_AVAILABLE = False
    MediaStreamTrack = object

VIDEO_CLOCK_RATE = 90000                # the RTP clock for video, fixed by the standard
SIGNED_URL_TTL_S = 299                  # KVS refuses a signature older than 5 minutes
RECONNECT_WAIT_S = 5.0
QUEUE_DEPTH = 4                         # access units a viewer may fall behind before it is brought up to now
# A viewer in any of these is gone. `disconnected` too: a browser closed with its X says no goodbye, and a viewer
# that comes back sends a fresh offer.
DEAD_STATES = ("disconnected", "failed", "closed")


class WebRtcStreamer:
    """One signalling channel, held open, and one peer connection per viewer."""

    def __init__(self, on_status: Callable[[str], None], on_watched: Callable[[bool], None],
                 on_keyframe_wanted: Callable[[], None]) -> None:
        self.on_status = on_status
        self.on_watched = on_watched                    # anyone watching, or about to, or nobody now
        self.on_keyframe_wanted = on_keyframe_wanted
        self.channel_arn = ""
        self.region = ""
        self.get_credentials: Callable[[], object] = lambda: None
        self.endpoints: Dict[str, str] = {}
        self._viewers: Dict[str, Tuple["RTCPeerConnection", "H264Track"]] = {}
        self._tracks: Set["H264Track"] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None
        self._thread = Thread(target=self._run, name="webrtc", daemon=True)
        self._is_stopping = False

    # --- lifecycle --------------------------------------------------------------------------------------------

    def start(self, channel_arn: str, get_credentials: Callable[[], object]) -> None:
        """Stream on this channel: `iotc.py` calls it once the cloud names the channel. Again, after a reconnect,
        it does nothing: the channel does not change."""
        if self._thread.is_alive():
            return
        self.channel_arn = channel_arn
        self.region = channel_arn.split(":")[3]
        self.get_credentials = get_credentials
        self.on_status("stream: connecting")
        self._thread.start()

    def stop(self) -> None:
        """Cancel the signalling task and let it close the viewers, rather than stopping the loop under them."""
        self._is_stopping = True
        if self._loop is not None and self._task is not None:
            self._loop.call_soon_threadsafe(self._task.cancel)
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def on_encoded(self, access_unit: bytes) -> None:
        """GStreamer's thread, once per encoded frame: stamped now and handed to the event loop."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        stamp = monotonic()
        try:
            loop.call_soon_threadsafe(self._dispatch, access_unit, stamp)
        except RuntimeError:            # the loop closed between the check and the call
            pass

    def _dispatch(self, access_unit: bytes, stamp: float) -> None:
        for track in tuple(self._tracks):
            track.offer(access_unit, stamp)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._runner())
        except Exception:               # a dead stream must never take the app with it
            logger.exception("webrtc thread failed")
        finally:
            self._loop.close()

    async def _runner(self) -> None:
        self._task = asyncio.current_task()
        try:
            await self._signalling_loop()
        except asyncio.CancelledError:
            pass
        for viewer_id in list(self._viewers):
            await self._drop_viewer(viewer_id)

    # --- the signalling channel -------------------------------------------------------------------------------

    async def _signalling_loop(self) -> None:
        """Connected to the channel for the whole run, again and again if it drops."""
        while not self._is_stopping:
            try:
                url = await asyncio.to_thread(self.build_signed_url)
                async with websockets.connect(url) as socket:
                    print(f"[webrtc] master on {self.channel_arn.rsplit('/', 2)[-2]}, waiting for viewers", flush=True)
                    self.on_status("stream: ready")
                    async for message in socket:
                        await self._on_signal(message, socket)
            except Exception as error:
                if self._is_stopping:
                    return
                logger.exception("signalling channel dropped")
                print(f"[webrtc] signalling: {type(error).__name__}: {error} -- again in {RECONNECT_WAIT_S:.0f} s",
                      flush=True)
                self.on_status("stream: offline")
                await asyncio.sleep(RECONNECT_WAIT_S)

    def build_signed_url(self) -> str:
        """A SigV4-signed wss:// URL for the MASTER role; no client id, which is a viewer's. Blocking: off-loop."""
        credentials = self.get_credentials()
        if not self.endpoints:
            self.endpoints = self._describe_endpoints(credentials)
        request = AWSRequest(method="GET", url=self.endpoints["WSS"], params={"X-Amz-ChannelARN": self.channel_arn})
        SigV4QueryAuth(_botocore_credentials(credentials), "kinesisvideo", self.region,
                       SIGNED_URL_TTL_S).add_auth(request)
        return request.prepare().url

    def _describe_endpoints(self, credentials) -> Dict[str, str]:
        """Where this channel's master endpoints are. Kept: they do not move."""
        client = self._aws_client("kinesisvideo", credentials)
        response = client.get_signaling_channel_endpoint(
            ChannelARN=self.channel_arn,
            SingleMasterChannelEndpointConfiguration={"Protocols": ["HTTPS", "WSS"], "Role": "MASTER"})
        return {entry["Protocol"]: entry["ResourceEndpoint"] for entry in response["ResourceEndpointList"]}

    def _aws_client(self, service: str, credentials, endpoint_url: Optional[str] = None):
        return boto3.client(service, region_name=self.region, endpoint_url=endpoint_url,
                            aws_access_key_id=credentials.access_key_id,
                            aws_secret_access_key=credentials.secret_access_key,
                            aws_session_token=credentials.session_token)

    def _ice_servers(self) -> list:
        """KVS's STUN server and the TURN relays it hands out: a viewer off this network needs them. Blocking."""
        credentials = self.get_credentials()
        client = self._aws_client("kinesis-video-signaling", credentials, endpoint_url=self.endpoints["HTTPS"])
        config = client.get_ice_server_config(ChannelARN=self.channel_arn, ClientId="MASTER")
        servers = [RTCIceServer(urls=f"stun:stun.kinesisvideo.{self.region}.amazonaws.com:443")]
        servers.extend(RTCIceServer(urls=server["Uris"], username=server["Username"], credential=server["Password"])
                       for server in config["IceServerList"])
        return servers

    async def _on_signal(self, message: str, socket) -> None:
        message_type, payload, viewer_id = _decode_signal(message)
        if message_type == "SDP_OFFER":
            await self._on_offer(payload, viewer_id, socket)
        elif message_type == "ICE_CANDIDATE":
            await self._on_ice_candidate(payload, viewer_id)

    # --- one viewer -------------------------------------------------------------------------------------------

    async def _on_offer(self, payload: dict, viewer_id: str, socket) -> None:
        """A viewer pressed play: a peer connection carrying the stream, for them."""
        print(f"[webrtc] viewer {viewer_id} connecting", flush=True)
        await self._drop_viewer(viewer_id)  # a reload sends a second offer for the same id
        ice_servers = await asyncio.to_thread(self._ice_servers)
        peer = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))
        track = H264Track(self.on_keyframe_wanted)
        self._viewers[viewer_id] = (peer, track)
        self._tracks.add(track)
        self.on_watched(True)

        @peer.on("connectionstatechange")
        async def on_state_change() -> None:
            print(f"[webrtc] viewer {viewer_id}: {peer.connectionState}", flush=True)
            if peer.connectionState in DEAD_STATES:
                await self._drop_viewer(viewer_id)
            self._report_viewers()

        peer.addTrack(track)
        pin_h264(peer)                       # before the offer is taken: the codecs are settled there
        await peer.setRemoteDescription(RTCSessionDescription(sdp=payload["sdp"], type=payload["type"]))
        await peer.setLocalDescription(await peer.createAnswer())
        await socket.send(_encode_signal("SDP_ANSWER", peer.localDescription, viewer_id))
        self._report_viewers()
        self.on_keyframe_wanted()            # so the first picture comes now, not at the next keyframe

    async def _on_ice_candidate(self, payload: dict, viewer_id: str) -> None:
        viewer = self._viewers.get(viewer_id)
        if viewer is None or not payload.get("candidate"):
            return
        candidate = candidate_from_sdp(payload["candidate"])
        candidate.sdpMid = payload.get("sdpMid")
        candidate.sdpMLineIndex = payload.get("sdpMLineIndex")
        await viewer[0].addIceCandidate(candidate)

    async def _drop_viewer(self, viewer_id: str) -> None:
        """Forget a viewer: its track out of the stream first, then the connection closed."""
        viewer = self._viewers.pop(viewer_id, None)
        if viewer is None:
            return
        peer, track = viewer
        self._tracks.discard(track)
        if not self._tracks:
            self.on_watched(False)
        await peer.close()

    def _report_viewers(self) -> None:
        watching = sum(1 for peer, _ in self._viewers.values() if peer.connectionState == "connected")
        self.on_status(f"stream: {watching} watching" if watching else "stream: ready")


class H264Track(MediaStreamTrack):
    """One viewer's view of the stream: the access units it has yet to be sent, a few at most."""

    kind = "video"

    def __init__(self, on_keyframe_wanted: Callable[[], None]) -> None:
        super().__init__()
        self.on_keyframe_wanted = on_keyframe_wanted
        self._queue: asyncio.Queue = asyncio.Queue()
        self._started_at: Optional[float] = None

    def offer(self, access_unit: bytes, stamp: float) -> None:
        """Event-loop thread. A viewer behind is brought up to now, not sent stale frames: H.264 with its middle
        dropped is broken until the next keyframe, so all that waits is dropped and a keyframe asked for."""
        if self._queue.qsize() >= QUEUE_DEPTH:
            _drain(self._queue)
            self.on_keyframe_wanted()
        self._queue.put_nowait((access_unit, stamp))

    def stop(self) -> None:
        """Wake `recv()`, so that the sender can finish when the connection closes."""
        super().stop()
        self._queue.put_nowait(None)

    async def recv(self) -> "av.Packet":
        item = await self._queue.get()
        if item is None:
            raise MediaStreamError          # the track was stopped: aiortc unwinds the sender on this
        access_unit, stamp = item
        if self._started_at is None:
            self._started_at = stamp
        packet = av.Packet(access_unit)
        packet.pts = int((stamp - self._started_at) * VIDEO_CLOCK_RATE)
        packet.time_base = Fraction(1, VIDEO_CLOCK_RATE)
        return packet


def pin_h264(peer: "RTCPeerConnection") -> None:
    """H.264 and nothing else on the video transceiver: what the property-alarm demo's viewers are sent."""
    codecs = [codec for codec in RTCRtpSender.getCapabilities("video").codecs if codec.mimeType == "video/H264"]
    for transceiver in peer.getTransceivers():
        if transceiver.kind == "video":
            transceiver.setCodecPreferences(codecs)


def _decode_signal(message: str) -> Tuple[str, dict, str]:
    """KVS wraps each signalling message's payload in base64'd JSON."""
    try:
        envelope = json.loads(message)
        payload = json.loads(b64decode(envelope["messagePayload"]).decode("utf-8"))
        return envelope["messageType"], payload, envelope.get("senderClientId", "")
    except (ValueError, KeyError):
        return "", {}, ""


def _encode_signal(action: str, description, viewer_id: str) -> str:
    payload = {"type": description.type, "sdp": description.sdp}
    return json.dumps({"action": action,
                       "messagePayload": b64encode(json.dumps(payload).encode("utf-8")).decode("ascii"),
                       "recipientClientId": viewer_id})


def _botocore_credentials(credentials):
    """The SDK's credentials, as botocore's signer takes them."""
    return Credentials(access_key=credentials.access_key_id, secret_key=credentials.secret_access_key,
                       token=credentials.session_token)


def _drain(queue: asyncio.Queue) -> None:
    while not queue.empty():
        queue.get_nowait()
