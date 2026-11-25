"""The calibration web app: one page per calibration, each a tab -- the desk first, then the camera; then the tests,
the tiles, and the chat with the language model on the Ara240 about them -- served from `main.py`, beside the
/IOTCONNECT app, on the same arm, camera, tiles and model:

    pages = serve(host, port, worker, feed, work, settings, cube, agent)    # on a thread of its own

The arm is the worker's (`applib/worker.py`): the pages' requests and the app's
commands are its commands alike, one at a time. The camera is the feed's: the
pages' live view and the stream are two branches of its one pipeline. Ctrl-C,
`main.py`'s, holds the arm where it is and closes the port.

The pages' own files -- HTML, scripts, styles -- are in the pilot's `web/`.
"""

import fcntl
import ipaddress
import socket
import struct
from pathlib import Path
from threading import Thread
from typing import List, Optional, Tuple

from applib.agent import AgentService
from applib.camera.feed import Feed
from applib.redcube import RedCube
from applib.settings import Settings
from applib.tilework import TileWork
from applib.worker import Worker
from calibration.camera.api import CameraSession
from calibration.chat.api import ChatSession
from calibration.common import arm_api
from calibration.common.server import Server
from calibration.desk.api import DeskSession
from calibration.tests.api import TestsSession
from calibration.tiles.api import TilesSession

WEB = Path(__file__).resolve().parents[2] / "web"


def serve(host: str, port: int, worker: Worker, feed: Feed, work: TileWork, settings: Settings, cube: RedCube,
          agent: Optional[AgentService], config_dir: Path) -> Thread:
    """The pages, served on a thread of their own from now on; that thread."""
    server = Server(WEB, home="desk", pages=["camera", "tests", "tiles", "chat"])
    arm_api.register(server, worker)
    DeskSession(worker, config_dir).register(server)
    CameraSession(worker, feed, config_dir).register(server)
    TestsSession(worker, feed, cube, config_dir).register(server)
    tiles = TilesSession(worker, work, settings, config_dir)
    tiles.register(server)
    ChatSession(tiles, agent).register(server)
    thread = Thread(target=server.serve, args=(host, port), name="pages", daemon=True)
    thread.start()
    print(f"pages      : http://localhost:{port}/", flush=True)
    if host in ("0.0.0.0", "::"):
        for interface, address in local_addresses(is_ipv6_wanted=host == "::"):
            print(f"  {interface:<8} http://{address}:{port}/", flush=True)
    return thread


_SIOCGIFADDR = 0x8915                     # an interface's IPv4 address, in a struct ifreq


def local_addresses(is_ipv6_wanted: bool) -> List[Tuple[str, str]]:
    """Where another machine can reach this one, by interface: each one's IPv4 address, and its global IPv6 ones,
    in brackets as a URL takes them. Loopback is left out, and so is link-local IPv6 (fe80::): a browser will not
    take the interface such a URL needs."""
    found = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        for _, interface in socket.if_nameindex():
            try:
                reply = fcntl.ioctl(probe.fileno(), _SIOCGIFADDR, struct.pack("256s", interface[:15].encode()))
            except OSError:
                continue                                    # no IPv4 address on it
            address = socket.inet_ntoa(reply[20:24])
            if not address.startswith("127."):
                found.append((interface, address))
    addresses_v6 = Path("/proc/net/if_inet6")
    if is_ipv6_wanted and addresses_v6.exists():
        # Each line: address in hex, interface index, prefix length, scope, flags, interface; scope 00 is global.
        for line in addresses_v6.read_text().splitlines():
            hex_address, _, _, scope, _, interface = line.split()
            if scope == "00":
                found.append((interface, f"[{ipaddress.IPv6Address(int(hex_address, 16))}]"))
    return found
