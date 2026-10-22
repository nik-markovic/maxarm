"""A small web server: the pages' files and a JSON API, on Python's own `http.server`.

    server = Server(Path("calibration"), home="desk", pages=["camera"])
    server.post("/api/move", handler)          # handler(body: dict) -> dict
    server.add_state(provider)                 # GET /api/state merges every provider's dict
    server.get_bytes("/api/camera/check.jpg", provider)   # provider() -> (payload, type), or None
    server.get_stream("/api/camera/feed", frames)         # frames() -> JPEG after JPEG
    server.serve("127.0.0.1", 8080)

Every POST reply also carries `state`, as it is once the request is done.

Nothing to install, which keeps it easy to carry to the board. `/` goes to the
home page. Only files with a web suffix are served, and only from the page
directories (`home`, `common` and any other directory named in `pages`).
"""

import json
import math
import mimetypes
import sys
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

Handler = Callable[[dict], dict]
BytesProvider = Callable[[], Optional[Tuple[bytes, str]]]
Frames = Callable[[], Iterator[bytes]]

WEB_SUFFIXES = {".html", ".css", ".js", ".svg", ".png", ".ico", ".md"}
# Between the JPEGs of a live view: an <img> shows each one as it arrives.
STREAM_BOUNDARY = b"frame"


class BadRequest(ValueError):
    """The request is missing something, or has it in the wrong shape. The message says what."""


def number(body: dict, key: str, low: float = -math.inf, high: float = math.inf) -> float:
    """`body[key]` as a float, inside [low, high], or `BadRequest`."""
    try:
        value = float(body[key])
    except (KeyError, TypeError, ValueError):
        raise BadRequest(f"`{key}` should be a number, not {body.get(key)!r}") from None
    if not low <= value <= high:
        raise BadRequest(f"`{key}` is {value:g}; it goes from {low:g} to {high:g}")
    return value


class Server:
    def __init__(self, root: Path, home: str, pages: Sequence[str] = ()) -> None:
        self.root = root.resolve()
        self.home = home
        self.pages = {home, "common", *pages}
        self.routes: Dict[Tuple[str, str], Handler] = {}
        self.byte_routes: Dict[str, BytesProvider] = {}
        self.streams: Dict[str, Frames] = {}
        self.states: List[Callable[[], dict]] = []
        self.limits: List[Callable[[], dict]] = []
        self.get("/api/state", lambda body: _merged(self.states))
        self.get("/api/limits", lambda body: _merged(self.limits))

    def get(self, path: str, handler: Handler) -> None:
        self.routes[("GET", path)] = handler

    def post(self, path: str, handler: Handler) -> None:
        self.routes[("POST", path)] = handler

    def get_bytes(self, path: str, provider: BytesProvider) -> None:
        """GET `path` answers with what `provider()` gives, (payload, content type); None is a 404."""
        self.byte_routes[path] = provider

    def get_stream(self, path: str, frames: Frames) -> None:
        """GET `path` answers with the JPEGs `frames()` yields, for as long as the page keeps watching."""
        self.streams[path] = frames

    def add_state(self, provider: Callable[[], dict]) -> None:
        self.states.append(provider)

    def add_limits(self, provider: Callable[[], dict]) -> None:
        self.limits.append(provider)

    def serve(self, host: str, port: int) -> None:
        """Serve until Ctrl-C, which is raised to the caller."""
        httpd = ThreadingHTTPServer((host, port), _RequestHandler)
        httpd.daemon_threads = True
        httpd.app = self
        try:
            httpd.serve_forever()
        finally:
            httpd.server_close()

    def handle(self, request: "_RequestHandler", method: str) -> None:
        path = urlsplit(request.path).path
        route = self.routes.get((method, path))
        try:
            if route is not None and method == "POST":
                reply = route(request.read_json())
                request.send_json(HTTPStatus.OK, {**reply, "state": _merged(self.states)})
            elif route is not None:
                request.send_json(HTTPStatus.OK, route({}))
            elif method == "GET" and path in self.streams:
                request.send_stream(self.streams[path]())
            elif method == "GET" and path in self.byte_routes:
                found = self.byte_routes[path]()
                if found is None:
                    request.send_json(HTTPStatus.NOT_FOUND, {"status": "error", "detail": f"no {path} yet"})
                else:
                    request.send_bytes(HTTPStatus.OK, *found)
            elif method == "GET":
                self._send_file(request, path)
            else:
                request.send_json(HTTPStatus.NOT_FOUND, {"status": "error", "detail": f"no {path}"})
        except BadRequest as error:
            request.send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "detail": str(error)})
        except Exception as error:                  # the page shows it; the console keeps the trace
            traceback.print_exc(file=sys.stderr)
            request.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"status": "error", "detail": repr(error)})

    def _send_file(self, request: "_RequestHandler", path: str) -> None:
        page, _, rest = path.strip("/").partition("/")
        if not page:
            return request.redirect(f"/{self.home}/")
        if page in self.pages and not rest and not path.endswith("/"):
            return request.redirect(f"/{page}/")    # so that the page's relative links resolve
        file = self._file_for(page, rest or "index.html")
        if file is None:
            return request.send_json(HTTPStatus.NOT_FOUND, {"status": "error", "detail": f"no {path}"})
        kind = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        request.send_bytes(HTTPStatus.OK, file.read_bytes(), kind)

    def _file_for(self, page: str, relative: str) -> Optional[Path]:
        if page not in self.pages:
            return None
        directory = self.root / page
        file = (directory / relative).resolve()
        if file.suffix not in WEB_SUFFIXES or directory not in file.parents or not file.is_file():
            return None
        return file


class _RequestHandler(BaseHTTPRequestHandler):
    server_version = "maxarm-calibration"

    def do_GET(self) -> None:
        self.server.app.handle(self, "GET")

    def do_POST(self) -> None:
        self.server.app.handle(self, "POST")

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as error:
            raise BadRequest(f"the body is not JSON: {error}") from None
        if not isinstance(body, dict):
            raise BadRequest("the body should be a JSON object")
        return body

    def send_json(self, status: HTTPStatus, reply: dict) -> None:
        self.send_bytes(status, json.dumps(reply).encode(), "application/json")

    def send_bytes(self, status: HTTPStatus, payload: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_stream(self, frames: Iterator[bytes]) -> None:
        """An MJPEG stream: each JPEG replaces the last in the page's <img>, until the page goes."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={STREAM_BOUNDARY.decode()}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            for frame in frames:
                self.wfile.write(b"--%s\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                                 % (STREAM_BOUNDARY, len(frame)) + frame + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass                                    # the page was closed or reloaded
        finally:
            close = getattr(frames, "close", None)
            if close:
                close()

    def redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        """Quiet: the page polls twice a second. Failures are printed where they happen."""


def _merged(providers: List[Callable[[], dict]]) -> dict:
    """Every provider's part, and `stamp`, the time it was put together: the page drops anything older."""
    merged: dict = {"stamp": time.time_ns()}
    for provider in providers:
        merged.update(provider())
    return merged
