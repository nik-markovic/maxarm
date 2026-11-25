"""The Chat page's requests: a request put to the language model on the Ara240, which finds a word of the tiles and
has the arm set it down; stopping it; a new visitor; what went to and from the model for an answer.

The model, its tools and the conversation are `applib/agent`, as the /IOTCONNECT app's `agent` command has them. The
tiles are the Tiles page's -- the same detection, the same PLACE, with the tile offset and the reader chosen there
(`tiles/api.py`) -- so LOCATE TILES on either page serves both, and a word the model sets down shows on both.

| request                 | body | does                                                                            |
| ----------------------- | ---- | ------------------------------------------------------------------------------- |
| `POST /api/chat/ask`    | text | the request to the model, answered on a thread of its own: the page follows it  |
|                         |      | in the state; refused at once while another is answered or the model is not     |
|                         |      | ready                                                                           |
| `POST /api/chat/stop`   |      | the request being answered cut short; the arm is RESET's                         |
| `POST /api/chat/forget` |      | a new visitor: the conversation, and what the model is told of it, start again  |
| `POST /api/chat/trace`  | n    | what went to and from the model for answer `n`: asked for apart, not polled     |

The state's `chat` is `AgentService.state()`: where the model is, the conversation, the request being answered and
the word being set down. The service is the /IOTCONNECT app's `agent` command's too (`main.py`): one model, one
request at a time, whichever asks; its tools run the app's `detect` and `place`. Without strands-agents, there is
none, and the model is `off`.
"""

from typing import Optional

from applib.agent import AgentService
from applib.commands import CommandError
from calibration.common.server import BadRequest, Server, number
from calibration.tiles.api import TilesSession

MAX_TEXT = 300              # a request's characters at most: a sentence or two
OFF = {"model": {"state": "off", "detail": "strands-agents is not installed (src/requirements.txt)", "since": None,
                 "elapsed_s": None, "typical_s": 0, "name": "the language model"},
       "messages": [], "thinking": None, "placing": ""}


class ChatSession:
    def __init__(self, tiles: TilesSession, agent: Optional[AgentService]) -> None:
        self.tiles = tiles
        self.agent = agent

    def register(self, server: Server) -> None:
        server.add_state(self.state)
        server.post("/api/chat/ask", self.ask)
        server.post("/api/chat/stop", self.stop)
        server.post("/api/chat/forget", self.forget)
        server.post("/api/chat/trace", self.trace)

    # --- the requests -----------------------------------------------------

    def ask(self, body: dict) -> dict:
        text = " ".join(str(body.get("text", "")).split())
        if not text:
            raise BadRequest("no text given")
        if len(text) > MAX_TEXT:
            raise BadRequest(f"a request is {MAX_TEXT} characters at most, not {len(text)}")
        if self.agent is None:
            return {"status": "refused", "detail": f"the language model is off: {OFF['model']['detail']}"}
        try:
            self.agent.submit(text, source="page")
        except CommandError as reason:
            return {"status": "refused", "detail": str(reason)}
        return {"status": "done", "detail": ""}

    def stop(self, body: dict) -> dict:
        if self.agent is None or not self.agent.stop():
            return {"status": "refused", "detail": "nothing is being answered"}
        return {"status": "done", "detail": ""}

    def forget(self, body: dict) -> dict:
        if self.agent is not None:
            self.agent.forget()
        return {"status": "done", "detail": ""}

    def trace(self, body: dict) -> dict:
        trace = self.agent.trace(int(number(body, "n", 1))) if self.agent is not None else None
        if trace is None:
            return {"status": "refused", "detail": "that answer is no longer kept"}
        return {"status": "done", "detail": "", "trace": trace}

    # --- for the page -----------------------------------------------------

    def state(self) -> dict:
        problem = self.tiles.camera_file.get()[1]
        return {"chat": self.agent.state() if self.agent is not None else OFF,
                "unavailable": {"chat": problem} if problem else {}}
