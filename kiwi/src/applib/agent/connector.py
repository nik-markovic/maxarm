"""Where the language model is, and how far it has got: on the Ara240, behind NXP's eIQ AAF Connector (`connector/`),
its own process and venv on this board, port 3000. Optional, and minutes to load.

    ara = AraConnector(on_change=print)
    ara.tools = [...]                       # what the probe carries: AgentService sets them
    ara.start()                             # watched on its own thread from here on: AgentService starts it
    ara.status()        # {"state": "loading", "detail": ..., "since": ..., "elapsed_s": 130, "typical_s": 345, ...}
    ara.model()         # a strands model, for one request: once ready

| state           | when                                                                                    |
| --------------- | --------------------------------------------------------------------------------------- |
| `not installed` | no `connector/venv/` and no service: none to come until `connector/install.sh`          |
| `stopped`       | installed, not running: `connector/run.sh`, or `install-service.sh` for every boot      |
| `waiting`       | the service waits for the Ara240's proxy after a boot, or 30 s after a failure          |
| `loading`       | the connector runs, its port not open yet: the model going onto the Ara240, ~6 min      |
| `ready`         | the port answers, the model is listed ready, and a request with the tools was answered  |
| `failed`        | the service failed, or a request did; looked at again shortly                           |

Ready is not the port alone: once the port opens, one request carrying the agent's tools is put to the model --
nothing run, its answer dropped. It also has the connector build its tool-call guide for those tools, seconds the
first request that carries them would otherwise wait.

**Another model, Amazon Bedrock say,** is another class with `ModelSource`'s members -- `model()` a strands
`BedrockModel` -- built in place of this one by `main.py`. The service, its tools, the prompt
and the pages do not change; a model with a wider window could take a longer recap (`chat.py`).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Callable, List, Optional, Protocol, Tuple

if TYPE_CHECKING:
    from strands.models import Model

ARA_URL = "http://127.0.0.1:3000/v1"        # connector/run.sh and its service: on the board, beside the app
ARA_MODEL = "Qwen2.5-7B-Instruct"
NAME = "Qwen2.5-7B-Instruct on the Ara240"
SERVICE = "iotc-aaf-connector"              # connector/install-service.sh
CONNECTOR_DIR = Path(__file__).resolve().parents[3] / "connector"
LOAD_S = 345                                # the model's load, the connector started to its port open

# Sampling, per request, over the model's own default, which is greedy. Connector 2.2 takes these three and none of its
# other llm_params -- no seed, no max_tokens; the repetition penalty only through our patch. Low: the
# same request finds much the same words, and set down already, another is taken (`service.py`). Never all three 0:
# greedy, the tool calls break.
TEMPERATURE = 0.3
TOP_K = 20
TOP_P = 0.8
# None: with one, the model writes more of its tool calls out as text, and repeats its guesses all the same.
# A repeated guess is checked once.
REPEAT_PENALTY = 1.0
REQUEST_TIMEOUT_S = 120.0   # one model call, a long prompt and a long answer with room to spare

LOOK_S = 5.0                # how often the connector is looked at while it is not ready
READY_LOOK_S = 30.0         # and once it is
PORT_TIMEOUT_S = 3.0
PROBE_SYSTEM = "Answer with a single word."
PROBE_QUESTION = "Say ready."

CHECKING = "checking"            # before the first look, a moment after the start
NOT_INSTALLED = "not installed"
STOPPED = "stopped"
WAITING = "waiting"
LOADING = "loading"
READY = "ready"
FAILED = "failed"


class ModelSource(Protocol):
    """What `AgentService` asks of where its model is."""
    name: str
    lock: Lock              # held for each request to the model: it answers one at a time
    tools: List[dict]       # the agent's tools, as OpenAI lists them: what the probe carries

    def start(self) -> None: ...

    def status(self) -> dict: ...

    def model(self) -> "Model": ...

    def ask_plain(self, system: str, question: str) -> str: ...

    def failed(self, error: Exception) -> None: ...


class AraConnector:
    def __init__(self, on_change: Optional[Callable[[dict], None]] = None) -> None:
        self.name = NAME
        self.lock = Lock()
        self.tools: List[dict] = []
        self.on_change = on_change or (lambda status: None)
        self._state, self._detail, self._since = "", "", None
        self._wake = Event()
        self._watcher = Thread(target=self._watch, name="connector", daemon=True)

    def start(self) -> None:
        """Watch the connector from now on, on a thread of its own."""
        if not self._watcher.is_alive():
            self._watcher.start()

    @property
    def is_ready(self) -> bool:
        return self._state == READY

    def status(self) -> dict:
        """The state, why, and since when (epoch s) with how long ago -- the page's clock need not be the board's."""
        elapsed = int(time.time() - self._since) if self._since else None
        return {"state": self._state or CHECKING, "detail": self._detail, "since": self._since, "elapsed_s": elapsed,
                "typical_s": LOAD_S, "name": self.name}

    def model(self) -> "Model":
        """strands' OpenAI model on the connector, a tool call written out as text made the call (`repair.py`)."""
        from .repair import RepairingModel
        # max_retries 0: a request the connector failed is not worth two more of a minute each.
        return RepairingModel(self.tools,
                              client_args={"api_key": "unused", "base_url": ARA_URL, "timeout": REQUEST_TIMEOUT_S,
                                           "max_retries": 0},
                              model_id=ARA_MODEL, stream=False,
                              params={"temperature": TEMPERATURE, "top_p": TOP_P,
                                      "extra_body": {"top_k": TOP_K, "repeat_penalty": REPEAT_PENALTY}})

    def ask_plain(self, system: str, question: str) -> str:
        """One question with no tools, the model's answer to it: its choice among words (`finder.py`). Asked as a
        request's tool runs, `lock` held by it; the model answers one at a time anyway."""
        return self._post([{"role": "system", "content": system}, {"role": "user", "content": question}])

    def failed(self, error: Exception) -> None:
        """A request failed: say so, and look again now."""
        self._set(FAILED, f"a request failed ({type(error).__name__}); looking again", None)
        self._wake.set()

    # --- watching ---------------------------------------------------------------------------------------------

    def _watch(self) -> None:
        while True:
            try:
                self._look()
            except Exception as error:          # the watcher must not die: it is the only one
                self._set(FAILED, f"{type(error).__name__}: {error}", None)
            self._wake.wait(READY_LOOK_S if self._state == READY else LOOK_S)
            self._wake.clear()

    def _look(self) -> None:
        if self._state == READY and self.lock.locked():
            return                              # a request is being answered: it is there
        models = _listed_models()
        if models is None:
            self._set(*_local_state())
        elif not models.get(ARA_MODEL):
            self._set(LOADING, "the connector answers, the model is not loaded yet", self._since)
        elif self._state != READY:
            if not self.lock.acquire(blocking=False):
                return
            try:
                self._probe()
                self._set(READY, "", None)
            except Exception as error:
                self._set(FAILED, f"the model did not answer a first request ({type(error).__name__}: {error})",
                          None)
            finally:
                self.lock.release()

    def _probe(self) -> None:
        self._post([{"role": "system", "content": PROBE_SYSTEM}, {"role": "user", "content": PROBE_QUESTION}],
                   self.tools)

    @staticmethod
    def _post(messages: List[dict], tools: Optional[List[dict]] = None) -> str:
        """One request to the connector, with the agent's sampling: the answer's text."""
        body = {"model": ARA_MODEL, "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
                "repeat_penalty": REPEAT_PENALTY, "messages": messages}
        if tools:
            body["tools"] = tools
        request = urllib.request.Request(f"{ARA_URL}/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as reply:
            return json.load(reply)["choices"][0]["message"].get("content") or ""

    def _set(self, state: str, detail: str, since: Optional[float]) -> None:
        is_changed = (state, detail) != (self._state, self._detail)
        self._state, self._detail, self._since = state, detail, since
        if is_changed:
            self.on_change(self.status())


def _listed_models() -> Optional[dict]:
    """The connector's models and whether each is ready; None if its port does not answer, or not in time -- it
    answers nothing else while it generates."""
    try:
        with urllib.request.urlopen(f"{ARA_URL}/models", timeout=PORT_TIMEOUT_S) as reply:
            return {model["id"]: bool(model.get("ready", True)) for model in json.load(reply).get("data", [])}
    except (OSError, ValueError, urllib.error.URLError):
        return None


def _local_state() -> Tuple[str, str, Optional[float]]:
    """What this board says about a connector whose port does not answer: its process, its service, its install."""
    started = _process_started(b"bin/connector")
    if started is not None:
        return LOADING, "loading the model onto the Ara240", started
    active, sub = _service_state()
    if active == "activating":
        return WAITING, ("restarting after a failure" if sub == "auto-restart" else "waiting for the Ara240's proxy"), None
    if active == "failed":
        return FAILED, f"the service failed: journalctl -u {SERVICE}", None
    if not active and not (CONNECTOR_DIR / "venv" / "bin" / "connector").exists():
        return NOT_INSTALLED, "connector/install.sh installs it, with the model (11 GB)", None
    return STOPPED, "installed, not running: connector/run.sh, or connector/install-service.sh at every boot", None


def _service_state() -> Tuple[str, str]:
    """The service's ActiveState and SubState; ("", "") when there is no such service, or no systemd."""
    try:
        shown = subprocess.run(["systemctl", "show", SERVICE, "--property=LoadState,ActiveState,SubState"],
                               capture_output=True, text=True, timeout=PORT_TIMEOUT_S).stdout
    except (OSError, subprocess.SubprocessError):
        return "", ""
    values = dict(line.split("=", 1) for line in shown.splitlines() if "=" in line)
    if values.get("LoadState") != "loaded":
        return "", ""
    return values.get("ActiveState", ""), values.get("SubState", "")


def _process_started(marker: bytes) -> Optional[float]:
    """When the process whose command line holds `marker` started, in epoch seconds; None if none runs."""
    try:
        boot = next(float(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines()
                    if line.startswith("btime "))
    except (OSError, StopIteration):
        return None
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if marker not in (entry / "cmdline").read_bytes():
                continue
            # Field 22, starttime in clock ticks after boot: the 20th after the command name's closing parenthesis.
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            return boot + int(fields[19]) / os.sysconf("SC_CLK_TCK")
        except (OSError, IndexError, ValueError):
            continue
    return None


def describe_status(status: dict) -> str:
    """One line, for the console and the cloud: 'ready', 'stopped: installed, not running: ...', 'loading: ..., since
    21:03:12, about 5:45 in all'."""
    state, detail, since = status["state"], status["detail"], status["since"]
    if state == LOADING and since:
        detail = f"{detail}, since {time.strftime('%H:%M:%S', time.localtime(since))}, about {LOAD_S // 60}:" \
                 f"{LOAD_S % 60:02d} in all"
    return f"{state}: {detail}" if detail else state


def describe_unready(status: dict) -> str:
    """Why a request cannot be put to the model now, in a sentence."""
    state, detail = status["state"], status["detail"]
    if state == NOT_INSTALLED:
        return "The language model is not installed on this board: connector/install.sh, with an Ara240."
    if state == STOPPED:
        return "The language model is installed but not running: connector/run.sh, or connector/install-service.sh."
    if state == WAITING:
        return f"The language model's service is {detail}: it loads the model after that, minutes."
    if state == LOADING:
        left = max(0, int(status["typical_s"] - (time.time() - status["since"]))) if status["since"] else 0
        return ("The language model is loading onto the Ara240" +
                (f": about {left // 60 + 1} min left." if left else ", nearly done."))
    if state == FAILED:
        return f"The language model is not answering: {detail}."
    return "The language model is being looked for: ask again in a moment."
