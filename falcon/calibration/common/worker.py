"""The one thread that talks to the arm.

Requests arrive on the web server's threads. Each becomes a command, and the
commands run here one at a time, in order; the request waits for its command
and returns how it went. A slider sends a command per change, so a command
marked `latest_only` that is still waiting when another of its kind arrives is
dropped -- "superseded" -- and only the newest runs. RESET cuts short whatever
is running, drops everything waiting, and sends the arm home.

After a command that used the arm, and with nothing else waiting, the servos
are read: the page shows where the arm says it is next to where it was sent,
because a small step is not always a step taken.
"""

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Optional, Tuple, Union

from maxarm import MoveResult, TransportError

from .motion import Motion

Action = Callable[[Motion], Union[MoveResult, str, None]]


class Refused(Exception):
    """A command that was not carried out, and nothing moved. The message says why."""


@dataclass
class Command:
    kind: str
    label: str
    action: Action
    is_latest_only: bool
    uses_arm: bool
    done: threading.Event = field(default_factory=threading.Event)
    outcome: dict = field(default_factory=dict)

    def finish(self, status: str, detail: str = "") -> None:
        self.outcome = {"status": status, "detail": detail}
        self.done.set()


class Worker:
    def __init__(self, motion: Motion) -> None:
        self.motion = motion
        self.queue: Deque[Command] = deque()
        self.condition = threading.Condition()
        self.doing: Optional[str] = None
        self.last: dict = {}
        # Commands that used the arm, counted as they start: a page can tell
        # whether anything else has moved the arm since its own last command.
        self.count = 0
        self.connection = {"status": "connecting", "detail": ""}
        self.is_running = True
        self.thread = threading.Thread(target=self._run, name="arm", daemon=True)

    def start(self) -> None:
        self.thread.start()
        self.connect(wait=False)

    def run(self, kind: str, label: str, action: Action, *, latest_only: bool = False,
            uses_arm: bool = True) -> dict:
        """Queue `action(motion)`, wait for it, and say how it went: status, detail, and the state after."""
        command = Command(kind, label, action, latest_only, uses_arm)
        with self.condition:
            if latest_only:
                for waiting in [queued for queued in self.queue if queued.kind == kind]:
                    self.queue.remove(waiting)
                    waiting.finish("superseded")
            self.queue.append(command)
            self.condition.notify()
        command.done.wait()
        return command.outcome

    def connect(self, wait: bool = True) -> dict:
        def connect(motion: Motion) -> MoveResult:
            self.connection = {"status": "connecting", "detail": ""}
            try:
                result = motion.connect()
            except Exception as error:
                self.connection = {"status": "failed", "detail": str(error)}
                raise
            self.connection = {"status": "ready", "detail": ""}
            return result
        if not wait:
            threading.Thread(target=self.connect, daemon=True).start()
            return {"status": "queued", "detail": ""}
        return self.run("connect", "connecting: the board resets and homes the arm", connect)

    def reset(self) -> dict:
        with self.condition:
            for waiting in self.queue:
                waiting.finish("superseded", "RESET")
            self.queue.clear()
            self.motion.stopping.set()
            self.motion.arm.stop()
        return self.run("reset", "RESET: letting go, lifting, going home", Motion.reset)

    def shutdown(self) -> None:
        """Hold the arm where it is, finish, and close the port."""
        with self.condition:
            self.is_running = False
            for waiting in self.queue:
                waiting.finish("superseded", "shutting down")
            self.queue.clear()
            self.motion.stopping.set()
            self.motion.arm.stop()
            self.condition.notify_all()
        self.thread.join(timeout=10.0)
        if self.motion.is_connected:
            self.motion.arm.disconnect()

    def state(self) -> dict:
        motion = self.motion
        return {
            "connection": dict(self.connection),
            "doing": self.doing,
            "queued": len(self.queue),
            "space": motion.space.value,
            "sent": list(motion.sent),
            "measured": list(motion.measured) if motion.measured else None,
            "cup": motion.cup,
            "suction": motion.is_suction_on,
            "last": dict(self.last),
        }

    def _run(self) -> None:
        while True:
            with self.condition:
                while self.is_running and not self.queue:
                    self.condition.wait()
                if not self.is_running:
                    return
                command = self.queue.popleft()
                self.motion.stopping.clear()
                self.doing = command.label
                if command.uses_arm:
                    self.count += 1
            self._execute(command)
            with self.condition:
                is_idle = not self.queue
            if command.uses_arm and self.motion.is_connected and is_idle:
                self.doing = "reading where the arm is"
                self.motion.read()
            self.doing = None

    def _execute(self, command: Command) -> None:
        try:
            if command.uses_arm and command.kind != "connect" and not self.motion.is_connected:
                raise Refused("the arm is not connected"
                              + (f": {self.connection['detail']}" if self.connection["detail"] else ""))
            status, detail = _outcome(command.action(self.motion))
        except Refused as reason:
            status, detail = "refused", str(reason)
        except Exception as error:
            status, detail = "error", str(error) or repr(error)
            if isinstance(error, TransportError):
                self.motion.is_connected = False
                self.connection = {"status": "failed", "detail": detail}
        self.last = {"label": command.label, "status": status, "detail": detail}
        command.finish(status, detail)


def _outcome(result: Union[MoveResult, str, None]) -> Tuple[str, str]:
    if isinstance(result, MoveResult):
        return result.status.value, result.detail
    return "done", result or ""
