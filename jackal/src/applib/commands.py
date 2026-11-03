"""Commands, whatever they come from, run off the camera's and the cloud's threads (the property-alarm demo's design,
`work/refs/iotc-eiq-imx95-property-alarm`).

Every source funnels through `CommandService`: /IOTCONNECT C2D (`iotc.py`), the
console (`console.py`), and whatever comes next -- speech, an agent. A source that
already has the verb and its argument, as C2D does, builds a `Command` and calls
`submit_command()`; one that has a line of text calls `submit()` and has it
parsed. Both end in the same handler and the same `CommandResult`.

A handler returns the sentence to answer with, or raises `CommandError` with the
reason; both come back as a `CommandResult`, `is_ok` and that text. The console
prints it, C2D acknowledges the command with it, and the stream shows it for a few
seconds -- one answer for every way in.

    service = CommandService(app.on_command, on_result=overlay.show_answer)
    result = service.submit("place TEAM", source="console").result()
    print(result.message)
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# The verbs: the device template's command names (`files/device-template.json`), so a C2D command names one directly.
DETECT = "detect"
PLACE = "place"
FIND_CUBE = "find-cube"
READER_LOCATION = "reader-location"
VERBS = (DETECT, PLACE, FIND_CUBE, READER_LOCATION)

NOT_UNDERSTOOD = f"Not a command: the commands are {', '.join(VERBS)}."


@dataclass(frozen=True)
class Command:
    verb: str
    argument: str = ""
    text: str = ""              # the line as it came, kept for the console


@dataclass(frozen=True)
class CommandResult:
    is_ok: bool
    message: str
    source: str = "console"
    command: Optional[Command] = None


class CommandError(Exception):
    """A command that cannot be carried out; the message, said as it is, is why."""


# How a line of text may name each verb: its words apart, joined, or as the template has them; `reader` alone is
# `reader-location`.
_NAMINGS = {verb: [verb.split("-"), [verb.replace("-", "")], [verb]] for verb in VERBS}
_NAMINGS[READER_LOCATION].append(["reader"])


def parse(text: str) -> Optional[Command]:
    """A line of text as a command -- `place TEAM`, `find cube`, `reader left` -- or None if it is none. Everything
    after the verb is its argument."""
    words = text.split()
    lowered = [word.lower() for word in words]
    for verb, namings in _NAMINGS.items():
        for naming in namings:
            if lowered[:len(naming)] == naming:
                return Command(verb, " ".join(words[len(naming):]), text.strip())
    return None


class CommandService:
    """Runs commands on a pool of threads. The handler keeps them from running into each other (`app.py`)."""

    def __init__(self, handler: Callable[[Command], str], max_workers: int = 2,
                 on_result: Optional[Callable[[CommandResult], None]] = None) -> None:
        self.handler = handler
        # Told of every finished command, whatever its source: the stream's answer line hangs off this.
        self.on_result = on_result or (lambda result: None)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="command")

    def submit(self, text: str, source: str = "console") -> "Future[CommandResult]":
        """A line of text, parsed and run."""
        return self._executor.submit(self.run, text, source)

    def submit_command(self, command: Command, source: str = "c2d") -> "Future[CommandResult]":
        """A command its source has already named the verb of."""
        return self._executor.submit(self.execute, command, source)

    def run(self, text: str, source: str = "console") -> CommandResult:
        command = parse(text)
        if command is None:
            print(f"[cmd] {source}: {text.strip()!r} is not a command", flush=True)
            return self._finish(CommandResult(False, NOT_UNDERSTOOD, source))
        return self.execute(command, source)

    def execute(self, command: Command, source: str = "c2d") -> CommandResult:
        """Run one command: the only place a handler is called, whatever the source."""
        print(f"[cmd] {source}: {command.verb}{' ' + repr(command.argument) if command.argument else ''}", flush=True)
        try:
            return self._finish(CommandResult(True, self.handler(command), source, command))
        except CommandError as error:
            print(f"[cmd] refused: {error}", flush=True)
            return self._finish(CommandResult(False, str(error), source, command))
        except Exception as error:              # a bug in a handler must not take the app down
            logger.exception("command %s raised", command.verb)
            return self._finish(CommandResult(False, f"Something went wrong: {type(error).__name__}: {error}",
                                              source, command))

    def _finish(self, result: CommandResult) -> CommandResult:
        try:
            self.on_result(result)
        except Exception:                        # the stream's answer line is the least of it
            logger.exception("the on_result hook raised")
        return result

    def stop(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
