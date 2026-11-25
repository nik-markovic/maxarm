"""TilesApp: what each command does, and what the stream shows meanwhile -- the one place the app decides anything.

The property-alarm demo's composition-root pattern: the app is handed what it works
with and calls out to it, and nothing calls into it but

    on_command(command)        one command, from C2D or the console        [a command thread]

so the handlers below, read top to bottom, are the whole behavior of the app:

- `detect`: park the arm, take the snapshot, find the tiles and read their
  letters; the stream shows them, and the dashboard is told the letters and which
  tiles are out of reach or too close to another.
- `place LETTERS`: detect first, as a precaution -- tiles may have been moved by
  hand -- then set the letters down along the dropoff arc for where the reader
  stands, each tile gripped the tile offset under its top, and detect again.
- `find-cube [red]`: park the arm, find the red cube in the live view and bring
  the cup over it; the stream shows the cube until the next detect or place.
- `reader-location [back|front|left|right]`: where the person reading the word
  stands, kept in `config/appcfg.json`; without one, where they stand now.
- `agent SENTENCE`: the sentence to the language model on the Ara240
  (`applib/agent`), whose tools are these handlers -- `detect`, and `place`,
  which it starts on a thread of its own and does not wait for -- and the
  letters as last seen (`agent_desk()`). Not on the arm's worker, as its tools
  come back in through `on_command()`; the model takes one sentence at a time,
  the Chat page's too, and refuses another meanwhile. Its answer goes to the
  dashboard as `answer`.
- `stop`: the calibration pages' RESET, and the model's request too: the model
  cut short, whatever moves the arm cut short -- a page's move as well -- the
  commands waiting for the arm dropped, then the cup lets go, lifts clear and
  goes home.

The arm is the calibration pages' too: every command that moves it is a command
of theirs, on the one thread that talks to the arm (`applib/worker.py`), so a
command waits for what a page asked first, and the other way round. The worker
connects the arm at the start and again RECONNECT_S after it fails or is lost;
the app runs on without it, and a command meanwhile is refused, saying why.
`reader-location`, which moves nothing, does not wait.

The tiles, the tile height, the tile offset and the reader are the Tiles page's
too: one `TileWork`, one `appcfg.json`. What the dashboard is told is written
into `TelemetryState`, and `iotc.py` sends it: the cloud is not called from here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple, Union

from maxarm import MoveResult

from applib import commands
from applib.agent.desk import Desk, word_check
from applib.camera.fit import CameraFit
from applib.camera.store import CameraFile
from applib.commands import Command, CommandError
from applib.overlay.stream import StreamOverlay
from applib.redcube import RedCube
from applib.settings import Settings
from applib.telemetry import TelemetryState
from applib.tilework import TileWork
from applib.worker import READING, Worker

if TYPE_CHECKING:                       # for the type only: the agent needs strands-agents, which may not be there
    from applib.agent import AgentService

READER_WORDS = {"back": "behind the arm", "front": "in front of the arm", "left": "at the arm's left",
                "right": "at the arm's right"}


class TilesApp:
    def __init__(self, worker: Worker, work: TileWork, cube: RedCube, settings: Settings, camera_file: CameraFile,
                 overlay: StreamOverlay, telemetry: TelemetryState) -> None:
        self.worker = worker            # the one thread that moves the arm, the pages' too
        self.motion = worker.motion     # the handlers' arm: they run on the worker's thread
        self.work = work                # the tiles: found, read, placed, and a word of them set down
        self.cube = cube                # the red cube found, and the cup brought over it
        self.settings = settings        # the tile height, the tile offset and the reader, config/appcfg.json
        self.camera_file = camera_file  # config/camera.json, the calibration
        self.overlay = overlay          # what the stream shows over the picture
        self.telemetry = telemetry      # written here, sent by iotc.py
        self.agent: Optional["AgentService"] = None     # the language model on the Ara240, set_agent()
        self._handlers = {
            commands.DETECT: self._detect,
            commands.PLACE: self._place,
            commands.FIND_CUBE: self._find_cube,
            commands.READER_LOCATION: self._reader_location,
            commands.AGENT: self._agent,
            commands.STOP: self._stop,
        }
        overlay.watch_arm(self.arm_state)

    def on_command(self, command: Command) -> str:
        """Run one command and return its answer; `CommandError`, saying why, if it cannot be done."""
        handler = self._handlers.get(command.verb)
        if handler is None:
            raise CommandError(f"There is no command {command.verb}.")
        if command.verb in (commands.READER_LOCATION, commands.AGENT, commands.STOP):
            return handler(command)
        outcome = self.worker.run(command.verb, f"command: {command.text or command.verb}",
                                  lambda motion: self._on_arm(handler, command))
        if outcome["status"] == "superseded":
            raise CommandError(f"Dropped: {outcome['detail'] or 'a stop'} came while it waited for the arm.")
        if outcome["status"] != "done":
            raise CommandError(_sentence(outcome["detail"] or outcome["status"]))
        return outcome["detail"]

    def arm_state(self) -> Tuple[str, str]:
        """What the worker is doing with the arm, for whichever page or command, and why the arm cannot be used:
        "connecting", the failure, or "" once it is connected. For the stream's bar."""
        doing = self.worker.doing or ""
        connection = self.worker.connection
        problem = {"ready": "", "connecting": "connecting"}.get(connection["status"], connection["detail"])
        return ("" if doing == READING else doing), problem

    def set_agent(self, agent: Optional["AgentService"]) -> None:
        """The language model, once built: its tools are this app's handlers, so it is built after the app."""
        self.agent = agent

    def agent_desk(self) -> Desk:
        """What the language model's tools do with the tiles: this app's own commands, on the arm's worker."""
        def run(verb: str, argument: str = "") -> str:
            return self.on_command(Command(verb, argument, f"agent: {verb} {argument}".strip()))
        return Desk(seen=self.work.letters_seen, detect=lambda: run(commands.DETECT),
                    check=word_check(self.work, self.settings), place=lambda word: run(commands.PLACE, word))

    # --- the commands -----------------------------------------------------------------------------------------

    def _detect(self, command: Command) -> str:
        fit = self._calibration()
        self.overlay.show_cube(False)
        self.overlay.set_doing("Detecting the tiles: the arm parks, the camera takes its snapshot")
        try:
            return _answer(self.work.detect(self.motion, fit, self.settings.get("tile_height_mm")))
        finally:
            self._report()

    def _place(self, command: Command) -> str:
        letters = "".join(command.argument.split()).upper()
        if not letters:
            raise CommandError("Place which letters? Like: place TEAM.")
        fit = self._calibration()
        self.overlay.show_cube(False)
        self.overlay.set_doing(f"Placing {letters}: finding the tiles first")
        try:
            _answer(self.work.detect(self.motion, fit, self.settings.get("tile_height_mm")))
        finally:
            self._report()
        steps, stamp = self.work.plan(letters, self.settings.get("tile_offset_mm"), self.settings.get("reader"))
        self.overlay.set_doing(f"Placing {letters}")
        try:
            return _answer(self.work.place(self.motion, steps, stamp, fit))
        finally:
            self._report()

    def _find_cube(self, command: Command) -> str:
        color = command.argument.strip().lower()
        if color not in ("", "red"):
            raise CommandError(f"Only the red cube is found, not {command.argument.strip()}.")
        fit = self._calibration()
        self.overlay.set_doing("Finding the red cube: the arm parks, the camera looks")
        self.overlay.show_cube(True)
        return _answer(self.cube.find(self.motion, fit))

    def _reader_location(self, command: Command) -> str:
        reader = command.argument.strip().lower()
        if reader:
            try:
                self.settings.set(reader=reader)
            except ValueError as reason:
                raise CommandError(_sentence(str(reason))) from None
            return f"Words are set down for a reader {READER_WORDS[reader]} from now on."
        reader = self.settings.get("reader")
        return f"Words are set down for a reader {READER_WORDS[reader]}: reader-location {', '.join(READER_WORDS)}."

    def _agent(self, command: Command) -> str:
        if self.agent is None:
            raise CommandError("The language model is off: strands-agents is not installed (src/requirements.txt).")
        sentence = command.argument.strip()
        if not sentence:
            raise CommandError("Ask what? Like: agent a 7-letter word.")
        answer = self.agent.ask(sentence, source="app")
        self.telemetry.set_once(answer=answer)
        return answer

    def _stop(self, command: Command) -> str:
        """Not behind the commands it stops: the worker's RESET cuts short what runs and drops what waits."""
        if self.agent is not None:
            self.agent.stop()
        outcome = self.worker.reset()
        if outcome["status"] != "done":
            why = outcome["detail"] or outcome["status"]
            raise CommandError(_sentence(f"stopped, but the arm did not go home: {why}"))
        return "Stopped: the cup let go, lifted clear and went home."

    # --- inside -----------------------------------------------------------------------------------------------

    def _on_arm(self, handler, command: Command) -> str:
        """A command's handler, on the worker's thread; the bar's doing cleared after it, before the next runs."""
        try:
            return handler(command)
        finally:
            self.overlay.set_doing("")

    def _calibration(self) -> CameraFit:
        """The camera calibration, or why the tiles cannot be found with it."""
        fit, problem = self.camera_file.get()
        problem = problem or self.work.view_problem(fit)
        if problem:
            raise CommandError(_sentence(problem))
        return fit

    def _report(self) -> None:
        """The tiles as last found, for the dashboard now and on the stream's bar."""
        if self.work.found:
            self.telemetry.set(**self.work.summary())
            self.telemetry.wake()


def _answer(result: Union[MoveResult, str]) -> str:
    """A move's outcome as the command's answer; `CommandError` unless it was done."""
    if isinstance(result, MoveResult):
        if not result:
            raise CommandError(_sentence(result.detail or result.status.value))
        return _sentence(result.detail)
    return _sentence(result)


def _sentence(text: str) -> str:
    """A capital at the start and a full stop at the end."""
    text = text.strip()
    if not text:
        return "Done."
    return text[0].upper() + text[1:] + ("" if text.endswith((".", "?", "!")) else ".")
