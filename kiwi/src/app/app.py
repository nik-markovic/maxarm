"""TilesApp: what each command does, and what the stream shows meanwhile -- the one place the app decides anything.

The property-alarm demo's composition-root pattern: the app is handed what it works
with and calls out to it, and nothing calls into it but

    on_command(command)        one command, from C2D or the console        [a command thread]

so the handlers below, read top to bottom, are the whole behaviour of the app:

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
  letters as last seen (`agent_desk()`). Held outside the lock, as its tools
  come back in through `on_command()`; the model takes one sentence at a time,
  and refuses another meanwhile. Its answer goes to the dashboard as `answer`.
- `stop`: what the calibration pages' RESET does, and the model's request too:
  the model cut short, whatever moves the arm cut short, the commands waiting
  for the arm dropped, then the cup lets go, lifts clear and goes home.

The arm is connected on a thread of its own (`keep_arm_connected()`): at the
start, and again RECONNECT_S after it fails or is lost, so the app runs on
without it and a command meanwhile is refused, saying why.

The tile height, the tile offset and the reader are the calibration app's Tiles page's too: one
`appcfg.json`. What the dashboard is told is written into `TelemetryState`, and
`iotc.py` sends it: the cloud is not called from here. One command moves the arm
at a time, under a lock, so a command that comes while another runs waits for it;
`reader-location`, which moves nothing, does not wait.
"""

from __future__ import annotations

from threading import Event, Lock
from typing import TYPE_CHECKING, Optional, Union

from maxarm import MoveResult, TransportError

from applib import commands
from applib.agent.desk import Desk, word_check
from applib.camera.fit import CameraFit
from applib.camera.store import CameraFile
from applib.commands import Command, CommandError
from applib.motion import RECONNECT_S, Motion, Refused
from applib.overlay.stream import StreamOverlay
from applib.redcube import RedCube
from applib.settings import Settings
from applib.telemetry import TelemetryState
from applib.tilework import TileWork
from applib.word import WordRefused

if TYPE_CHECKING:                       # for the type only: the agent needs strands-agents, which may not be there
    from applib.agent import AgentService

READER_WORDS = {"back": "behind the arm", "front": "in front of the arm", "left": "at the arm's left",
                "right": "at the arm's right"}


class TilesApp:
    def __init__(self, motion: Motion, work: TileWork, cube: RedCube, settings: Settings, camera_file: CameraFile,
                 overlay: StreamOverlay, telemetry: TelemetryState) -> None:
        self.motion = motion
        self.work = work                # the tiles: found, read, placed, and a word of them set down
        self.cube = cube                # the red cube found, and the cup brought over it
        self.settings = settings        # the tile height, the tile offset and the reader, config/appcfg.json
        self.camera_file = camera_file  # config/camera.json, the calibration
        self.overlay = overlay          # what the stream shows over the picture
        self.telemetry = telemetry      # written here, sent by iotc.py
        self.agent: Optional["AgentService"] = None     # the language model on the Ara240, set_agent()
        self.arm_problem = ""           # why the arm is not connected; "" once it is: `_set_arm_problem()`
        self.closing = Event()          # the app is stopping: `close()`
        self._lock = Lock()
        self._stops = 0                 # `stop`s so far: a command that waited across one is dropped
        self._handlers = {
            commands.DETECT: self._detect,
            commands.PLACE: self._place,
            commands.FIND_CUBE: self._find_cube,
            commands.READER_LOCATION: self._reader_location,
            commands.AGENT: self._agent,
            commands.STOP: self._stop,
        }
        self._set_arm_problem("connecting")

    def on_command(self, command: Command) -> str:
        """Run one command and return its answer; `CommandError`, saying why, if it cannot be done."""
        handler = self._handlers.get(command.verb)
        if handler is None:
            raise CommandError(f"There is no command {command.verb}.")
        if command.verb in (commands.READER_LOCATION, commands.AGENT):
            return handler(command)
        if command.verb == commands.STOP:
            self._cut_short()                           # before the lock: whatever holds it lets go
        stops = self._stops
        with self._lock:
            if self.closing.is_set():
                raise CommandError("The app is stopping.")
            if self._stops != stops:
                raise CommandError("Dropped: a stop came while it waited for the arm.")
            if not self.motion.is_connected:
                raise CommandError(_sentence(f"the arm is not connected, {self.arm_problem}; it is tried again "
                                             f"every {RECONNECT_S:.0f} s"))
            try:
                return handler(command)
            except (Refused, WordRefused) as reason:
                raise CommandError(_sentence(str(reason))) from None
            except TransportError as error:
                self.motion.is_connected = False
                self.motion.arm.disconnect()            # the port let go at once: plugged in again, the arm takes it
                self._set_arm_problem(f"it stopped answering: {error}")
                raise CommandError(_sentence(f"the arm stopped answering: {error}; it is tried again every "
                                             f"{RECONNECT_S:.0f} s")) from None
            finally:
                self.overlay.set_doing("")

    def keep_arm_connected(self) -> None:
        """Connect the arm, and again RECONNECT_S after each failure or loss, until the app stops. Its own thread."""
        said = None
        while not self.closing.is_set():
            if not self.motion.is_connected:
                with self._lock:
                    try:
                        self.motion.connect()
                        self._set_arm_problem("")
                    except Exception as error:
                        self._set_arm_problem(str(error) or repr(error))
                if self.arm_problem != said:
                    said = self.arm_problem
                    if said:
                        print(f"arm        : {said}; tried again every {RECONNECT_S:.0f} s", flush=True)
                    else:
                        print("arm        : connected and homed", flush=True)
            self.closing.wait(RECONNECT_S)

    def close(self) -> None:
        """The app stopping: the arm held where it is, nothing more started."""
        self.closing.set()
        self.motion.stopping.set()
        self.motion.arm.stop()

    def set_agent(self, agent: Optional["AgentService"]) -> None:
        """The language model, once built: its tools are this app's handlers, so it is built after the app."""
        self.agent = agent

    def agent_desk(self) -> Desk:
        """What the language model's tools do with the tiles: this app's own commands, under its lock."""
        def run(verb: str, argument: str = "") -> str:
            return self.on_command(Command(verb, argument, f"agent: {verb} {argument}".strip()))
        return Desk(seen=self.work.letters_seen, detect=lambda: run(commands.DETECT),
                    check=word_check(self.work, self.settings), place=lambda word: run(commands.PLACE, word))

    def wait_idle(self, timeout_s: float) -> bool:
        """Until no command moves the arm, at most `timeout_s`; whether none does."""
        if not self._lock.acquire(timeout=timeout_s):
            return False
        self._lock.release()
        return True

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
        colour = command.argument.strip().lower()
        if colour not in ("", "red"):
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
        """Under the lock, once what ran has let go of it (`_cut_short()`)."""
        self.motion.stopping.clear()
        self.overlay.set_doing("Stopping: letting go, lifting, going home")
        result = self.motion.reset()
        if not result:
            why = result.detail or result.status.value
            raise CommandError(_sentence(f"stopped, but the arm did not go home: {why}"))
        return "Stopped: the cup let go, lifted clear and went home."

    # --- inside -----------------------------------------------------------------------------------------------

    def _cut_short(self) -> None:
        """The model's request, the arm's move, and the commands waiting for the arm: all cut short."""
        if self.agent is not None:
            self.agent.stop()
        self._stops += 1
        self.motion.stopping.set()
        self.motion.arm.stop()

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

    def _set_arm_problem(self, problem: str) -> None:
        """Why the arm cannot be used, "" once it can: for the commands' refusals and the stream's bar."""
        self.arm_problem = problem
        self.overlay.set_arm_problem(problem)


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
