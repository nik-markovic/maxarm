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
  stands, gripped at the grip height, and detect again.
- `find-cube [red]`: park the arm, find the red cube in the live view and bring
  the cup over it; the stream shows the cube until the next detect or place.
- `reader-location [back|front|left|right]`: where the person reading the word
  stands, kept in `config/appcfg.json`; without one, where they stand now.

The grip height and the reader are the calibration app's Tiles page's too: one
`appcfg.json`. What the dashboard is told is written into `TelemetryState`, and
`iotc.py` sends it: the cloud is not called from here. One command moves the arm
at a time, under a lock, so a command that comes while another runs waits for it;
`reader-location`, which moves nothing, does not wait.
"""

from __future__ import annotations

from threading import Lock
from typing import Union

from maxarm import MoveResult, TransportError

from applib import commands
from applib.camera.fit import CameraFit
from applib.camera.store import CameraFile
from applib.commands import Command, CommandError
from applib.motion import Motion, Refused
from applib.overlay.stream import StreamOverlay
from applib.redcube import RedCube
from applib.settings import Settings
from applib.telemetry import TelemetryState
from applib.tilework import TileWork
from applib.word import WordRefused

READER_WORDS = {"back": "behind the arm", "front": "in front of the arm", "left": "at the arm's left",
                "right": "at the arm's right"}


class TilesApp:
    def __init__(self, motion: Motion, work: TileWork, cube: RedCube, settings: Settings, camera_file: CameraFile,
                 overlay: StreamOverlay, telemetry: TelemetryState) -> None:
        self.motion = motion
        self.work = work                # the tiles: found, read, placed, and a word of them set down
        self.cube = cube                # the red cube found, and the cup brought over it
        self.settings = settings        # the grip height and the reader, config/appcfg.json
        self.camera_file = camera_file  # config/camera.json, the calibration
        self.overlay = overlay          # what the stream shows over the picture
        self.telemetry = telemetry      # written here, sent by iotc.py
        self._lock = Lock()
        self._handlers = {
            commands.DETECT: self._detect,
            commands.PLACE: self._place,
            commands.FIND_CUBE: self._find_cube,
            commands.READER_LOCATION: self._reader_location,
        }
        self._rest()

    def on_command(self, command: Command) -> str:
        """Run one command and return its answer; `CommandError`, saying why, if it cannot be done."""
        handler = self._handlers.get(command.verb)
        if handler is None:
            raise CommandError(f"There is no command {command.verb}.")
        if command.verb == commands.READER_LOCATION:
            return handler(command)
        with self._lock:
            if self.motion.stopping.is_set():
                raise CommandError("The app is stopping.")
            if not self.motion.is_connected:
                raise CommandError("The arm is not connected: restart the app.")
            try:
                return handler(command)
            except (Refused, WordRefused) as reason:
                raise CommandError(_sentence(str(reason))) from None
            except TransportError as error:
                self.motion.is_connected = False
                raise CommandError(f"The arm stopped answering: {error}. Restart the app.") from None
            finally:
                self.overlay.set_doing("")
                self._rest()

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
            return _answer(self.work.detect(self.motion, fit))
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
            _answer(self.work.detect(self.motion, fit))
        finally:
            self._report()
        steps, stamp = self.work.plan(letters, self.settings.get("grip_mm"), self.settings.get("reader"))
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
            self._rest()
            return f"Words are set down for a reader {READER_WORDS[reader]} from now on."
        reader = self.settings.get("reader")
        return f"Words are set down for a reader {READER_WORDS[reader]}: reader-location {', '.join(READER_WORDS)}."

    # --- inside -----------------------------------------------------------------------------------------------

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
        self._rest()

    def _rest(self) -> None:
        """What the stream's bar says while no command runs."""
        found = self.work.found
        tiles = (f"{len(found['tiles'])} tiles: {self.work.summary()['tiles'].upper()}" if found and found["tiles"]
                 else "no tiles found" if found else "waiting for a command")
        self.overlay.set_resting(f"{tiles}   reader {self.settings.get('reader')}   grip "
                                 f"{self.settings.get('grip_mm'):g} mm")


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
