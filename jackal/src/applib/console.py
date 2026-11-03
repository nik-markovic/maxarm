"""The console as a source of commands: a line typed is run as the same command C2D sends, its answer printed.

    Console(service).run()      # until `quit`, Ctrl-D or Ctrl-C; on the main thread

The lines are the device template's commands as text -- `detect`, `place TEAM`,
`find-cube`, `reader-location left` -- parsed by `commands.parse()`. A command
waits for the one before it to finish, as C2D's do, so the console waits for its
answer before it asks for the next line.
"""

from __future__ import annotations

from .commands import VERBS, CommandService

QUIT = ("quit", "exit")
HELP = (f"commands: {', '.join(VERBS)} -- e.g. `place TEAM`, `reader-location left`; "
        f"{' or '.join(QUIT)}, Ctrl-D or Ctrl-C to stop")


class Console:
    def __init__(self, service: CommandService) -> None:
        self.service = service

    def run(self) -> None:
        print(HELP, flush=True)
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                return
            if line.lower() in QUIT:
                return
            if line.lower() in ("help", "?"):
                print(HELP, flush=True)
            elif line:
                result = self.service.submit(line, source="console").result()
                print(f"{'' if result.is_ok else 'refused: '}{result.message}", flush=True)
