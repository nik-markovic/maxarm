#!/usr/bin/env python3
"""The calibration web app. One page per calibration, each a tab; the desk is the first.

    ../.venv/bin/python calibration/calibration.py                  # then open http://localhost:8080
    ../.venv/bin/python calibration/calibration.py --host 0.0.0.0   # on the board, for another machine's browser

The arm is on MAXARM_DEVICE, /dev/ttyUSB0 by default. Connecting resets the
board, which homes the arm: 4 s, sometimes 15. This process is the only thing
that talks to the arm; the page asks it. Ctrl-C holds the arm where it is and
closes the port.
"""

import argparse
import sys
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent))

from maxarm import MaxArm                                    # noqa: E402
from maxarm.spaces import CONFIG_DIR                         # noqa: E402
from common import arm_api                                   # noqa: E402
from common.motion import Motion                             # noqa: E402
from common.server import Server                             # noqa: E402
from common.worker import Worker                             # noqa: E402
from desk.api import DeskSession                             # noqa: E402


def run(host: str, port: int, config_dir: Path = CONFIG_DIR) -> None:
    worker = Worker(Motion(MaxArm(config_dir=None), config_dir))
    server = Server(HERE, home="desk")
    arm_api.register(server, worker)
    DeskSession(worker, config_dir).register(server)
    worker.start()
    print(f"calibration: http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}/"
          f"  -- Ctrl-C to stop", flush=True)
    try:
        server.serve(host, port)
    except KeyboardInterrupt:
        print("\nholding the arm where it is, and closing the port", flush=True)
    finally:
        worker.shutdown()


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1",
                        help="the address to serve on: 0.0.0.0 to reach it from another machine")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv[1:])
    run(args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
