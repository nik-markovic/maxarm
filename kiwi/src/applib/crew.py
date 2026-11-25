"""The board's other cores: worker processes, for work that one Python process does one piece at a time.

    results = crew.run(task, [(argument, ...), ...], here=the_same_work_done_in_this_process)

`task(*each)` is run for each tuple, in the workers, and the results come back in order. This process works too,
with `here(*each)`: from the last tuple back, taking each the workers have not yet come to, until the two meet. So
whoever is free takes the next piece, and a process slowed by other work -- this one has the camera and the pages
to serve -- takes fewer.

Processes, not threads. The fits (`find.py`) are numpy on small arrays, most of it under the interpreter's lock:
three threads make them no faster than two, and the app's own threads wait on the same lock. Nor do six processes
make six times one's: the cores share little cache. The workers are kept from one call to the next, as
starting them takes seconds.

They are forked from a fork server, a process of its own, and never from this one: a fork of the app would cut its
camera and server threads off mid-step, with whatever locks they held. Where there is no fork server (Windows) or
one core only, there are no workers. When workers fail or fall silent they are given up, a line on stderr says
why, and the next call starts new ones. Either way `run()` does what is left with `here`.

The app lets its workers go as it stops. Killed outright it cannot, and nothing else would tell them: so each
watches for the app's end, and ends with it.

A task and its arguments are pickled to the worker, and its result back: a task is a module's own function.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import sys
import threading
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Callable, List, Optional, Sequence

START_METHOD = "forkserver"
# Every core but one, which this process keeps for its own share and for the camera.
WORKERS = min(5, (os.cpu_count() or 1) - 1) if START_METHOD in multiprocessing.get_all_start_methods() else 0
# A piece of work takes seconds. A worker silent this long is given up.
PATIENCE_S = 120.0

_workers: Optional[ProcessPoolExecutor] = None
_workers_lock = threading.Lock()


def run(task: Callable, arguments: Sequence[tuple], here: Callable) -> list:
    """What `task(*each)` gives for each of `arguments`, in order; `here` is `task` as this process does it."""
    results: list = [None] * len(arguments)
    begun = _begin(task, arguments)
    left = len(arguments)
    while begun and left and begun[left - 1].cancel():       # not yet a worker's: this process's, then
        left -= 1
        results[left] = here(*arguments[left])
    if begun:
        try:
            results[:left] = [each.result(timeout=PATIENCE_S) for each in begun[:left]]
            return results
        except Exception as problem:
            _give_up(problem)
    results[:left] = [here(*each) for each in arguments[:left]]
    return results


def _begin(task: Callable, arguments: Sequence[tuple]) -> Optional[List[Future]]:
    """`task(*each)` handed to the workers for each of `arguments`; None if there are no workers to hand it to."""
    global _workers
    if WORKERS < 1:
        return None
    try:
        with _workers_lock:
            if _workers is None:
                _workers = ProcessPoolExecutor(WORKERS, mp_context=multiprocessing.get_context(START_METHOD),
                                               initializer=_settle_in)
            return [_workers.submit(task, *each) for each in arguments]
    except Exception as problem:
        _give_up(problem)
        return None


def _give_up(problem: Exception) -> None:
    global _workers
    with _workers_lock:
        workers, _workers = _workers, None
    if workers is not None:
        workers.shutdown(wait=False, cancel_futures=True)
    print(f"crew: the worker processes failed, so this process does their work: {type(problem).__name__}: {problem}",
          file=sys.stderr, flush=True)


def _settle_in() -> None:
    """A worker's first step. It leaves interrupts to the app: Ctrl-C reaches every process of the terminal's group,
    and systemd's stop every one of the service; the app stops on it, and lets its workers go. And it watches for
    the app's end."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    threading.Thread(target=_end_with_the_app, daemon=True).start()


def _end_with_the_app() -> None:
    multiprocessing.parent_process().join()
    os._exit(0)
