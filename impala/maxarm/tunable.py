#!/usr/bin/env python3
"""The handful of constants a user may need to change, without editing code.

Each one lives in the module that uses it, under its own name, and the
environment variable that overrides it is that name with `MAXARM_` in front:

    SUCTION_RELEASE_MS = tunable("SUCTION_RELEASE_MS", 3000.0)
    # MAXARM_SUCTION_RELEASE_MS=4000 ./main.py

The prefix exists only in the environment, so the name in the code and the
variable that changes it cannot drift apart.
"""

import os


def tunable(name: str, default: float) -> float:
    """`default`, unless `MAXARM_<name>` is set. A value that is not a number stops the import."""
    text = os.environ.get(f"MAXARM_{name}")
    if text is None:
        return default
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"MAXARM_{name}={text!r} is not a number") from None
