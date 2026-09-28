#!/usr/bin/env python3
"""Dump the board's own espmax.py and __espmax.mpy over the REPL.

Read-only and motionless: it lists the filesystem and streams file bytes back
as base64. Nothing is written to the board and no name is bound in its
namespace -- every command is a throwaway expression, same contract as
`maxarm_link`.

Why bother when `MaxArm/` ships the same sources: the kit's `__espmax.mpy`
raises "Unreachable position x:...", while the live board prints the required
reach and the link lengths instead. That is a different build, so the kit copy
cannot be assumed to be what is actually running.

Output lands in `anaconda/agenttools/boarddump/`.
"""

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "datacollection"))

from maxarm_link import MaxArmLink  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "boarddump"

# 192 raw bytes per round trip. Small enough that the base64 reply and the
# repr() around it both stay clear of v1.12's fragmented heap.
CHUNK = 192


def list_board_files(link: MaxArmLink):
    return link.evaluate("__import__('os').listdir()")


def read_board_file(link: MaxArmLink, name: str) -> bytes:
    """Stream one file back in base64 chunks."""
    size = link.evaluate(f"__import__('os').stat({name!r})[6]")
    parts = []
    for offset in range(0, size, CHUNK):
        payload = link.evaluate(
            "(lambda f: (f.seek({off}), "
            "__import__('ubinascii').b2a_base64(f.read({n})), f.close())[1])"
            "(open({name!r}, 'rb'))".format(off=offset, n=CHUNK, name=name)
        )
        parts.append(base64.b64decode(payload))
    return b"".join(parts)


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    with MaxArmLink() as link:
        listing = list_board_files(link)
        print("board filesystem:", listing)
        (OUT_DIR / "listing.txt").write_text("\n".join(sorted(listing)) + "\n")

        for name in sorted(listing):
            if not (name.endswith(".py") or name.endswith(".mpy")):
                continue
            blob = read_board_file(link, name)
            (OUT_DIR / name).write_bytes(blob)
            print(f"  {name}: {len(blob)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
