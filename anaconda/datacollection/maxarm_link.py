#!/usr/bin/env python3
"""Transport to the MaxArm's live MicroPython REPL on /dev/ttyUSB0.

The stock board's main.py arms a timer ISR and then falls through to the REPL,
so `arm`, `nozzle` and `robot` stay live globals we can drive from here. The
0xAA 0x55 binary protocol of Chapter 10 is NOT parsed on this port.

Nothing is written to the board: no files, no flash, and no names defined in
its namespace. Every command is a throwaway expression evaluated in the raw
REPL, which lives entirely in RAM.

The per-step readback is what makes a reach test honest: ESPMax's
set_servo_in_range() silently clamps servo 2 and 3 pulses and still reports
success, so the returned bool alone cannot be trusted.
"""

import ast
import time
from typing import List, Optional, Tuple

import serial

DEVICE = "/dev/ttyUSB0"
BAUD = 115200

CTRL_A = b"\x01"
CTRL_B = b"\x02"
CTRL_C = b"\x03"
CTRL_D = b"\x04"

RAW_REPL_BANNER = b"raw REPL"

# The board's compiled IK prints diagnostics to stdout when a target cannot be
# solved -- the required reach and the link lengths, one line per failure. So
# stdout is not ours alone, and our own result has to be picked out by marker.
RESULT_MARKER = "#R#"

Position = Tuple[float, float, float]

# Move, let the servos settle, then report the commanded verdict next to the
# measured pose -- in one round trip and without binding a single name on the
# board. The lambda holds set_position()'s result while the tuple's left-to-right
# evaluation does the settle delay before the readback.
PROBE_EXPRESSION = (
    "(lambda ok: (time.sleep_ms({wait}), (ok, arm.read_position()))[1])"
    "(arm.set_position(({x}, {y}, {z}), {ms}))"
)


class ReplError(RuntimeError):
    """The board returned a traceback or nothing parseable."""


class MaxArmLink:
    """Raw-REPL session against the arm. Use as a context manager."""

    def __init__(self, device: str = DEVICE, baud: int = BAUD, timeout: float = 5.0) -> None:
        self.device = device
        self.baud = baud
        self.timeout = timeout
        self.port: Optional[serial.Serial] = None

    def __enter__(self) -> "MaxArmLink":
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def open(self) -> None:
        port = serial.Serial()
        port.port = self.device
        port.baudrate = self.baud
        port.timeout = 1.0
        # These are set low to *try* to avoid the EN/BOOT auto-reset, but measured
        # on 2026-09-22 it does not work: opening this port resets the ESP32
        # anyway (`rst:0x1 (POWERON_RESET)` in the boot log). main.py then re-runs,
        # which calls arm.go_home() -- so **every connection homes the arm**.
        # Budget ~4 s before the board answers, and expect motion on connect.
        port.dtr = False
        port.rts = False
        port.open()
        port.dtr = False
        port.rts = False
        self.port = port
        self._wait_for_prompt()
        self._enter_raw_repl()
        # main.py imports time at module scope, so it is normally already a live
        # global. Checking is read-only; importing it here would bind a name.
        self.is_board_time_available = bool(self.evaluate("'time' in globals()"))

    def close(self) -> None:
        if self.port is None:
            return
        try:
            self.port.write(CTRL_B)  # back to the friendly REPL for the next user
            self.port.flush()
        finally:
            self.port.close()
            self.port = None

    def _require_port(self) -> serial.Serial:
        if self.port is None:
            raise ReplError("link is not open")
        return self.port

    # Opening the port hard-resets the ESP32, so main.py re-runs on every connect.
    # Measured over 5 opens: `Please wait...` at a steady 2.89 s, then the prompt at
    # 3.27 s -- except one run in five, which stalled ~10 s longer (13.29 s total),
    # apparently a timeout inside BLE init. The old 5 s budget therefore expired
    # mid-init roughly one connect in five, and _enter_raw_repl()'s Ctrl-C then
    # landed on a half-built main.py, leaving `ble` and the timer ISR undefined.
    PROMPT_TIMEOUT = 25.0

    def _wait_for_prompt(self) -> None:
        """Wait out the port-open re-init. Do not send anything until it lands."""
        port = self._require_port()
        # No newline first: writing into a booting board is what caused the trouble.
        banner = self._read_until(b">>>", timeout=self.PROMPT_TIMEOUT)
        if b">>>" not in banner:
            # Fall back to nudging it, in case we attached after main.py had settled.
            port.write(b"\r\n")
            banner = self._read_until(b">>>", timeout=5.0)
            if b">>>" not in banner:
                raise ReplError(
                    f"no prompt within {self.PROMPT_TIMEOUT}s of opening the port; "
                    f"last reply: {banner[-200:]!r}"
                )
        port.reset_input_buffer()

    def _read_until(self, needle: bytes, timeout: float) -> bytes:
        port = self._require_port()
        chunks: List[bytes] = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            pending = port.read(port.in_waiting or 1)
            if pending:
                chunks.append(pending)
                if needle in b"".join(chunks):
                    break
            else:
                time.sleep(0.02)
        return b"".join(chunks)

    def _enter_raw_repl(self, attempts: int = 3) -> None:
        """Switch to the classic raw REPL and prove it took.

        v1.12 predates raw-paste mode, so this is the plain Ctrl-A protocol.
        Confirming the banner matters: if Ctrl-A does not take, the friendly
        REPL line-edits everything sent after it into garbage.
        """
        port = self._require_port()
        for _ in range(attempts):
            port.write(CTRL_C)
            time.sleep(0.2)
            port.write(CTRL_C)  # a second one interrupts a running loop
            time.sleep(0.2)
            port.reset_input_buffer()
            port.write(CTRL_A)
            banner = self._read_until(RAW_REPL_BANNER, timeout=2.0)
            if RAW_REPL_BANNER in banner:
                self._read_until(b">", timeout=1.0)
                return
        raise ReplError(f"board did not enter raw REPL; last reply: {banner!r}")

    def run(self, code: str) -> str:
        """Execute code in the raw REPL and return its stdout."""
        port = self._require_port()
        port.reset_input_buffer()
        port.write(code.encode() + CTRL_D)
        port.flush()

        # The board answers OK, stdout, 0x04, stderr, 0x04, then the raw prompt.
        raw = self._read_until(b"\x04>", timeout=self.timeout)
        if b"OK" not in raw:
            raise ReplError(f"no acknowledgement from board: {raw!r}")
        if b"\x04>" not in raw:
            raise ReplError(f"timed out mid-reply after {self.timeout}s: {raw!r}")
        body = raw.split(b"OK", 1)[1]
        fields = body.split(CTRL_D)
        stdout = fields[0].decode("utf-8", "replace").strip()
        stderr = fields[1].decode("utf-8", "replace").strip() if len(fields) > 1 else ""
        if stderr:
            raise ReplError(f"board raised:\n{stderr}")
        return stdout

    def evaluate(self, expression: str):
        """Evaluate a board expression and return it as a Python literal.

        The result is tagged with a marker because the board's IK writes its own
        diagnostics to stdout on unsolvable targets, and those would otherwise
        be parsed as the answer.

        A MemoryError here usually means the board's heap is fragmented rather
        than genuinely full, so one collection and retry is worth trying.
        `gc` is already imported by the board's main.py, so this binds nothing.
        """
        code = f"print('{RESULT_MARKER}', repr({expression}))"
        try:
            text = self.run(code)
        except ReplError as exc:
            if "MemoryError" not in str(exc):
                raise
            self.run("gc.collect()")
            text = self.run(code)

        payload = None
        for line in text.splitlines():
            if line.startswith(RESULT_MARKER):
                payload = line[len(RESULT_MARKER):].strip()
        if payload is None:
            raise ReplError(f"no result line in reply {text!r}")
        try:
            return ast.literal_eval(payload)
        except (SyntaxError, ValueError) as exc:
            raise ReplError(f"unparseable result {payload!r} in reply {text!r}") from exc

    def get_position(self) -> Optional[Position]:
        """Servo-feedback pose in mm, or None if the bus servos did not answer."""
        measured = self.evaluate("arm.read_position()")
        return tuple(float(v) for v in measured) if measured else None

    def get_cached_position(self) -> Position:
        """Last pose the board *believes* it commanded (no servo read)."""
        return tuple(float(v) for v in self.evaluate("arm.position"))

    def is_solvable(self, position: Position) -> bool:
        """Cheap no-motion IK check. Ignores servo clamping, so advisory only.

        verify_position() feeds x straight into the solver while set_position()
        negates it first, so the sign is flipped here to match a real move.
        """
        x, y, z = position
        return bool(self.evaluate(f"arm.verify_position({-x}, {y}, {z})"))

    def probe(
        self,
        position: Position,
        duration_ms: int,
        settle_ms: int,
    ) -> Tuple[Optional[bool], Optional[Position]]:
        """Command one move and read the pose back. Returns (verdict, measured).

        verdict is True on accepted, False on refused, None inside the board's
        50 mm blind cylinder. measured is None when the servo read failed.
        """
        x, y, z = position
        wait = int(duration_ms) + int(settle_ms)
        if self.is_board_time_available:
            verdict, measured = self.evaluate(PROBE_EXPRESSION.format(
                x=x, y=y, z=z, ms=int(duration_ms), wait=wait))
        else:
            # Two round trips, settling on this side. Slower, still touches nothing.
            verdict = self.evaluate(f"arm.set_position(({x}, {y}, {z}), {int(duration_ms)})")
            time.sleep(wait / 1000.0)
            measured = self.evaluate("arm.read_position()")
        pose = tuple(float(v) for v in measured) if measured else None
        return verdict, pose

    def go_home(self, duration_ms: int = 2000) -> None:
        self.run(f"arm.go_home({int(duration_ms)})")
        time.sleep(duration_ms / 1000.0 + 0.3)
