"""The conversation with the language model, kept twice: in full for the Chat page and the console, and as one or two
lines the model reads at the start of every request (`recap()`). With each answer, what went to and from the model
for it, for the page to show on request (`trace()`).

    chat = Chat()
    turn = chat.begin_turn("a 7-letter word")       # a request: after VISIT_QUIET_S of quiet, a new visitor's
    chat.add("user", "a 7-letter word", source="page")
    chat.recap()            # 'Earlier: 2 minutes ago, for "go", you set down THEOREM. Set down before it: CAT.'

The model is given no history: its window is 4096 tokens, prompt and answer together, and the tools' descriptions
take a share of every prompt. What it is told instead is a short log of what happened, with how long ago -- the
last request, the word it chose and what became of it, and the words set down before. Enough for "explain it",
"place it after all" or "another one", which need the last word, and for not repeating one; a few dozen tokens.

A booth has one visitor after another: after VISIT_QUIET_S without a request, or `forget()`, the recap starts
again. The pages' messages stay until `forget()`, the newest MAX_MESSAGES. RAM only: after a restart nothing here
is true.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import List, Optional

VISIT_QUIET_S = 600.0
MAX_MESSAGES = 60
MAX_TURNS = 20            # a visit's requests kept: their words are not set down twice
REQUEST_CHARS = 60          # a request, as the recap quotes it

# What became of a turn's word.
PROPOSED = "proposed, not set down"
PLACING = "being set down"
PLACED = "set down"
NOT_PLACED = "not set down"


@dataclass
class Message:
    n: int
    role: str                   # "user", "assistant", "step" (a tool call), "event" (the app's own news)
    text: str
    at: float                   # time.time()
    data: dict = field(default_factory=dict)
    trace: List[dict] = field(default_factory=list)    # an answer's: what went to and from the model, `trace()`

    def to_json(self) -> dict:
        """Without the trace, asked for apart: the pages poll the messages, and a trace is a few KB."""
        return {"n": self.n, "role": self.role, "text": self.text, "at": self.at, "has_trace": bool(self.trace),
                **self.data}


@dataclass
class Turn:
    request: str
    at: float                   # time.monotonic()
    word: str = ""
    status: str = ""            # PROPOSED, PLACING, PLACED or NOT_PLACED once there is a word


class Chat:
    def __init__(self) -> None:
        self._lock = Lock()
        self._messages: List[Message] = []
        self._count = 0
        self._turns: List[Turn] = []

    # --- the pages' messages ----------------------------------------------------------------------------------

    def add(self, role: str, text: str, trace: Optional[List[dict]] = None, **data) -> Message:
        with self._lock:
            self._count += 1
            message = Message(self._count, role, text, time.time(), dict(data), list(trace or []))
            self._messages.append(message)
            del self._messages[:-MAX_MESSAGES]
            return message

    def update(self, message: Message, text: Optional[str] = None, **data) -> None:
        with self._lock:
            if text is not None:
                message.text = text
            message.data.update(data)

    def messages(self) -> List[dict]:
        with self._lock:
            return [message.to_json() for message in self._messages]

    def trace(self, n: int) -> Optional[List[dict]]:
        """What went to and from the model for message `n`, an answer; None if it has none or is no longer kept."""
        with self._lock:
            message = next((message for message in self._messages if message.n == n), None)
            return list(message.trace) if message is not None and message.trace else None

    def forget(self) -> None:
        """A new visitor: nothing said before is shown or recalled."""
        with self._lock:
            self._messages.clear()
            self._turns.clear()

    # --- what the model is told -------------------------------------------------------------------------------

    def begin_turn(self, request: str) -> Turn:
        now = time.monotonic()
        with self._lock:
            if self._turns and now - self._turns[-1].at > VISIT_QUIET_S:
                self._turns.clear()
            turn = Turn(" ".join(request.split()), now)
            self._turns.append(turn)
            del self._turns[:-MAX_TURNS]
            return turn

    def set_word(self, turn: Turn, word: str, status: str) -> None:
        with self._lock:
            turn.word, turn.status = word, status

    def previous_request(self) -> str:
        """The request before the one being answered, this visit; "" if there is none."""
        with self._lock:
            return self._turns[-2].request if len(self._turns) > 1 else ""

    def words_used(self) -> List[str]:
        """The words set down this visit, oldest first."""
        with self._lock:
            return [turn.word for turn in self._turns if turn.word and turn.status in (PLACING, PLACED)]

    def recap(self) -> str:
        """The last word of the turns before the one being answered; "" for a new visitor. Named as the system prompt
        names it, spelled out for "explain". The words before it are not named."""
        now = time.monotonic()
        with self._lock:
            earlier = [turn for turn in self._turns[:-1] if now - turn.at <= VISIT_QUIET_S]
        if not earlier:
            return ""
        last = next((turn for turn in reversed(earlier) if turn.word), None)
        if last is None:
            return f'Before: "{_quoted(earlier[-1].request)}", {describe_age(now - earlier[-1].at)}; no word chosen.'
        return (f"The last word: {last.word} ({'-'.join(last.word)}), {last.status}, {describe_age(now - last.at)}, "
                f'for "{_quoted(last.request)}".')


def _quoted(request: str) -> str:
    return request if len(request) <= REQUEST_CHARS else request[:REQUEST_CHARS - 3] + "..."


def describe_age(seconds: float) -> str:
    """'just now', '40 seconds ago', '7 minutes ago': one figure, which the model can repeat without arithmetic."""
    if seconds < 5:
        return "just now"
    if seconds < 90:
        return f"{round(seconds)} seconds ago"
    return f"{round(seconds / 60)} minutes ago"
