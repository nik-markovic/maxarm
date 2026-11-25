"""`AgentService`: a request answered by the language model, the tiles its tools -- the /IOTCONNECT app's `agent`
command and the calibration app's Chat page, the same way.

    agent = AgentService(AraConnector(), desk, on_say=lambda text, is_ok: print(text))
    agent.ask("a 7-letter word")            # the answer, after seconds; CommandError if it cannot be asked
    agent.submit("go", source="page")       # the same on a thread of its own: the page follows `state()`
    agent.stop()                            # the request being answered, cut short
    agent.forget()                          # a new visitor: the conversation starts again
    agent.trace(n)                          # what went to and from the model for answer `n` of the chat

The property-alarm demo's design (`app/agent.py` there): the model in another process (`connector.py`), strands-
agents asking it, a new strands `Agent` for every request, nothing streamed, every tool call printed -- the model's
last sentence says what it meant to do, not what it did. On top of it:

**The model reads the request, code finds the word.** `find_word` takes what the request wants -- a clue, the user's
own word, or neither for the longest -- and the word comes from the dictionary, the tiles' letters counted in code;
for a clue the model chooses among the words nearest it in meaning (`finder.py`). Made by the model, the words were
mostly none, or not of the tiles (STATUS-kiwi §5, §6).

**A word is set down while the model talks.** `find_word` starts the arm and returns: the model's last words -- the
word and what it means -- come while the arm works, minutes. When it is done, the chat and `on_say` are told, which
is the cloud's `answer`; the model is not asked again. Meanwhile another request may be put to it: nothing more can
be set down or looked at, but words can be found in the tiles left. The first request once a word is set down looks
at the tiles again before the model is asked, unless someone has since: the visitor has likely put them back.

**Each request starts afresh**, told the letters and what happened before in a line (`chat.py`), never its history:
the window is 4096 tokens, and every model call of a request reads the whole of it again. What went to and from the
model is kept with its answer, for the page (`trace()`).

**Bounded, not queued.** One request at a time: another, while one is answered, is refused, and so is a request
before the model is ready -- with how far it has got. At most MAX_STEPS tool calls a request, the rest refused with
"answer now"; cut short after REQUEST_LIMIT_S. A tool call the model writes out as text is made all the same
(`repair.py`).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from threading import Lock, Thread, Timer
from typing import Any, Callable, List, Literal, Optional, Tuple

from ..commands import CommandError
from . import prompt
from .chat import NOT_PLACED, PLACED, PLACING, PROPOSED, Chat, Turn
from .connector import READY, ModelSource, describe_unready
from .desk import Desk
from .dictionary import Dictionary
from .finder import GIVEN, WordFinder
from .repair import written_call
from .watch import Watch
from .words import (RELATION, is_another, is_placing_refused, is_word_given, meaningful_clue, spaced, take_letters,
                    wanted_length)

logger = logging.getLogger(__name__)

try:
    from strands import Agent, tool
    IS_STRANDS_AVAILABLE = True
except ImportError:                         # the apps run without it: no agent
    IS_STRANDS_AVAILABLE = False

MAX_STEPS = 4                               # a look, two finds, and one to spare
REQUEST_LIMIT_S = 180.0
STEP_DOING = {"find_word": "Finding a word in the tiles",
              "detect_tiles": "Looking at the tiles: the arm moves aside, the camera reads them"}


@dataclass
class _Request:
    text: str
    source: str
    turn: Optional[Turn]
    started: float = field(default_factory=time.monotonic)
    agent: Any = None                       # strands' Agent, while it runs: to cut it short
    doing: str = "Reading the request"
    calls: int = 0                          # model calls so far
    steps: int = 0                          # tool calls so far
    fitting: List[str] = field(default_factory=list)   # the words found that the tiles spell, in order
    placed: str = ""                        # the word it started setting down
    trace: List[dict] = field(default_factory=list)    # what went to and from the model, in turn
    is_asked_again: bool = False            # an answer dropped and the model asked again (`watch.py`)
    is_stopped: bool = False                # `stop()`: no tool call more, nothing set down


class AgentService:
    def __init__(self, source: ModelSource, desk: Desk, on_say: Optional[Callable[[str, bool], None]] = None) -> None:
        self.source = source
        self.desk = desk
        # What is said once the arm is done, and whether it set the word down: the cloud's `answer`.
        self.on_say = on_say or (lambda text, is_ok: None)
        self.chat = Chat()
        self.finder = WordFinder(Dictionary(), source)
        self._lock = Lock()                             # one request at a time
        self._request: Optional[_Request] = None        # the one being answered
        self._placing = ""                              # the word the arm is setting down
        self._placed_stamp: Optional[int] = None        # the detection after the last word set down
        self.source.tools = [_openai_spec(each) for each in self._build_tools(_Request("", "probe", None))]
        self.source.start()

    # --- asking -----------------------------------------------------------------------------------------------

    def ask(self, text: str, source: str = "console") -> str:
        """The model's answer to `text`, after whatever tools it calls; CommandError if it cannot be asked now."""
        self._take(text)
        try:
            return self._answer(text, source)
        finally:
            self._lock.release()

    def submit(self, text: str, source: str = "page") -> None:
        """`ask()` on a thread of its own: the answer goes into the chat. CommandError, at once, as `ask()`."""
        self._take(text)

        def answer() -> None:
            try:
                self._answer(text, source)
            except CommandError:
                pass                                    # in the chat already
            except Exception as error:                  # a bug here must not pass unsaid
                logger.exception("the agent's request failed")
                self.chat.add("assistant", f"Something went wrong: {type(error).__name__}: {error}", is_failed=True)
            finally:
                self._lock.release()
        Thread(target=answer, name="agent", daemon=True).start()

    def stop(self) -> bool:
        """Cut the request being answered short, at the model's next safe point; whether there was one."""
        request = self._request
        if request is None:
            return False
        request.is_stopped, request.doing = True, "Stopping"
        if request.agent is not None:
            request.agent.cancel()
        return True

    def forget(self) -> None:
        self.chat.forget()

    def state(self) -> dict:
        """For the Chat page: the model, the conversation, the request being answered and the word being set down."""
        return {**self.activity(), "messages": self.chat.messages()}

    def activity(self) -> dict:
        """The model, the request being answered and the word being set down, without the conversation: for the
        stream's bar, every frame."""
        request = self._request
        # Seconds, not the time it started: the page's clock need not be the board's.
        thinking = None if request is None else {"text": request.text, "doing": request.doing, "calls": request.calls,
                                                 "steps": request.steps,
                                                 "seconds": int(time.monotonic() - request.started)}
        return {"model": self.source.status(), "thinking": thinking, "placing": self._placing}

    def trace(self, n: int) -> Optional[List[dict]]:
        """What went to and from the model for the chat's message `n`, an answer; None if there is none."""
        return self.chat.trace(n)

    def _take(self, text: str) -> None:
        if not text.strip():
            raise CommandError("Ask what? Like: a 7-letter word.")
        status = self.source.status()
        if status["state"] != READY:
            raise CommandError(describe_unready(status))
        if not self._lock.acquire(blocking=False):
            raise CommandError("Still answering the last request: ask again when it is done.")

    def _answer(self, text: str, source: str) -> str:
        turn = self.chat.begin_turn(text)
        self.chat.add("user", text, source=source)
        request = _Request(text, source, turn)
        print(f"[agent] {source}: {text!r}", flush=True)
        self._request = request
        self._look_again_if_due(request)
        message = prompt.compose(text, self._describe_desk(), self.chat.recap())
        request.trace += [{"kind": "system", "text": prompt.SYSTEM_PROMPT}, {"kind": "request", "text": message}]
        try:
            with self.source.lock:
                request.agent = Agent(model=self.source.model(), tools=self._build_tools(request),
                                      system_prompt=prompt.SYSTEM_PROMPT, callback_handler=None,
                                      hooks=[Watch(request)])
                limit = Timer(REQUEST_LIMIT_S, request.agent.cancel)
                limit.daemon = True                     # not to hold the app up as it stops
                limit.start()
                try:
                    result = request.agent(message)
                finally:
                    limit.cancel()
        except Exception as error:
            logger.debug("the agent failed", exc_info=True)
            print(f"[agent] {type(error).__name__}: {error}", flush=True)
            self.source.failed(error)
            said = f"The language model did not answer ({type(error).__name__}): {error}"
            self.chat.add("assistant", said, trace=request.trace, is_failed=True,
                          seconds=round(time.monotonic() - request.started, 1))
            self._request = None
            raise CommandError(said) from error
        answer = self._finish(request, result)
        self._request = None                            # once the answer is in the chat: the page shows one, not both
        print(f"[agent] {answer!r}", flush=True)
        return answer

    def _finish(self, request: _Request, result: Any) -> str:
        """The answer as the user is given it, its figures in the chat, and the word it chose remembered."""
        answer = str(result).strip()
        if written_call(answer, self.source.tools):
            answer = ""                             # a tool call written out where none could be made: nothing to say
        is_cut = result.stop_reason == "cancelled"
        if not answer:
            answer = (f"{', '.join(request.fitting[:3])} would fit." if request.fitting
                      else "I could not find a word in time." if is_cut else "I have no answer to that.")
        answer = _sentence_case(answer, request.fitting)
        usage = result.metrics.accumulated_usage
        figures = {"seconds": round(time.monotonic() - request.started, 1), "calls": request.calls,
                   "tokens_in": usage.get("inputTokens", 0), "tokens_out": usage.get("outputTokens", 0),
                   "is_cut": is_cut}
        self.chat.add("assistant", answer, trace=request.trace, **figures)
        print(f"[agent] {figures['seconds']} s, {request.calls} model calls, {figures['tokens_in']} tokens read, "
              f"{figures['tokens_out']} written{', cut short' if is_cut else ''}", flush=True)
        return answer

    # --- the tools ----------------------------------------------------------------------------------------------

    def _build_tools(self, request: _Request) -> list:
        """The tools, one set per request. Their docstrings are what the model reads, in every call: short."""

        # `place` is asked for, unused: without it, and the prompt's sentence on it, the model wrote most of its calls
        # out as text; whether to set the word down is the request's, read in code (STATUS-kiwi §8).
        @tool
        def find_word(want: Literal["longest", "clue", "word"], clue: str = "", word: str = "",
                      place: bool = True) -> str:
            """Find a real word the tiles spell, and have the arm set it down.

            Args:
                want: "longest" for the longest word or a number of letters, "clue" or "word".
                clue: for "clue", what the word should be or mean, in the user's words, like "a small swamp animal".
                word: for "word", the user's own word, which they ask to spell or place.
                place: false if the user said not to set it down.
            """
            said = [want, f'clue="{clue}"' if clue else "", f'word="{word}"' if word else ""]
            argument = ", ".join(part for part in said if part)
            return self._step(request, "find_word", argument, lambda: self._find(request, want, clue, word))

        @tool
        def detect_tiles() -> str:
            """Look at the tiles again: the arm moves aside and the camera reads every letter."""
            return self._step(request, "detect_tiles", "", lambda: self._detect())

        return [find_word, detect_tiles]

    def _step(self, request: _Request, name: str, argument: str, call: Callable[[], Tuple[str, dict]]) -> str:
        """One tool call: in the chat and on the console before it runs -- it takes time -- and with what came back."""
        request.steps += 1
        if request.is_stopped:
            return "Stopped by the user: nothing more is done. Answer now."
        if request.steps > MAX_STEPS:
            if request.steps > MAX_STEPS + 2:
                request.agent.cancel()
            return "No more tool calls: answer the user now, with what you have."
        print(f"    [agent] {name}({argument})", flush=True)
        step = self.chat.add("step", "", tool=name, argument=argument, is_done=False)
        request.doing = STEP_DOING[name]
        started = time.monotonic()
        try:
            result, data = call()
        except Exception as error:              # refused or failed: an answer for the model too, to pass on
            why = (str(error) or repr(error)).rstrip(".")
            result, data = f"{name} could not be done: {why}. Tell the user so.", {"is_failed": True}
        self.chat.update(step, result, is_done=True, seconds=round(time.monotonic() - started, 1), **data)
        request.doing = f"Thinking · pass {request.calls + 1}"
        print(f"    [agent] {name} -> {result!r}", flush=True)
        return result

    def _find(self, request: _Request, want: str, clue: str, word: str) -> Tuple[str, dict]:
        """The word the request wants (`finder.py`), and set down unless the request says not to or the arm is busy.
        What the model fills in decides, `want` only between a word and a clue both given; the length, and whether
        to set the word down, are read from the request in code (STATUS-kiwi §6, §9)."""
        tiles, why_none = self._letters_now()
        if tiles is None:
            return why_none, {}
        if want == "word" and not word:
            word, clue = clue, ""                       # the user's word, given as a clue
        if word and not is_word_given(request.text):
            word, clue = "", clue or word               # "a fruit": a clue, not the word FRUIT
        clue = meaningful_clue(clue)
        relation = RELATION.search(request.text)
        if clue and relation and not RELATION.search(clue):
            clue = f"{relation.group(0).lower()} {clue}"   # "noun for avenge", passed on as "avenge" (STATUS-kiwi §13)
        given = word if word and (want != "clue" or not clue) else ""
        place = not is_placing_refused(request.text)
        length = wanted_length(request.text)
        if not length and is_another(request.text):     # "another one" after "a 5-letter word": of 5 letters
            length = wanted_length(self.chat.previous_request())
        found = self.finder.find(tiles, "" if given else clue, given, length, self.chat.words_used(),
                                 check=None if self._placing else self.desk.check)
        if found.ideas:
            request.trace.append({"kind": "ideas", "text": found.ideas, "seconds": found.ideas_seconds})
        if found.asked:
            request.trace.append({"kind": "pick", "text": found.asked, "seconds": found.seconds})
        spelled = [found.word] + (found.shown if found.kind != GIVEN else [])      # a given word may not fit
        request.fitting += [each for each in dict.fromkeys(spelled) if each and each not in request.fitting]
        data = {"kind": found.kind, "clue": found.clue or clue, "words": found.shown, "word": found.word,
                "said": found.said, "is_agreed": found.is_agreed, "missing": found.missing, "problem": found.problem,
                "place": place}
        if not found.word and "in the way" in found.problem:     # PLACE's plan: tiles where the word would go
            # Said so: told "not clear", the model called STEAMED too long, an N left in the dropoff area (STATUS §9).
            data["dropoff"] = (self.desk.seen() or {}).get("dropoff", "")
            held = f": {spaced(data['dropoff'])}" if data["dropoff"] else ""
            return (f"{found.problem.split(',')[0]} fits, but tiles lie where words are set down, in the dropoff "
                    f"area{held}. The word's length is not the reason. Ask the user to move those tiles out of the "
                    "dropoff area, then ask again."), data
        if not found.word:
            return found.tell, data
        if not place:
            self.chat.set_word(request.turn, found.word, PROPOSED)
            return f"{found.about}; not set down, as asked. Tell the user the word and what it means.", data
        busy = self._placing or request.placed
        if busy:
            self.chat.set_word(request.turn, found.word, PROPOSED)
            data["busy"] = busy
            return f"{found.about}; not set down: the arm is setting down {busy}. Tell the user.", data
        if request.is_stopped:                          # stopped while the word was being found
            self.chat.set_word(request.turn, found.word, PROPOSED)
            return f"{found.about}; not set down: stopped by the user. Tell the user the word.", data
        request.placed = found.word
        self.chat.set_word(request.turn, found.word, PLACING)
        self._start_placing(found.word, request.turn)
        data["is_placed"] = True
        return f"{found.about}. The arm has started setting it down. Tell the user the word and what it means.", data

    def _detect(self) -> Tuple[str, dict]:
        if self._placing:
            return f"The arm is setting down {self._placing}: the tiles can be looked at once it is done.", {}
        self.desk.detect()
        return f"Looked again. {self._describe_desk()}. If the user wants a word, call find_word now.", {}

    def _look_again_if_due(self, request: _Request) -> None:
        """The tiles looked at again, before the model is asked, when nothing has looked since a word was set down:
        the visitor has likely put them back (PILOT-kiwi, LLM Issues)."""
        seen = self.desk.seen()
        if self._placing or not seen or self._placed_stamp is None or seen.get("stamp") != self._placed_stamp:
            return
        self._placed_stamp = None
        step = self.chat.add("step", "", tool="detect_tiles", argument="a word was set down: looking again first",
                             is_done=False)
        request.doing = STEP_DOING["detect_tiles"]
        started = time.monotonic()
        try:
            self.desk.detect()
            result, data = f"Looked again. {self._describe_desk()}", {}
        except Exception as error:                  # said, and the request goes on with the tiles as last seen
            result, data = f"Could not look again: {str(error) or repr(error)}", {"is_failed": True}
        self.chat.update(step, result, is_done=True, seconds=round(time.monotonic() - started, 1), **data)
        request.trace.append({"kind": "look", "text": result})
        print(f"    [agent] looked again first -> {result!r}", flush=True)

    def _start_placing(self, word: str, turn: Optional[Turn]) -> None:
        self._placing = word

        def place() -> None:
            try:
                self.desk.place(word)
                said, status = f"{word} is set down.", PLACED
            except Exception as error:
                said, status = f"{word} was not set down: {str(error) or repr(error)}", NOT_PLACED
            finally:
                seen = self.desk.seen()
                self._placed_stamp = seen.get("stamp") if seen else None
                self._placing = ""
            self.chat.set_word(turn, word, status)
            self.chat.add("event", said, word=word, is_ok=status == PLACED)
            print(f"[agent] {said}", flush=True)
            try:
                self.on_say(said, status == PLACED)
            except Exception:
                logger.exception("on_say raised")
        Thread(target=place, name="agent-place", daemon=True).start()

    # --- what the model is told -------------------------------------------------------------------------------

    def _letters_now(self) -> Tuple[Optional[str], str]:
        """The letters a word can take now -- those left once the word being set down has taken its own -- or None
        and why there are none."""
        seen = self.desk.seen()
        if seen is None:
            return None, "No tiles looked at yet: call detect_tiles first."
        letters = take_letters(seen["on_offer"], self._placing) if self._placing else seen["on_offer"]
        if not letters:
            return None, "No tiles left to use: every tile is set down, out of reach or not read."
        return letters, ""

    def _describe_desk(self) -> str:
        """The tiles, as the model is told them at the start of each request and after it looks again."""
        seen = self.desk.seen()
        if seen is None:
            return "No tiles looked at yet: detect_tiles finds them."
        letters, why_none = self._letters_now()
        said = f"Tiles you can use ({len(letters)}): {spaced(letters)}" if letters else why_none
        if self._placing:
            said = (f"The arm is setting down {self._placing} now: nothing else can be set down or looked at until "
                    f"it is done. Left after it: {said[0].lower()}{said[1:]}")
        if seen["dropoff"] and not self.chat.words_used():
            said += f"\nIn the dropoff area, left to right from behind the arm: {seen['dropoff']}"
        return said


def _sentence_case(answer: str, keep: List[str]) -> str:
    """An answer the model wrote all in capitals -- told to write the word in capitals, it sometimes writes all of it
    so -- in lower case but the sentences' first letters, 'I', and the words in `keep`."""
    letters = [character for character in answer if character.isalpha()]
    if len(letters) < 12 or sum(character.isupper() for character in letters) < 0.8 * len(letters):
        return answer
    lowered = re.sub(r"[A-Za-z]+", lambda found: found.group(0) if found.group(0) in keep
                     else "I" if found.group(0) == "I" else found.group(0).lower(), answer)
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda found: found.group(1) + found.group(2).upper(), lowered)


def _openai_spec(decorated) -> dict:
    """A strands tool as the connector is sent it, for its probe: the same guide built for the same tools."""
    spec = decorated.tool_spec
    return {"type": "function", "function": {"name": spec["name"], "description": spec["description"],
                                             "parameters": spec["inputSchema"]["json"]}}
