"""What the model does in a request, watched through strands' hooks: its calls counted, for the page's bubble while it
thinks, and what went to and from it kept, for each answer's trace on the page (`service.py`).

    Agent(..., hooks=[Watch(request)])          # request: .text, .calls, .steps, .doing, .trace, .is_asked_again

A model call is kept as what the model wrote -- its text, each tool call as name(arguments), and whether a call was
written out as text and made all the same (`repair.py`) -- with its seconds; a tool call as what it gave back.

An answer that says a word is set down when no tool has been called -- the model copying the recap's line, "FORSET
(F-O-R-S-E-T), set down" -- is dropped and the model asked again, once. Not when the request asks
to explain: the last word was set down, and saying so is right.
"""

from __future__ import annotations

import json
import re
import time

CLAIM = re.compile(r"\b(?:set(?:ting)?\s+(?:\w+\s+)?down|is set|plac(?:e|ed|ing)|the arm (?:sets|is setting))\b",
                   re.IGNORECASE)
EXPLAIN = re.compile(r"\b(?:explain|elaborate|mean|meaning|define|definition|tell me more|what is|what's)\b",
                     re.IGNORECASE)

try:
    from strands.hooks import AfterModelCallEvent, AfterToolCallEvent, BeforeModelCallEvent, HookProvider
except ImportError:                     # no agent without strands: nothing here is used then
    HookProvider = object


class Watch(HookProvider):
    def __init__(self, request) -> None:
        self.request = request
        self.started = 0.0
        self.repairs = 0                # the model's calls made of text, before the call watched

    def register_hooks(self, registry, **kwargs) -> None:
        registry.add_callback(BeforeModelCallEvent, self._before_model)
        registry.add_callback(AfterModelCallEvent, self._after_model)
        registry.add_callback(AfterToolCallEvent, self._after_tool)

    def _before_model(self, event) -> None:
        self.request.calls += 1
        self.request.doing = f"Thinking · pass {self.request.calls}"
        self.started = time.monotonic()
        self.repairs = getattr(event.agent.model, "repairs", 0)

    def _after_model(self, event) -> None:
        if event.stop_response is not None:
            said = described(event.stop_response.message)
        else:
            said = f"no answer: {type(event.exception).__name__}: {event.exception}"
        if getattr(event.agent.model, "repairs", 0) > self.repairs:
            said = f"(the call written out as text, made all the same)\n{said}"
        if (event.stop_response is not None and not self.request.steps and not self.request.is_asked_again
                and "toolUse" not in str(event.stop_response.message) and CLAIM.search(said)
                and not EXPLAIN.search(self.request.text)):
            event.retry = True
            self.request.is_asked_again = True
            said = f"(said a word is set down, with no tool called: asked again)\n{said}"
        self.request.trace.append({"kind": "model", "text": said, "seconds": round(time.monotonic() - self.started, 1)})

    def _after_tool(self, event) -> None:
        content = (event.result or {}).get("content", [])
        self.request.trace.append({"kind": "tool", "text": " ".join(part.get("text", "") for part in content)})


def described(message: dict) -> str:
    """A model's message as the trace shows it: its text, and each tool call as name(arguments)."""
    parts = []
    for block in message.get("content", []):
        if block.get("text", "").strip():
            parts.append(block["text"].strip())
        elif "toolUse" in block:
            parts.append(f"-> {block['toolUse']['name']}({json.dumps(block['toolUse'].get('input', {}))})")
    return "\n".join(parts)
