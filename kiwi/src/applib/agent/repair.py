"""A tool call the model wrote out as text, made the call it meant: strands' OpenAI model with one repair.

    model = RepairingModel(tool_specs, client_args=..., model_id=..., params=...)    # as strands' OpenAIModel

Qwen2.5-7B on the Ara240 makes a tool call by starting its answer with a `<tool_call>` tag, which the connector
turns into a call. Often it writes the call out instead, as text, or the tag comes out as another token -- which,
turns on the prompt's wording and on chance. The shapes it wrote, when find_word took guesses,
every one of them read here:

    find_word(guesses="ORES THEM HOME", place=true)          find_word guesses="ORES THEM HOME".
    find_word({"guesses": "MOOSE ROAST", "place": true})      Find_word with guesses: Real, Mark, Horse.
    HE\\n\\n{"name": "find_word", "arguments": {...}}         <tool_call>\\n{"name": "find_word", ... (cut short)

An answer is taken for a call only when it opens with one: a tool's name -- or `"name": "<tool>"` -- within its
first OPENING_CHARS, followed by what starts arguments. Its string arguments are the quoted text after their
names -- or, none named, the first quoted text not a name is the first of them; booleans `name=true|false`, else a
bare true or false.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple
from uuid import uuid4

try:
    from strands.models.openai import OpenAIModel
except ImportError:                     # `written_call()` alone is wanted without strands; the model is not built
    OpenAIModel = object

OPENING_CHARS = 40


class RepairingModel(OpenAIModel):
    def __init__(self, tool_specs: List[dict], **config: Any) -> None:
        super().__init__(**config)
        self.tool_specs = tool_specs        # as OpenAI lists them: {"type": "function", "function": {...}}
        self.repairs = 0                    # calls made of text so far

    def _format_non_streaming_response(self, response: Any) -> list:
        message = response.choices[0].message if getattr(response, "choices", None) else None
        if message is not None and not getattr(message, "tool_calls", None):
            call = written_call(getattr(message, "content", None) or "", self.tool_specs)
            if call is not None:
                self.repairs += 1
                print(f"    [agent] a tool call written out as text, made: {message.content!r}", flush=True)
                response = _as_call(response, *call)
        return super()._format_non_streaming_response(response)


def written_call(text: str, tool_specs: List[dict]) -> Optional[Tuple[str, dict]]:
    """The call `text` writes out, as (name, arguments); None if it is an answer."""
    names = "|".join(re.escape(spec["function"]["name"]) for spec in tool_specs)
    # What may follow a tool's name to start its arguments: a bracket, "with", a colon, an argument's name, the end.
    starts = [r"\(", r"\{", r"with\b", ":", "$"] + [rf"{re.escape(parameter)}\b" for spec in tool_specs
                                                    for parameter in spec["function"].get("parameters", {})
                                                    .get("properties", {})]
    opening = re.search(rf'"name"\s*:\s*"({names})"|\b({names})\b\s*(?={"|".join(starts)})', text, re.IGNORECASE)
    if not names or opening is None or opening.start() > OPENING_CHARS:
        return None
    name = (opening.group(1) or opening.group(2)).lower()
    spec = next(spec["function"] for spec in tool_specs if spec["function"]["name"] == name)
    rest = text[opening.end():]
    properties = spec.get("parameters", {}).get("properties", {})
    strings = [parameter for parameter, schema in properties.items() if schema.get("type") == "string"]
    arguments = {}
    for parameter, schema in properties.items():
        value = (_string(rest, parameter) if schema.get("type") == "string"
                 else _boolean(rest, parameter) if schema.get("type") == "boolean" else None)
        if value is not None:
            arguments[parameter] = value
    if strings and not any(parameter in arguments for parameter in strings):
        value = _first_quoted(rest, set(properties) | {"name", "arguments", name})
        if value is not None:
            arguments[strings[0]] = value               # `find_word("a farm animal")`: the first string argument
    required = spec.get("parameters", {}).get("required", [])
    for parameter in required:              # a choice left out: the first, the tool weighs what else is given
        if parameter not in arguments and properties.get(parameter, {}).get("enum"):
            arguments[parameter] = properties[parameter]["enum"][0]
    missing = [parameter for parameter in required if parameter not in arguments]
    return None if missing else (name, arguments)


def _string(text: str, parameter: str) -> Optional[str]:
    """The quoted text after `parameter` -- its closing quote perhaps cut off -- or after `parameter:` to the end of
    the sentence."""
    # `name=` or `"name":`, not the name alone: `want="clue", clue="..."` holds "clue" as a value first.
    named = re.search(rf'\b{re.escape(parameter)}"?\s*[:=]\s*"([^"]*)"?', text, re.IGNORECASE)
    if named:
        return named.group(1).strip()
    bare = re.search(rf'\b{re.escape(parameter)}\s*[:=]\s*([^."\n]+)', text, re.IGNORECASE)
    return bare.group(1).strip() if bare else None


def _first_quoted(text: str, names: set) -> Optional[str]:
    """The first quoted text that is not a name: an argument written without its own."""
    for quoted in re.finditer(r'"([^"]*)"', text):
        if quoted.group(1).lower() not in names and quoted.group(1).strip():
            return quoted.group(1).strip()
    return None


def _boolean(text: str, parameter: str) -> Optional[bool]:
    named = re.search(rf"{re.escape(parameter)}\W*(true|false)\b", text, re.IGNORECASE)
    bare = named or re.search(r"\b(true|false)\b", text, re.IGNORECASE)
    return None if bare is None else bare.group(1).lower() == "true"


def _as_call(response: Any, name: str, arguments: dict) -> Any:
    """`response` as it would have come with the call made: no text, one tool call, finished for it."""
    call = SimpleNamespace(id=f"call_{uuid4().hex[:24]}", type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))
    message = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
                           usage=getattr(response, "usage", None))
