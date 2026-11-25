"""What the language model is told: the system prompt, and the message each request becomes.

    prompt.compose("a 7-letter word", desk="Tiles you can use (9): A E E H M O R S T", recap="")

The system prompt is the same for every request; the message carries what changes -- the letters, a line of what
happened before (`chat.py`), the request. Both are read again on every model call of a request, and what the model
writes costs far more than what it reads. Hence what it is asked to write: what the request wants --
the longest, a clue or the user's own word -- never the words themselves, which code finds (`finder.py`).

The wording is tried, not chosen: whether the model makes a tool call or writes it out as text turns on it, down to
the last sentence, and an example it is shown it copies, labels and all. Without a choice to
make, `want`, it answered "go" by explaining GO; and the explanation says the word "letter by letter", not "spell",
which "spell AvnetDemo" was taken for.

The model is Qwen2.5-7B-Instruct: it cannot see the letters in a word -- its tokens are pieces of words -- nor
count them, so the letters are written a space apart, and every count is code's (`words.py`).
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are the voice of a robot arm at an NXP i.MX95 demo. The arm sets words down with letter tiles.
Every request asks for a word made of the tiles, unless it asks to explain or elaborate. For a word, first call \
find_word, and the code finds a real word the tiles spell. Tell it what the user wants:
- "longest" for "go", "run", "start" or "execute", for the longest word, or for a number of letters;
- "clue" when the user says what the word should be or mean, with the clue, like "a small swamp animal";
- "word" when the user asks to spell, write or place a word they name, with that word.
Set place to false if the user says not to set the word down. For "another one", want what the last request wanted.
Example: the user says "go": find_word(want="longest"). The user says "spell NXP": find_word(want="word", \
word="NXP").
Then tell the user the word and, in one sentence, what it means.
To explain or elaborate, call no tool: say the last word letter by letter, what it means, and use it in a sentence, \
as at a spelling bee.
Call detect_tiles only when the user asks to look at the tiles again, or no tiles are known.
Answer in one or two short sentences. Write the word itself in capitals."""


def compose(request: str, desk: str, recap: str) -> str:
    """The message the model reads for a request: the desk, what happened before, then the request itself."""
    lines = [desk] + ([recap] if recap else []) + [f'The user says: "{" ".join(request.split())}"']
    return "\n".join(lines)
