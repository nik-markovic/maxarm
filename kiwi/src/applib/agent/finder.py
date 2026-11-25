"""The word a request asks for, found in code: the user's own, the longest the tiles spell, one of a length, or one
that fits a clue. The words are the dictionary's (`dictionary.py`); where more than one would do, the choice among a
few is the model's -- it knows what a toad is, and cannot count letters (STATUS-kiwi §7, §9, §13).

    finder = WordFinder(Dictionary(), source)
    found = finder.find("AEEHMORST", clue="a large farm animal", given="", length=0, used=[], check=desk.check)
    found.word          # 'HORSE'; "" when there is none, and `found.tell` says why

- **Given** -- "spell AvnetDemo": the user's word as it is, a name or not, if the tiles hold its letters; never another
  in its place.
- **The longest** -- "go": the longest word the tiles spell, of a length the commonest; one the arm cannot set down
  gives way to the next.
- **A length** -- "a 5-letter word": the commonest words of the length but the little ones -- THERE, ONE, WAS, Zipf
  LITTLE or more -- SHORTLIST of them put to the model, which chooses the most vivid.
- **A clue** -- "a small swamp animal": SHORTLIST words the tiles spell put to the model, which names the one the clue
  fits best -- the longer of two that fit as well -- or NONE. The model is asked the clue as passed on, misspellings
  and all: it reads "to witter, dry out" as wither. Code's own lookups take it respelled ("labirinth" is pieces to
  the search by meaning), a word as near two others left out: "witter" made winter, the list SNOW, DEW, WET. The
  list, one form of each word, none of the clue's own (SPOT for "a spotted big cat"):
  - the model's own ideas -- asked first, without the tiles, whose words it cannot see -- of IDEA_LETTERS or more
    that the tiles spell, IDEAS_KEPT at most: VENDETTA for "noun for avenge", which neither of the others finds;
  - the words WordNet ties to the clue's -- MAZE to labyrinth, NEW to old, DATE to fruit -- RELATED at most; its
    opposites only for a clue that asks for one, and off the list otherwise: WET for "dry out", near it in meaning;
  - the model's shorter ideas, IDEAS_KEPT at most: below WordNet's, as on top LOVE and NOTE took VALENTINE's place;
  - the nearest the clue in meaning, "synonym for" and the like left out of it.
  An idea of the model's, of any length, that WordNet ties to the clue too is the word, the model not asked to
  choose among the 12: STATE for "synonym for country", which it passed over for TOWN when asked. Of more such
  ideas the model chooses, among them alone: the first was TALE, not ANECDOTE; the longest TARN, not SEA. The
  search by meaning takes EVERYDAY words only: rarer ones come by WordNet and the model's ideas, not by their
  pieces (DONA for a gift). A word the model names that is
  not on the list -- LABYRINTH, for a list without MAZE -- is none: the list's first, set down in its place, was
  TEST. How each part was chosen, and what it does: STATUS-kiwi §13.

A word set down before in this visit (`used`) is chosen again only when no other is there.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from .dictionary import Dictionary
from .words import content_words, describe_missing, joined_words, missing_letters, without_relation

GIVEN, LONGEST, LENGTH, CLUE = "given", "longest", "length", "clue"
SHORTLIST = 12              # the words put to the model, as tried (STATUS-kiwi §7)
RELATED = 8                 # WordNet's words among them at most, the nearest the clue first (STATUS-kiwi §13)
IDEAS_KEPT = 4              # the model's own words among them at most
IDEA_LETTERS = 6            # the model's own words are asked for, and kept, of this many letters or more: the long
                            # words the search by meaning misses, and the model cannot count, so code keeps them
SHOWN = 8                   # the longest words the page shows beside the one chosen
TRIED = 12                  # the longest words planned at most, for one the arm can set down
LITTLE = 6.0                # Zipf: the commonest words of English, little ones mostly (STATUS-kiwi §9)
EVERYDAY = 3.0              # Zipf: the longest and a length take words this common or more, a clue rarer ones too
NONE = "NONE"
OPPOSITE = re.compile(r"\b(?:opposite|antonym|reverse|contrary)", re.IGNORECASE)    # a clue that asks for one
CLUE_SYSTEM = "You match words to descriptions. Answer with one word."
CLUE_QUESTION = ('Words: {words}.\nDescription: "{clue}"\nWhich ONE word of the list does the description fit best? '
                 "Judge by meaning. If two fit equally well, choose the longer one. Answer with that word only, or "
                 "NONE if none fits.")
IDEAS_SYSTEM = "You list English words. Answer with words only, separated by commas."
IDEAS_QUESTION = 'List 12 different English words of {letters} letters or more that fit: "{clue}".'
LENGTH_SYSTEM = "You pick words. Answer with one word of the list."
LENGTH_QUESTION = ("List: {words}.\nPick the ONE word of the list that names a thing or an action, the most vivid one. "
                   "Answer with that word only.")


@dataclass
class Found:
    kind: str                   # GIVEN, LONGEST, LENGTH or CLUE
    word: str = ""              # the word to set down; "" for none
    about: str = ""             # the word, for the model: 'TOAD (T-O-A-D, 4 letters) fits "a small swamp animal"'
    tell: str = ""              # for the model, why there is none
    shown: List[str] = field(default_factory=list)    # the words weighed, for the page
    said: str = ""              # what the model named: one of `shown`, NONE, or another
    missing: str = ""           # what the tiles lack for a given word
    problem: str = ""           # why the arm cannot set the word down
    asked: str = ""             # what the model was asked, and its answer
    seconds: float = 0.0        # the model's time on it
    ideas: str = ""             # for a clue, what the model was asked for its own words, and its answer
    ideas_seconds: float = 0.0
    clue: str = ""              # for a clue, as its words were found: respelled
    is_agreed: bool = False     # for a clue, the word the model's own idea and WordNet's, not its choice


class WordFinder:
    def __init__(self, dictionary: Dictionary, source) -> None:
        self.dictionary = dictionary
        self.source = source            # a `connector.ModelSource`: `ask_plain()`

    def find(self, tiles: str, clue: str, given: str, length: int, used: Sequence[str],
             check: Optional[Callable[[str], str]]) -> Found:
        """The word for a request. `tiles`: the letters to use; `clue`, `given`: the model's reading of the request;
        `length`: the letters asked for, 0 for any; `check(word)`: why the arm cannot set it down, "" if it can --
        None when nothing can be planned, the arm busy."""
        check = check or (lambda word: "")
        given = "".join(re.findall(r"[A-Za-z]+", given)).upper()
        if given:
            return self._given(given, tiles, check)
        kind = CLUE if clue.strip() else LENGTH if length else LONGEST
        try:
            words = self.dictionary.spelled(tiles, length)
        except LookupError as reason:
            return Found(kind, tell=f"The dictionary is not there: {reason}. Tell the user.")
        if kind != CLUE:                # the rarer words are for a clue, which says what the word means: WANE
            words = [word for word in words if self.dictionary.zipf[word] >= EVERYDAY]
        if not words:
            some = f"of {length} letters " if length else ""
            return Found(kind, tell=f"The tiles spell no word {some}in the dictionary. Tell the user.")
        fresh = [word for word in words if word not in used] or words
        if kind == CLUE:
            return self._clue(" ".join(clue.split()), fresh, length, check)
        if kind == LENGTH:
            return self._of_length(fresh, length, check)
        return self._longest(fresh, check)

    def _given(self, word: str, tiles: str, check: Callable[[str], str]) -> Found:
        missing = missing_letters(word, tiles)
        if missing:
            return Found(GIVEN, shown=[word], missing=missing,
                         tell=f"{word} cannot be set down: the tiles have {describe_missing(missing, tiles)}. Tell "
                              "the user so, and choose no other word.")
        problem = check(word)
        if problem:
            return Found(GIVEN, shown=[word], problem=f"{word}, {problem}",
                         tell=f"The arm cannot set {word} down: {problem}. Tell the user.")
        return Found(GIVEN, word=word, shown=[word], about=f"{_spelled(word)}, as the user asked")

    def _longest(self, words: List[str], check: Callable[[str], str]) -> Found:
        problems = []
        for word in words[:TRIED]:
            problem = check(word)
            if not problem:
                return Found(LONGEST, word=word, shown=words[:SHOWN],
                             about=f"{_spelled(word)} is the longest word the tiles spell")
            problems.append(f"{word}, {problem}")
        return Found(LONGEST, shown=words[:SHOWN], problem=problems[0],
                     tell=f"The arm cannot set down the longest words: {problems[0]}. Tell the user.")

    def _of_length(self, words: List[str], length: int, check: Callable[[str], str]) -> Found:
        everyday = _one_form([word for word in words if self.dictionary.zipf[word] < LITTLE] or words,
                             self.dictionary.base)
        found = self._ask(LENGTH, everyday[:SHORTLIST], LENGTH_SYSTEM, LENGTH_QUESTION, clue="")
        return self._settle(found, check, f"is a {length}-letter word the tiles spell, the model's choice")

    def _clue(self, clue: str, words: List[str], length: int, check: Callable[[str], str]) -> Found:
        typed, clue = clue, self.dictionary.respelled(clue)    # the model reads past "witter": not winter, wither
        parts = content_words(clue)
        own = set().union(*(self.dictionary.base_forms(part) for part in parts))
        words = [word for word in words if not self.dictionary.base_forms(word) & own]     # not SPOT for "spotted"
        related = self.dictionary.related(parts + joined_words(clue))      # body_part: its kinds, NOSE, TOE
        if not OPPOSITE.search(typed):  # WET, the opposite of dry, for "to wither, dry out": by meaning, near it
            opposites = {word for word, how in related.items() if how == "opposite"}
            words = [word for word in words if word not in opposites]
            related = {word: how for word, how in related.items() if how != "opposite"}
        if not words:                   # the tiles spell only the clue's own words, or their opposites
            return Found(CLUE, tell=f'None of the words the tiles spell is "{typed}": nothing is set down. Tell the '
                                    "user so.")
        query = without_relation(clue) or clue
        everyday = [word for word in words if self.dictionary.zipf[word] >= EVERYDAY]  # not DONA, by its pieces
        try:
            nearest = self.dictionary.nearest(query, everyday)
            kin = self.dictionary.nearest(query, [word for word in words if word in related])[:RELATED]
        except LookupError as reason:
            return Found(CLUE, tell=f"Words cannot be found by meaning: {reason}. Tell the user.")
        ideas, asked, seconds = self._ideas(typed, words, length)
        agreed = [word for word in ideas if word in related]                # STATE for "synonym for country"
        long_ideas = [word for word in ideas if len(word) >= IDEA_LETTERS][:IDEAS_KEPT]
        short_ideas = [word for word in ideas if len(word) < IDEA_LETTERS][:IDEAS_KEPT]     # after WordNet's: LOVE
        shortlist = _one_form(agreed[:1] + long_ideas + kin + short_ideas + nearest, self.dictionary.base)[:SHORTLIST]
        if len(agreed) == 1:            # the model's idea and WordNet's: not left to its choice, which gave TOWN
            found = Found(CLUE, shown=shortlist, said=agreed[0], is_agreed=True,
                          asked=f"{agreed[0]}: the model's own idea, and WordNet's ({related[agreed[0]]}): taken")
        elif agreed:                    # of those alone: TEA, not MATE; SEA, not TARN; ANECDOTE, not TALE
            found = self._ask(CLUE, agreed, CLUE_SYSTEM, CLUE_QUESTION, typed)
            found.said = found.said if found.said in agreed else max(agreed, key=len)
            found.shown, found.is_agreed = _one_form(agreed + shortlist, self.dictionary.base)[:SHORTLIST], True
        else:
            found = self._ask(CLUE, shortlist, CLUE_SYSTEM, CLUE_QUESTION, typed)
        found.ideas, found.ideas_seconds, found.clue = asked, seconds, clue
        if kin:
            found.asked = f"WordNet: {', '.join(f'{word} {related[word]}' for word in kin)}\n{found.asked}"
        if found.said not in found.shown:   # NONE, or a word of its own: LABYRINTH, for a list without MAZE
            found.tell = f'None of the words the tiles spell is "{typed}": nothing is set down. Tell the user so.'
            return found
        return self._settle(found, check, f'fits "{typed}"')

    def _ideas(self, clue: str, words: List[str], length: int) -> Tuple[List[str], str, float]:
        """The model's own words for `clue` -- asked without the tiles, which it cannot see the words of -- those of
        `words`, in its order, one form of each, whatever their length; with what it was asked and said, and its
        time. None asked for a length under IDEA_LETTERS."""
        if length and length < IDEA_LETTERS:
            return [], "", 0.0
        question = IDEAS_QUESTION.format(letters=IDEA_LETTERS, clue=clue)
        started = time.monotonic()
        answer = self.source.ask_plain(IDEAS_SYSTEM, question)
        spelled = set(words)
        ideas = _one_form([word for word in re.findall(r"[A-Za-z]+", answer.upper()) if word in spelled],
                          self.dictionary.base)
        asked = f"{IDEAS_SYSTEM}\n{question}\n-> {answer.strip()}\nThe tiles spell: {' '.join(ideas) or 'none of them'}"
        return ideas, asked, round(time.monotonic() - started, 1)

    def _ask(self, kind: str, words: List[str], system: str, question: str, clue: str) -> Found:
        """The model's choice among `words`: one of them, NONE, or another word it named."""
        question = question.format(words=", ".join(words), clue=clue)
        started = time.monotonic()
        answer = self.source.ask_plain(system, question)
        return Found(kind, shown=words, said=_named(answer, words), asked=f"{system}\n{question}\n-> {answer.strip()}",
                     seconds=round(time.monotonic() - started, 1))

    def _settle(self, found: Found, check: Callable[[str], str], why: str) -> Found:
        """`found` with the word the model chose -- for a length, the first put to it if it named another -- if the
        arm can set it down."""
        word = found.said if found.said in found.shown else found.shown[0]
        problem = check(word)
        if problem:
            found.problem = f"{word}, {problem}"
            found.tell = f"{word} fits, but the arm cannot set it down: {problem}. Tell the user."
            return found
        found.word, found.about = word, f"{_spelled(word)} {why}"
        return found


def _one_form(words: Sequence[str], base: Callable[[str], str]) -> List[str]:
    """`words` in their order, each once and one form of each (`Dictionary.base()`), the word itself where it is
    among them: LION, not LION and LIONS; NEW for "opposite of old", not NEWEST."""
    among = set(words)
    kept, bases = [], set()
    for word in words:
        form = base(word)
        if form not in bases:
            bases.add(form)
            kept.append(form.upper() if form.upper() in among else word)
    return kept


def _named(answer: str, words: Sequence[str]) -> str:
    """The word the model's answer names: the first of `words` in it, or NONE, or else its first word."""
    named = re.findall(r"[A-Za-z]+", answer.upper())
    return next((word for word in named if word in words or word == NONE), named[0] if named else "")


def _spelled(word: str) -> str:
    """'TOAD (T-O-A-D, 4 letters)': spelled for the model, which copies it rather than spell it wrong (MEANT as
    M-E-A-N-M-E-N-T, STATUS-kiwi §9)."""
    return f"{word} ({'-'.join(word)}, {len(word)} letters)"
