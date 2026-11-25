"""The agent's dictionary: which words there are, which of them the tiles spell, which of those are nearest a clue in
meaning, and which WordNet ties to the clue's words. The words the arm sets down come from here, not from the model
(STATUS-kiwi §7, §13).

    words = Dictionary()                                    # loads on a thread of its own: seconds
    words.spelled("AEEHMORST", length=7)                   # ['MOTHERS', 'THEOREM', ...]: commonest first
    words.nearest("a small swamp animal", ["VENOM", "TOAD", "DOVE"])      # ['TOAD', 'VENOM', 'DOVE']
    words.related(["labyrinth", "old"])                     # {'MAZE': 'synonym', 'NEW': 'opposite', ...}
    words.respelled("synonym for labirinth")               # 'synonym for labyrinth'
    words.base("LIONS")                                     # 'lion': LION and LIONS are one word on a list

The words are `files/words-en.tsv`: English down to the rarer words a clue may want (WANE), commonest first, made by
`training/words/make-words.py`. The tiles spell a word when they hold every letter it has, as often as it has it,
counted here.

Meaning is told by a small embedding model on the CPU, potion-base-8M: a table of vectors for the pieces of words, a
text's vector their average. Each word's vector is made once, at the start -- about a second on the board -- and a
clue's when it is asked, a millisecond; the nearest words are those whose vectors point most the same way. A word
the model holds only as pieces has their meaning, not its own -- VENDETTA is ve+nde+tta -- and so does a misspelt
one: hence `respelled()`, and WordNet beside it. `scripts/install.sh` puts the model in Hugging Face's cache at
REVISION; it is read from there, never fetched.

WordNet (Princeton's) knows what meaning cannot: that MAZE is another word for a labyrinth, NEW the opposite of
old, DATE a fruit, a NEWT an animal. `related()` gives the words it ties to a clue's: the same sense, a similar one,
the opposite, a form made from it, a kind of it. They are `files/words-wordnet.tsv`, made with the word list by
`training/words/make-words.py` -- WordNet itself, through nltk, is far too slow to load on the board (STATUS-kiwi
§13) -- and so are the words' base forms, `base()`. Without the file the dictionary works on, without them.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from threading import Event, Thread
from typing import Dict, List, Sequence, Set

import numpy as np

WORDS_FILE = Path(__file__).resolve().parents[3] / "files" / "words-en.tsv"
WORDNET_FILE = WORDS_FILE.with_name("words-wordnet.tsv")
MODEL = "minishlab/potion-base-8M"                          # as scripts/install.sh fetches it
REVISION = "bf8b056651a2c21b8d2565580b8569da283cab23"
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json")
LOAD_WAIT_S = 30.0          # a request in the app's first seconds waits for the dictionary this long at most
FIX_LETTERS = 4             # a word this long or longer that is not known is respelled...
FIX_EDITS = 2               # ...to the known word fewest letters added, dropped, changed or swapped away, at most
TIES = {"synonym": "synonym", "similar": "similar", "opposite": "opposite", "form": "related form", "kind": "kind of"}


class Dictionary:
    def __init__(self, path: Path = WORDS_FILE) -> None:
        self.path = path
        self.words: List[str] = []                  # longest first, of a length the commonest first
        self.zipf: Dict[str, float] = {}            # how common each is: 6 "the", 3 "toad"
        self.problem = "still loading"              # why `spelled()` cannot answer; "" once it can
        self.meaning_problem = "still loading"      # why `nearest()` cannot; "" once it can
        self.relation_problem = "still loading"     # why `related()` says nothing; "" once WordNet is there
        self._counts = np.zeros((0, 26), np.uint8)  # each word's count of each letter, A to Z
        self._lengths = np.zeros(0, np.int64)
        self._vectors = np.zeros((0, 0), np.float32)
        self._rows: Dict[str, int] = {}
        self._model = None
        self._base: Dict[str, str] = {}             # a word -> what it is a form of, lower case: LIONS -> lion
        self._forms: Dict[str, List[str]] = {}      # a base form -> the words of it: lion -> LION, LIONS
        self._ties: Dict[str, str] = {}             # a WordNet word -> its line of words-wordnet.tsv, past the name
        self._known: Set[str] = set()               # lower case: what is not respelled
        self._spellings: Dict[tuple, List[str]] = {}    # (first letter, length): what a word may be respelled to
        self._listed, self._loaded = Event(), Event()
        Thread(target=self._load, name="dictionary", daemon=True).start()

    def spelled(self, letters: str, length: int = 0) -> List[str]:
        """The words the tiles' `letters` spell, each tile once -- of `length` letters unless 0 -- longest first, of
        a length the commonest first; LookupError if there is no word list."""
        self._listed.wait(LOAD_WAIT_S)
        if self.problem:
            raise LookupError(self.problem)
        fits = (self._counts <= _counted(letters)).all(axis=1)
        if length:
            fits &= self._lengths == length
        return [self.words[index] for index in np.flatnonzero(fits)]

    def nearest(self, clue: str, words: Sequence[str]) -> List[str]:
        """`words`, the dictionary's own, the nearest `clue` in meaning first; LookupError if meaning cannot be told."""
        self._loaded.wait(LOAD_WAIT_S)
        if self.meaning_problem:
            raise LookupError(self.meaning_problem)
        query = self._model.encode([clue], use_multiprocessing=False)[0]
        scores = self._vectors[[self._rows[word] for word in words]] @ (query / np.linalg.norm(query))
        return [words[index] for index in np.argsort(-scores, kind="stable")]

    def related(self, words: Sequence[str]) -> Dict[str, str]:
        """The dictionary's words WordNet ties to any of `words` (a clue's, lower case; more words joined by _, as
        WordNet names them: body_part), each with how: 'synonym', 'similar', 'opposite', 'related form' or 'kind of';
        {} without WordNet."""
        self._loaded.wait(LOAD_WAIT_S)
        if self.relation_problem:
            return {}
        found: Dict[str, str] = {}
        for word in words:
            head, _, last = word.lower().rpartition("_")
            names = [word.lower(), self.base(word)] + ([f"{head}_{self.base(last)}"] if head else [])  # body_parts
            for name in dict.fromkeys(names):
                for part in self._ties.get(name, "").split("\t"):
                    how, _, bases = part.partition(":")
                    for base in bases.split():
                        for each in self._forms.get(base, ()):
                            found.setdefault(each, TIES[how])
        return found

    def base_forms(self, word: str) -> Set[str]:
        """`word` in lower case and, for a word of the list, what it is a form of: 'spotted' -> {'spotted', 'spot'}."""
        return {word.lower(), self.base(word)}

    def base(self, word: str) -> str:
        """What `word` is a form of, lower case -- the one word that LION and LIONS, MEAN and MEANT, are -- or `word`
        itself, outside the list."""
        return self._base.get(word.upper(), word.lower())

    def respelled(self, text: str) -> str:
        """`text` with each word of FIX_LETTERS or more that is not known -- in the word list, WordNet or the
        embedding model's own words -- in the one known word fewest edits away, at most FIX_EDITS: 'synonym for
        LABIRINTH' -> 'synonym for labyrinth'. A word with more than one as near is left out, not guessed: "witter" is
        winter, wither, wetter, and neither the commonest nor the nearest in meaning was wither (STATUS-kiwi §13).
        Without the model, as it is: its words are the everyday ones WordNet lacks, SOMETHING among them."""
        self._loaded.wait(LOAD_WAIT_S)
        if self.meaning_problem:
            return text
        fixed = "".join(self._respelled_word(part) if part.isalpha() else part for part in _split(text))
        return re.sub(r"\s+([,;.])", r"\1", " ".join(fixed.split())).strip(" ,;.")

    # --- loading -------------------------------------------------------------------------------------------------

    def _load(self) -> None:
        started = time.monotonic()
        try:
            self._read_words()
            self.problem = ""
        except (OSError, ValueError) as error:
            self.problem = f"the word list could not be read: {error}"
        finally:
            self._listed.set()
        try:
            self.meaning_problem = self.problem or self._read_meaning()
        except Exception as error:                  # a bad download, a model2vec that changed: said, not raised
            self.meaning_problem = f"{MODEL} could not be loaded: {type(error).__name__}: {error}"
        try:
            self.relation_problem = self.problem or self._read_wordnet()
        except (OSError, ValueError) as error:
            self.relation_problem = f"{WORDNET_FILE.name} could not be read: {error}"
        listed = {word.lower() for word in self.words}
        self._known |= listed
        for word in sorted(listed | set(self._ties)):       # not the embedding model's words: names, pieces
            if len(word) >= 3:
                self._spellings.setdefault((word[0], len(word)), []).append(word)
        self._loaded.set()
        if self.problem:
            said = f"off, {self.problem}"
        else:
            lacking = [problem for problem in (self.meaning_problem, self.relation_problem) if problem]
            said = (f"{len(self.words)} words, by meaning and WordNet too ({time.monotonic() - started:.1f} s)"
                    if not lacking else f"{len(self.words)} words; without {'; '.join(lacking)}")
        print(f"dictionary : {said}", flush=True)

    def _read_words(self) -> None:
        listed = [line.split("\t") for line in self.path.read_text().splitlines() if line and line[0] != "#"]
        zipf = {columns[0]: float(columns[1]) for columns in listed}
        self._base = {columns[0]: columns[2] for columns in listed if len(columns) > 2}
        for word in zipf:
            self._forms.setdefault(self._base.get(word, word.lower()), []).append(word)
        if not zipf or not all(word.isascii() and word.isalpha() and word.isupper() for word in zipf):
            raise ValueError(f"{self.path.name} holds no words, or more than capitals")
        words = sorted(zipf, key=len, reverse=True)     # stable: of a length, the file's order, commonest first
        self._counts = np.array([_counted(word) for word in words], np.uint8)
        self._lengths = np.array([len(word) for word in words])
        self.words, self.zipf = words, zipf

    def _read_meaning(self) -> str:
        """The model and every word's vector; "" once loaded, or why not."""
        try:
            from huggingface_hub import hf_hub_download
            from model2vec import StaticModel
        except ImportError:
            return "model2vec is not installed (src/requirements.txt)"
        try:
            paths = [hf_hub_download(MODEL, name, revision=REVISION, local_files_only=True) for name in MODEL_FILES]
        except Exception:                           # huggingface_hub's LocalEntryNotFoundError, and the like
            return f"{MODEL} is not downloaded (scripts/install.sh)"
        model = StaticModel.from_pretrained(Path(paths[0]).parent)
        vectors = model.encode([word.lower() for word in self.words], use_multiprocessing=False)
        self._vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self._rows = {word: row for row, word in enumerate(self.words)}
        self._model = model
        self._known |= {token for token in model.tokenizer.get_vocab() if token.isascii() and token.isalpha()}
        return ""

    def _read_wordnet(self) -> str:
        """WordNet's ties, and its words as known ones; "" once read."""
        lines = [line for line in WORDNET_FILE.read_text().splitlines() if line and line[0] != "#"]
        self._ties = {name: rest for name, _, rest in (line.partition("\t") for line in lines)}
        self._known |= set(self._ties)
        return ""

    def _respelled_word(self, part: str) -> str:
        word = part.lower()
        if len(word) < FIX_LETTERS or word in self._known:
            return part
        nearest: Dict[str, int] = {}
        for length in range(len(word) - FIX_EDITS, len(word) + FIX_EDITS + 1):
            for known in self._spellings.get((word[0], length), ()):
                edits = _edits(word, known, FIX_EDITS)
                if edits <= FIX_EDITS:
                    nearest[known] = edits
        fewest = [known for known, edits in nearest.items() if edits == min(nearest.values())]
        return fewest[0] if len(fewest) == 1 else "" if fewest else part


def _edits(word: str, other: str, most: int) -> int:
    """Letters added, dropped, changed, or two next to each other swapped, from `word` to `other`; `most` + 1 once
    more than `most` are needed."""
    before, row = None, list(range(len(other) + 1))
    for i, letter in enumerate(word, 1):
        current = [i] + [0] * len(other)
        for j, other_letter in enumerate(other, 1):
            current[j] = min(row[j] + 1, current[j - 1] + 1, row[j - 1] + (letter != other_letter))
            if i > 1 and j > 1 and letter == other[j - 2] and word[i - 2] == other_letter:
                current[j] = min(current[j], before[j - 2] + 1)
        if min(current) > most:
            return most + 1
        before, row = row, current
    return row[-1]


def _split(text: str) -> List[str]:
    """`text` in runs of letters and runs of the rest, all of it: joined again, it is `text`."""
    parts: List[str] = []
    for character in text:
        if parts and parts[-1][-1].isalpha() == character.isalpha():
            parts[-1] += character
        else:
            parts.append(character)
    return parts


def _counted(letters: str) -> np.ndarray:
    """How many of each letter, A to Z, `letters` has; anything else is not counted."""
    codes = np.frombuffer(letters.upper().encode("ascii", "ignore"), np.uint8).astype(np.int64) - ord("A")
    return np.bincount(codes[(codes >= 0) & (codes < 26)], minlength=26)
