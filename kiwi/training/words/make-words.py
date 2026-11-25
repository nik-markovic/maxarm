#!/usr/bin/env python3
"""Make the agent's word list, files/words-en.tsv -- everyday English words, how common each is, and what each is a
form of -- and files/words-wordnet.tsv, which of them WordNet ties to each of its words.

    ../.venv/bin/python training/words/make-words.py      # -> both files, seconds

with `training/words/requirements.txt` installed; WordNet's data is fetched by nltk the first time.

How common: wordfreq's Zipf frequency -- 6 is "the", 4 "garden", 3 "toad" -- of film subtitles, Wikipedia, news,
books and the web, a snapshot of about 2021 that is no longer updated. Which are words: the PC's Hunspell en_US
dictionary, made from SCOWL (Fedora's hunspell-en-US, Debian's hunspell-en-us), asked in lower case, so that names
drop out; and WordNet, words with a meaning of their own -- inflections taken -- so that THE, WITH, BECAUSE and
GONNA drop out too, the commonest words of a length otherwise (STATUS-kiwi §7). Kept: 3 to 15 letters, A to Z,
Zipf MIN_ZIPF or more, but NOT_WORDS. A word's form, where it is one: the shortest of WordNet's base forms -- LIONS
lion, MEANT mean -- so that a list shows one of them.

WordNet's ties, for a clue's words (STATUS-kiwi §13): for each of WordNet's words, the list's words that share a
sense with it, have a similar one, the opposite one, a form made from it, or are a kind of it, up to KINDS_DEPTH
steps down -- not of one with more than KINDS_MOST kinds in the list ("entity", "object", "part"), which say nothing.
Its names of more words too, joined by _, where the list's words tie to them: "a body part" is body_part, whose kinds
are NOSE and TOE, where "part" is too wide. Every single word has its line, if only its name: the board knows from
them which words are spelled right.
Made here once: WordNet itself, through nltk, is far too slow to load on the board (STATUS-kiwi §13).

Host-side only; the board reads the files (`src/applib/agent/dictionary.py`). wordfreq's data is CC BY-SA 4.0, with
credits to the sources it was made from (https://github.com/rspeer/wordfreq); SCOWL's terms are MIT-like, and
WordNet's (Princeton) BSD-like.
"""

import sys
from collections import defaultdict
from pathlib import Path

import nltk
from nltk.corpus import wordnet
from spylls.hunspell import Dictionary
from wordfreq import iter_wordlist, zipf_frequency

FILES = Path(__file__).resolve().parents[2] / "files"
OUT = FILES / "words-en.tsv"
WORDNET_OUT = FILES / "words-wordnet.tsv"
HUNSPELL = "/usr/share/hunspell/en_US"
MIN_ZIPF = 2.5             # WANE 2.83 for a clue; the longest and a length take 3 or more (finder.py)
LETTERS = range(3, 16)
KINDS_DEPTH = 12            # deep enough for a NEWT under "animal"
KINDS_MOST = 2000           # a word with more kinds in the list than this says nothing of them
POS = ("n", "v", "a", "r")
# Abbreviations and Roman numerals that Hunspell and WordNet both take: no words to set down.
NOT_WORDS = set("""
    ACC ALT APR ASAP ATM AUG CORP DEC DEPT ESP EST ETC FEB FRI GOV III INC JAN JUN LTD MISC MON NOV OCT SEC SEP SEPT
    THU THUR THURS TUE TUES VAR VOL XII XIII XIV XVI XVII XIX XXX""".split())


def main() -> int:
    nltk.download("wordnet", quiet=True)
    hunspell = Dictionary.from_files(HUNSPELL)
    kept = []
    for word in iter_wordlist("en", wordlist="large"):          # commonest first
        zipf = zipf_frequency(word, "en", wordlist="large")
        if zipf < MIN_ZIPF:
            break
        if (word.isascii() and word.isalpha() and len(word) in LETTERS and word.upper() not in NOT_WORDS
                and hunspell.lookup(word) and wordnet.synsets(word)):
            kept.append((word.upper(), zipf, base_form(word)))
    lines = ["# Everyday English words, their Zipf frequency, and the word each is a form of where it is one, made by",
             "# training/words/make-words.py from wordfreq's English list (data CC BY-SA 4.0,",
             "# https://github.com/rspeer/wordfreq), Hunspell en_US (SCOWL) and WordNet: "
             f"{len(kept)} words of {LETTERS.start} to {LETTERS.stop - 1} letters,",
             f"# Zipf {MIN_ZIPF:g} or more.",
             *(f"{word}\t{zipf:.2f}" + (f"\t{base}" if base != word.lower() else "") for word, zipf, base in kept)]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"{OUT}: {len(kept)} words")
    ties = wordnet_ties({word: base for word, _, base in kept})
    names = sorted({name.lower() for name in wordnet.all_lemma_names() if is_name(name)})
    lines = ["# For each of WordNet's words -- of more words, joined by _, where it ties any -- the words of",
             "# words-en.tsv that WordNet ties to it, as their base",
             "# forms (words-en.tsv's third column, else the word): a sense they share (synonym), a similar sense",
             "# (similar), the opposite (opposite), a form made from it (form), or a kind of it (kind), "
             f"{KINDS_DEPTH} steps down",
             f"# at most, none for a word with more than {KINDS_MOST} kinds among the words.",
             "# Made by training/words/make-words.py from WordNet 3.0: Copyright 2006 by Princeton University,",
             "# all rights reserved; its license: https://wordnet.princeton.edu/license-and-commercial-use",
             *("\t".join([name] + [f"{how}:{' '.join(sorted(ties[name][how]))}" for how in HOWS if ties[name][how]])
               for name in names if "_" not in name or name in ties)]
    written = len(lines) - 6
    WORDNET_OUT.write_text("\n".join(lines) + "\n")
    print(f"{WORDNET_OUT}: {written} WordNet names, "
          f"{sum(len(listed) for tie in ties.values() for listed in tie.values())} ties")
    return 0


HOWS = ("synonym", "similar", "opposite", "form", "kind")


def is_name(name: str) -> bool:
    """Whether a WordNet name is a word, or words joined by _ -- body_part, big_cat -- of A to Z only."""
    return name.isascii() and all(part.isalpha() for part in name.split("_"))


def base_form(word: str) -> str:
    """The shortest of what WordNet takes `word` for: 'lions' lion, 'meant' mean, 'toad' toad. Every reading, as
    `synsets()` takes them -- `morphy()` gives the first only, the word itself for TIGERS, a name in WordNet."""
    forms = {word} | {base for pos in POS for base in wordnet._morphy(word, pos)}
    return min(forms, key=lambda form: (len(form), form))


def wordnet_ties(bases: dict) -> dict:
    """name -> how -> the base forms of the words in `bases` (word -> base) that WordNet ties to it so, worked out from
    each word's side: a form ties as its base does."""
    ties = defaultdict(lambda: {how: set() for how in HOWS})
    forms = defaultdict(int)                # how many of the words each base form stands for
    for word, base in bases.items():
        forms[base] += 1
    for word, base in bases.items():
        if base != word.lower() and base.upper() in bases:
            continue                        # LIONS: LION's ties, already

        for synset in wordnet.synsets(word.lower()):
            groups = [("synonym", [synset]), ("similar", synset.similar_tos() + synset.also_sees())]
            for lemma in synset.lemmas():
                groups.append(("opposite", [other.synset() for other in lemma.antonyms()]))
                groups.append(("form", [other.synset() for other in lemma.derivationally_related_forms()]))
            groups.append(("kind", list(synset.closure(lambda each: each.hypernyms() + each.instance_hypernyms(),
                                                       depth=KINDS_DEPTH))))
            for how, synsets in groups:
                for each in synsets:
                    for name in each.lemma_names():
                        if is_name(name) and name.lower() not in (word.lower(), base):
                            ties[name.lower()][how].add(base)
    for tie in ties.values():
        if sum(forms[base] for base in tie["kind"]) > KINDS_MOST:
            tie["kind"] = set()
    return ties


if __name__ == "__main__":
    sys.exit(main())
