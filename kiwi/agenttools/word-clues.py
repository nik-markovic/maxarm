#!/usr/bin/env python3
"""Clues put to the agent's word finder (`applib/agent/finder.py`) with the model on the Ara240 -- the finder's own
list, ideas and choice -- each against the words a person would take for it. Nothing moved: no arm, no camera, no
strands. Before and after a change to how a clue's words are found (STATUS-kiwi §13). On the board, the connector
ready; on a PC, through an SSH tunnel to its port 3000:

    venv/bin/python agenttools/word-clues.py                    # every clue below, ~10 s each
    venv/bin/python agenttools/word-clues.py "noun for avenge"  # those whose clue starts so

A word not among a clue's answers is not always wrong -- RHINO for "an animal" -- so the misses are to be read: a
fair word, or a bad one. Answers are spelled by their tiles and in the word list, checked first.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from applib.agent.connector import AraConnector                  # noqa: E402
from applib.agent.dictionary import Dictionary                   # noqa: E402
from applib.agent.finder import WordFinder                       # noqa: E402
from applib.agent.words import meaningful_clue, missing_letters, wanted_length  # noqa: E402

OWNER, CHECK, MLAMP, OWNER_RIL = "SOVNTZEEWADQTNM", "AEEHMORST", "AABEEGHIINOORRTU", "SOVNTREEWADITLM"
SPARE, MOUNTAIN = "GARDENSPLITOEUC", "MOUNTAINBREVLSE"
CLUES = [   # tiles, clue, the answers a person would take
    (OWNER, "A small swamp animal", "NEWT TOAD"), (CHECK, "A large herbivore domestic animal", "HORSE"),
    (OWNER, "a fruit", "DATE"), (CHECK, "something you sit on", "SEAT"), (CHECK, "a flower", "ROSE"),
    (OWNER, "a body part", "TOE NOSE"), (MLAMP, "a planet", "EARTH"), (OWNER, "something cold", "SNOW"),
    (CHECK, "a pet", "HAMSTER"), (MLAMP, "a musical instrument", "GUITAR ORGAN"),
    (OWNER, "synonym for labirinth", "MAZE"), (OWNER, "synonym for LABIRINTH", "MAZE"),
    (OWNER, "synonym for country", "STATE"), (OWNER, "similar for verse", "ODE SONNET"),
    (OWNER, "to witter, dry out, expire", "WANE WASTE"), (CHECK, "a red flowr", "ROSE"),
    (OWNER, "noun for avenge", "VENDETTA"), (OWNER, "synonym for tidy", "NEAT"), (OWNER, "synonym for angry", "MAD"),
    (MLAMP, "synonym for big", "HUGE GIANT GREAT"), (OWNER_RIL, "another word for gift", "TALENT"),
    (OWNER, "opposite of old", "NEW"), (OWNER, "opposite of start", "END"), (OWNER_RIL, "a word for anger", "IRE"),
    (MLAMP, "a color", "ORANGE BEIGE GREEN TAN"), (MLAMP, "an animal", "TIGER GOAT RHINO BEAR BOAR"),
    (MLAMP, "a vegetable", "BEET BEAN"), (MLAMP, "a vehicle", "BOAT AUTO TRAIN"), (OWNER_RIL, "a job", "TAILOR"),
    (OWNER_RIL, "the king of the jungle", "LION"), (OWNER, "something at a trade show", "DEMO STAND"),
    (OWNER_RIL, "a large body of water", "SEA"), (OWNER_RIL, "a place to sleep", "MOTEL DORM"),
    (MLAMP, "a metal", "IRON"), (CHECK, "a hot drink", "TEA"), (OWNER_RIL, "to eat dinner", "DINE"),
    (OWNER, "a bird", "DOVE SWAN"), (OWNER, "a gift of money to a school", "ENDOWMENT"),
    (OWNER, "making amends for a wrong", "ATONEMENT"), (OWNER_RIL, "to show how something works", "DEMONSTRATE DEMO"),
    (SPARE, "a spy's work", "ESPIONAGE"), (SPARE, "risky, unsafe", "DANGEROUS PERILOUS"),
    (SPARE, "a teenager", "ADOLESCENT"), (SPARE, "where the land meets the sea", "COASTLINE"),
    (SPARE, "a celebrity's group of followers", "ENTOURAGE"), (SPARE, "synonym for despair", "DESPERATION"),
    (SPARE, "a short funny story about a real event", "ANECDOTE"), (SPARE, "a small green vegetable", "PEA"),
    (SPARE, "a spotted big cat", "LEOPARD"), (SPARE, "noun for educate", "EDUCATION EDUCATOR"),
    (MOUNTAIN, "a ship that goes underwater", "SUBMARINE"), (MOUNTAIN, "someone who helps for free", "VOLUNTEER"),
    (MOUNTAIN, "a lift in a building", "ELEVATOR"), (MOUNTAIN, "something bought to remember a trip", "SOUVENIR"),
    (MOUNTAIN, "synonym for wonderful", "MARVELOUS"), (MOUNTAIN, "an important event in a project", "MILESTONE"),
    (MOUNTAIN, "a card for your sweetheart", "VALENTINE"), (MOUNTAIN, "noun for reveal", "REVELATION"),
    (MOUNTAIN, "opposite of happy", "MISERABLE"), (MOUNTAIN, "a very tall hill", "MOUNTAIN"),
]


def main() -> int:
    chosen = [(tiles, clue, answers) for tiles, clue, answers in CLUES
              if not sys.argv[1:] or any(clue.lower().startswith(text.lower()) for text in sys.argv[1:])]
    words = Dictionary()
    words.spelled("")                                       # waits for the list
    words.respelled("")                                     # and for meaning and WordNet
    for tiles, clue, answers in chosen:
        bad = [answer for answer in answers.split() if missing_letters(answer, tiles) or answer not in words.zipf]
        if bad:
            print(f"the answers for {clue!r} are not all on the tiles and in the word list: {' '.join(bad)}")
            return 1
    finder = WordFinder(words, AraConnector())
    right, times = 0, []
    for tiles, clue, answers in chosen:
        started = time.monotonic()
        found = finder.find(tiles, meaningful_clue(clue), "", wanted_length(clue), [], None)
        times.append(time.monotonic() - started)
        is_right = found.word in answers.split()
        right += is_right
        chose = found.word or f"none<{found.said}>"
        print(f"{'ok' if is_right else '--'} {clue[:36]:36} {chose:14} ideas {found.ideas_seconds:4.1f} s, "
              f"{times[-1]:4.1f} s in all: {' '.join(found.shown)}", flush=True)
    print(f"\n{right} of {len(chosen)} among the answers; {sum(times) / len(times):.1f} s a clue, "
          f"{max(times):.1f} at most")
    return 0


if __name__ == "__main__":
    sys.exit(main())
