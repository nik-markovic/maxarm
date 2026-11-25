"""The language model on the Ara240, and what it may do with the tiles: both apps' -- the /IOTCONNECT app's `agent`
command and the calibration app's Chat page.

    source = AraConnector(on_change=...)
    agent = AgentService(source, Desk(seen=..., detect=..., check=word_check(work, settings), place=...))

| module       | what it owns                                                                            |
| ------------ | --------------------------------------------------------------------------------------- |
| `service`    | `AgentService`: a request answered by the model, the tiles its tools                    |
| `connector`  | where the model is -- the Ara240's connector -- how far it has got, and its sampling    |
| `prompt`     | what the model is told: the system prompt, and the message each request becomes        |
| `finder`     | the word a request asks for: the user's own, the longest, or one for a clue             |
| `dictionary` | which words there are, which the tiles spell, which are near a clue: meaning, WordNet  |
| `words`      | words checked against the tiles' letters, in code                                       |
| `repair`     | a tool call the model wrote out as text, made the call it meant                         |
| `watch`      | the model's calls counted, and what went to and from it kept for the page              |
| `chat`       | the conversation: in full for the pages, a line of it for the model                     |
| `desk`       | what the tools do with the tiles, as each app does it                                   |

strands-agents is needed for `AgentService` alone (`IS_STRANDS_AVAILABLE`); without it the apps run, no agent.
model2vec is needed for words found by meaning alone; without it a clue is refused, all else answered. Without
files/words-wordnet.tsv, a clue's words are found by meaning and the model's ideas alone.
"""

from .connector import AraConnector, describe_status, describe_unready
from .desk import Desk, word_check
from .service import IS_STRANDS_AVAILABLE, AgentService

__all__ = ["AgentService", "AraConnector", "Desk", "IS_STRANDS_AVAILABLE", "describe_status", "describe_unready",
           "word_check"]
