"""The overlay's colours, as the pages had them: given to the pages too (`to_json()`), so that the view from above,
the tables and the picture keep to one set."""

from .canvas import ink

STATUS = {"ready": "#41c363", "too close": "#ffd23f", "elsewhere": "#9db0c8", "out of reach": "#f0524a"}
AREA = {"pickup": "#41c363", "dropoff": "#4c8dff"}
# A letter of the word by how its tile is doing: to come, being carried, set down; anything else, stopped, refused or
# failed, red.
WORD = {"waiting": "#ffffff", "carrying": "#ffd23f", "placed": "#4c8dff", "failed": "#f0524a"}
CARRIED = "#9db0c8"
UNREAD = "#f0524a"         # a letter not read: its tile's outline and its '?' (the owner's, PILOT-jackal)
REACH = "#f0524a"
REACH_MIN = "#f08c3c"
CUBE = "#ff4df8"
YELLOW = "#ffd23f"
WHITE = "#ffffff"
SHADOW = "#05070a"
BADGE_TEXT = "#0b0f14"
PILL = "#0a0d11"


def word_ink(status: str) -> str:
    return WORD.get(status, WORD["failed"])


def to_json() -> dict:
    return {"status": STATUS, "area": AREA, "word": WORD, "carried": CARRIED, "unread": UNREAD}


BGR = {name: ink(value) for name, value in {
    "reach": REACH, "reach_min": REACH_MIN, "cube": CUBE, "yellow": YELLOW, "white": WHITE, "shadow": SHADOW,
    "badge_text": BADGE_TEXT, "pill": PILL, "carried": CARRIED, "unread": UNREAD}.items()}
