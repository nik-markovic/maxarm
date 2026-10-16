"""Reading a tile: which letter it is, and which of its four quarter turns is upright.

The reader is a small int8 network (`training/tiles/train.py`) that looks at a tile
face and says which letter it is *if that turn is upright*, or that the turn is
not upright. Asked about all four turns, the upright one is the turn it is
surest is a letter. The subscript's corner and the letter's offset to the left
are what it uses for letters that look the same turned -- O, N and Z, M and W.

The model runs on whichever TFLite interpreter is installed: `tflite_runtime`
on the IMX95, LiteRT or TensorFlow on a PC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MODEL = Path(__file__).resolve().parent.parent / "files" / "reader.tflite"
INPUT_SIZE = 64
WINDOW_MM = 22.0


@dataclass(frozen=True)
class Reading:
    letter: str
    turn: int                   # quarter turns from the fitted baseline to the letter's
    confidence: float           # the network's probability for that letter, that turn
    runner_up: float            # the best any other turn managed, for any letter


class Reader:
    def __init__(self, model: Path = MODEL):
        self.interpreter = _interpreter(model)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]

    def read(self, faces: Sequence[np.ndarray]) -> Reading:
        """`faces` are the tile's four quarter turns, grey, INPUT_SIZE square."""
        letter_scores = np.array([self.probabilities(face)[:len(LETTERS)] for face in faces])
        best_per_turn = letter_scores.max(axis=1)
        turn = int(np.argmax(best_per_turn))
        others = np.delete(best_per_turn, turn)
        return Reading(LETTERS[int(np.argmax(letter_scores[turn]))], turn,
                       float(best_per_turn[turn]), float(others.max()))

    def probabilities(self, face: np.ndarray) -> np.ndarray:
        self.interpreter.set_tensor(self.input["index"], normalise(face)[None, :, :, None])
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output["index"])[0].astype(np.float32)
        scale, zero = self.output["quantization"]
        return (raw - zero) * scale if scale else raw


def normalise(grey: np.ndarray) -> np.ndarray:
    """Face near white, ink near black, from the middle of the window. Same as training."""
    grey = grey.astype(np.float32)
    quarter = INPUT_SIZE // 4
    centre = grey[quarter:3 * quarter, quarter:3 * quarter]
    low, high = np.percentile(centre, 2), np.percentile(centre, 90)
    return np.clip((grey - low) / max(high - low, 1.0) * 255, 0, 255).astype(np.uint8)


def _interpreter(model: Path):
    try:
        from tflite_runtime.interpreter import Interpreter          # the IMX95's
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tensorflow.lite import Interpreter            # type: ignore[no-redef]
            except ImportError as error:
                raise ImportError("no TFLite interpreter: tflite_runtime (IMX95), "
                                  "ai-edge-litert or tensorflow") from error
    return Interpreter(model_path=str(model))
