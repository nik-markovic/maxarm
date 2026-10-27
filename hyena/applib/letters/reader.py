"""The letter reader: a small int8 network that says which capital a tile face shows, if that turn is upright.

Asked about a face at each of its four quarter turns, it says of each "this
letter, if this turn is the right way up" or "not upright". It is trained on
drawn hand-printed capitals (`training/letters/`), and runs on whichever TFLite
interpreter is installed: `tflite_runtime` on the IMX95, LiteRT or TensorFlow on
a PC.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MODEL = Path(__file__).resolve().parent.parent.parent / "files" / "letters.tflite"
SIZE = 64                   # the reader's input, pixels square
WINDOW_MM = 27.0            # what it sees: the 25 mm face and a millimetre round it

# Capitals that read as themselves turned, by quarter turns: an N upside down is an N. An M upside down reads as a
# W, and the reader is taught so (`training/letters/render.py`): only the dot at the bottom right tells them apart.
SAME_TURNED = {1: "OX", 2: "HINOSXZ", 3: "OX"}


class Reader:
    def __init__(self, model: Path = MODEL):
        self.interpreter = _interpreter(model)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]

    def probabilities(self, face: np.ndarray) -> np.ndarray:
        """For one grey face, SIZE square: each capital's probability, that turn upright, then "not upright"."""
        self.interpreter.set_tensor(self.input["index"], normalise(face)[None, :, :, None])
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output["index"])[0].astype(np.float32)
        scale, zero = self.output["quantization"]
        return (raw - zero) * scale if scale else raw


def normalise(grey: np.ndarray) -> np.ndarray:
    """Face near white, ink near black, whatever the exposure: the same in training."""
    grey = grey.astype(np.float32)
    middle = grey[SIZE // 4:3 * SIZE // 4, SIZE // 4:3 * SIZE // 4]
    low, high = np.percentile(middle, 1), np.percentile(middle, 90)
    return np.clip((grey - low) / max(high - low, 1.0) * 255, 0, 255).astype(np.uint8)


def _interpreter(model: Path):
    try:
        from tflite_runtime.interpreter import Interpreter          # the IMX95's
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tensorflow.lite import Interpreter             # type: ignore[no-redef]
            except ImportError as error:
                raise ImportError("no TFLite interpreter: tflite_runtime (IMX95), "
                                  "ai-edge-litert or tensorflow") from error
    return Interpreter(model_path=str(model))
