#!/usr/bin/env python3
"""Train the letter reader on synthetic hand-printed tiles and export it as an int8 TFLite model.

The reader is shown a tile's face at each of its four quarter turns and says, for
each, which capital it is if that turn is the right way up -- or that it is not.
So one small network both reads the letter and says which way the tile faces.
The faces are drawn by `render.py`; nothing is downloaded.

    ../.venv/bin/python training/letters/train.py     # -> files/letters.tflite, files/letters.json, ~15 min

Host-side only; the result is what `tflite_runtime` runs on the IMX95. Every
random choice is seeded -- the renders per batch, the network's weights,
shuffling and dropout by `SEED` -- and TensorFlow is held to deterministic ops,
so the same package versions (`requirements.txt`) give the same model on the same
machine. `files/letters.json` records what went in and the model's hash.
"""

import hashlib
import json
import multiprocessing
import os
import platform
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import numpy as np                                   # noqa: E402

import render                                        # noqa: E402

SAMPLES = 300_000
VALIDATION = 12_000
EPOCHS = 16
SEED = 0
MODEL = ROOT / "files" / "letters.tflite"
RECORD = ROOT / "files" / "letters.json"
KERAS = HERE / "letters.keras"                       # the best epoch, kept here while training


def make_batch(seed_count):
    seed, count = seed_count
    r = random.Random(seed)
    images, labels = zip(*(render.sample(r) for _ in range(count)))
    return np.array(images, np.uint8), np.array(labels, np.int64)


def dataset(total, seed):
    chunk = 2000
    jobs = [(seed * 100_000 + index, chunk) for index in range(total // chunk)]
    with multiprocessing.Pool() as pool:
        parts = pool.map(make_batch, jobs)
    return np.concatenate([part[0] for part in parts]), np.concatenate([part[1] for part in parts])


def build():
    import tensorflow as tf
    from tensorflow.keras import layers

    inputs = tf.keras.Input((render.SIZE, render.SIZE, 1))
    x = layers.Rescaling(1.0 / 255)(inputs)
    for filters in (16, 32, 64, 96):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    # Flatten, not global pooling: where the dot is, bottom right or not, is how an M tells itself from a W.
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(render.LETTERS) + 1, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def main() -> int:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    started = time.monotonic()
    train_x, train_y = dataset(SAMPLES, seed=1)
    valid_x, valid_y = dataset(VALIDATION, seed=2)
    print(f"rendered {len(train_x)} + {len(valid_x)} in {time.monotonic() - started:.0f} s")

    model = build()
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # The best epoch is exported, not the last: dolphin's reader once collapsed to "not upright" a step after its best.
    model.fit(train_x[..., None], train_y, batch_size=256, epochs=EPOCHS,
              validation_data=(valid_x[..., None], valid_y), verbose=2,
              callbacks=[tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3),
                         tf.keras.callbacks.ModelCheckpoint(KERAS, monitor="val_loss", save_best_only=True)])
    model = tf.keras.models.load_model(KERAS)
    _, accuracy = model.evaluate(valid_x[..., None], valid_y, verbose=0)
    print(f"best epoch: validation accuracy {accuracy:.4f}")

    def representative():
        for index in range(600):
            yield [valid_x[index:index + 1, ..., None].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    MODEL.parent.mkdir(exist_ok=True)
    MODEL.write_bytes(converter.convert())
    KERAS.unlink()
    print(f"wrote {MODEL.relative_to(ROOT)} ({MODEL.stat().st_size // 1024} KB) in {time.monotonic() - started:.0f} s")
    RECORD.write_text(json.dumps(record(accuracy), indent=2) + "\n")
    print(f"wrote {RECORD.relative_to(ROOT)}")
    return 0


def record(accuracy: float) -> dict:
    """What went into the model, and its hash: a rebuild matches this or says how it differs."""
    import cv2
    import PIL
    import tensorflow as tf

    return {
        "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "render_sha256": hashlib.sha256((HERE / "render.py").read_bytes()).hexdigest(),
        "validation_accuracy": round(float(accuracy), 4),
        "training": {"samples": SAMPLES, "validation": VALIDATION, "epochs": EPOCHS, "seed": SEED},
        "versions": {"python": platform.python_version(), "tensorflow": tf.__version__,
                     "keras": tf.keras.__version__, "numpy": np.__version__,
                     "pillow": PIL.__version__, "opencv": cv2.__version__},
        "machine": platform.machine(),
        "built": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }


if __name__ == "__main__":
    sys.exit(main())
