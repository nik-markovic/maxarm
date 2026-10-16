#!/usr/bin/env python3
"""Train the tile reader on synthetic faces and export it as an int8 TFLite model.

The reader is asked about one tile face in each of its four quarter turns and
answers, for each, which letter it is if that turn is upright -- or that the
turn is not upright. So one small network both reads the letter and says which
way the tile faces: the subscript's corner and the letter's left offset are
what tell an O, an N or a Z its way up, and it learns those from the renders.

    ./training/tiles/make-reader.sh                  # from scratch: fonts, this, the check
    ../.venv/bin/python training/tiles/train.py      # -> files/reader.tflite, reader.json

Host-side only. The result is what `tflite_runtime` runs on the IMX95, and what
the Neutron converter takes.

Every random choice is seeded -- the renders by `render`'s own seeds per batch,
the network's initial weights, shuffling and dropout by `SEED` -- and
TensorFlow is held to deterministic ops, so the same fonts and package versions
(`fonts.txt`, `requirements.txt`) give the same model on the same machine.
`files/reader.json` records what went in and the model's hash, to compare with.
"""

import hashlib
import json
import multiprocessing
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import numpy as np                                   # noqa: E402

import render                                        # noqa: E402

SAMPLES = 240_000
VALIDATION = 12_000
EPOCHS = 14
SEED = 0
MODEL = ROOT / "files" / "reader.tflite"
RECORD = ROOT / "files" / "reader.json"


def make_batch(seed_count):
    seed, count = seed_count
    rng = random.Random(seed)
    faces = render.list_faces()
    images, labels = [], []
    while len(images) < count:
        letter = rng.choice(render.LETTERS)
        kind = rng.random()
        # Half upright letters; the rest turned letters and, a fifth of all,
        # things a letter finder turns up that are not letters at all.
        turn = 0 if kind < 0.45 else rng.randint(0, 3) if kind >= 0.8 else rng.randint(1, 3)
        negative = rng.choice(render.NEGATIVES) if kind >= 0.8 else ""
        image = render.render(rng.choice(faces), letter, turn, rng, negative)
        if image is None:
            continue
        images.append(image)
        labels.append(render.LETTERS.index(letter) if turn == 0 and not negative else render.TURNED)
    return np.array(images, np.uint8), np.array(labels, np.int64)


def dataset(total, seed):
    chunk = 2000
    jobs = [(seed * 100_000 + index, chunk) for index in range(total // chunk)]
    with multiprocessing.Pool() as pool:
        parts = pool.map(make_batch, jobs)
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


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
    # Flatten, not global pooling: where the subscript is relative to the
    # letter is the whole of how an O or an N shows its way up.
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
    # The best epoch is what gets exported, not the last: one run reached 0.9995
    # on validation at epoch 5 and collapsed to "not upright" for everything at 6.
    keras_path = ROOT / "files" / "reader.keras"
    model.fit(train_x[..., None], train_y, batch_size=256, epochs=EPOCHS,
              validation_data=(valid_x[..., None], valid_y), verbose=2,
              callbacks=[tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3),
                         tf.keras.callbacks.ModelCheckpoint(keras_path, monitor="val_loss",
                                                            save_best_only=True)])
    model = tf.keras.models.load_model(keras_path)
    _, accuracy = model.evaluate(valid_x[..., None], valid_y, verbose=0)
    print(f"best epoch: validation accuracy {accuracy:.4f}")

    def representative():
        for index in range(0, 600):
            yield [valid_x[index:index + 1, ..., None].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    MODEL.write_bytes(converter.convert())
    print(f"wrote {MODEL.relative_to(ROOT)} ({MODEL.stat().st_size // 1024} KB) "
          f"in {time.monotonic() - started:.0f} s")
    RECORD.write_text(json.dumps(record(accuracy), indent=2) + "\n")
    print(f"wrote {RECORD.relative_to(ROOT)}")
    return 0


def record(accuracy: float) -> dict:
    """What went into the model, and its hash: a rebuild matches this or says how it differs."""
    import cv2
    import PIL
    import tensorflow as tf

    fonts = sorted(render.FONT_DIR.rglob("*.ttf"))
    font_hash = hashlib.sha256()
    for path in fonts:
        font_hash.update(str(path.relative_to(render.FONT_DIR)).encode())
        font_hash.update(path.read_bytes())
    commit = subprocess.run(["git", "-C", str(render.FONT_DIR), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    return {
        "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "validation_accuracy": round(float(accuracy), 4),
        "fonts": {"google_fonts_commit": commit, "files": len(fonts),
                  "faces": len(render.list_faces()), "sha256": font_hash.hexdigest()},
        "training": {"samples": SAMPLES, "validation": VALIDATION, "epochs": EPOCHS, "seed": SEED},
        "versions": {"python": platform.python_version(), "tensorflow": tf.__version__,
                     "keras": tf.keras.__version__, "numpy": np.__version__,
                     "pillow": PIL.__version__, "opencv": cv2.__version__},
        "machine": platform.machine(),
        "built": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }


if __name__ == "__main__":
    sys.exit(main())
