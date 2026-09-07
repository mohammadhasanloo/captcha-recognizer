"""Train the recogniser and write the artefacts the README points at."""

from __future__ import annotations

import random
from pathlib import Path

import keras
import numpy as np

from captcha_recognizer.data import load_dataset, train_test_split
from captcha_recognizer.demo import (
    prediction_grid,
    training_curve,
    write_metrics,
    write_predictions_csv,
)
from captcha_recognizer.evaluate import Scores, score
from captcha_recognizer.model import build_model

DEFAULT_SEED = 0
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 32


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy and Keras so a run can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def train(
    samples_dir: Path,
    model_path: Path,
    docs_dir: Path,
    results_dir: Path,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
) -> Scores:
    set_seeds(seed)

    dataset, vocabulary = load_dataset(samples_dir, seed=seed)
    train_set, test_set = train_test_split(dataset, train_fraction=0.8)
    print(
        f"{len(dataset)} samples, {len(vocabulary)} distinct characters "
        f"({''.join(vocabulary.characters)})"
    )
    print(f"{len(train_set)} train / {len(test_set)} test")

    model = build_model(num_classes=len(vocabulary))
    history = model.fit(
        train_set.images,
        train_set.head_labels(),
        validation_data=(test_set.images, test_set.head_labels()),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    scores = score(model, test_set)
    print("\nheld-out accuracy")
    print(scores.format())

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    training_curve(history.history, docs_dir / "training_curve.png")
    prediction_grid(model, test_set, vocabulary, docs_dir / "demo.png")
    write_metrics(scores, results_dir / "metrics.json")
    write_predictions_csv(model, test_set, vocabulary, results_dir / "predictions.csv")
    print(f"\nsaved model to {model_path}, figures to {docs_dir}, results to {results_dir}")
    return scores
