"""Scoring a trained model."""

from __future__ import annotations

from dataclasses import dataclass

import keras
import numpy as np

from captcha_recognizer.data import CAPTCHA_LENGTH, Dataset, Vocabulary


@dataclass(frozen=True)
class Scores:
    """Per-position and whole-captcha accuracy."""

    per_character: tuple[float, ...]
    exact_match: float

    @property
    def mean_character(self) -> float:
        return float(np.mean(self.per_character))

    def format(self) -> str:
        lines = [
            f"  character {i + 1}: {accuracy:6.2%}"
            for i, accuracy in enumerate(self.per_character)
        ]
        lines.append(f"  mean per-character: {self.mean_character:6.2%}")
        lines.append(f"  whole captcha:      {self.exact_match:6.2%}")
        return "\n".join(lines)


def predict_indices(model: keras.Model, images: np.ndarray) -> np.ndarray:
    """Predict character indices for a batch of images, shape (n, CAPTCHA_LENGTH).

    A single batched call, rather than one prediction per image: the per-call
    overhead dominates on a test set of a few hundred samples.
    """
    head_probabilities = model.predict(images, verbose=0)
    return np.stack([probabilities.argmax(axis=1) for probabilities in head_probabilities], axis=1)


def predict_texts(model: keras.Model, images: np.ndarray, vocabulary: Vocabulary) -> list[str]:
    return [vocabulary.decode(row) for row in predict_indices(model, images)]


def score(model: keras.Model, dataset: Dataset) -> Scores:
    """Accuracy per character position, and the fraction of fully correct captchas.

    Whole-captcha accuracy is the number that matters for a captcha solver: all
    five characters have to be right for a submission to be accepted. It is much
    lower than the per-character figures, which is why it is reported here.
    """
    predicted = predict_indices(model, dataset.images)
    correct = predicted == dataset.labels
    return Scores(
        per_character=tuple(float(correct[:, i].mean()) for i in range(CAPTCHA_LENGTH)),
        exact_match=float(correct.all(axis=1).mean()),
    )
