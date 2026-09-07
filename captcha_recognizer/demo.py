"""Figures for the README: a prediction grid and the training curve."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import keras
import matplotlib

matplotlib.use("Agg")  # render to file; no display needed
import matplotlib.pyplot as plt  # noqa: E402

from captcha_recognizer.data import Dataset, Vocabulary  # noqa: E402
from captcha_recognizer.evaluate import Scores, predict_texts, score  # noqa: E402

CORRECT_COLOUR = "#1a7f37"
WRONG_COLOUR = "#cf222e"


def prediction_grid(
    model: keras.Model,
    dataset: Dataset,
    vocabulary: Vocabulary,
    output_path: Path,
    rows: int = 4,
    columns: int = 3,
) -> Path:
    """Save a grid of held-out captchas with the model's reading of each.

    Shows the raw image rather than the denoised one: that is what the model is
    handed in practice, and it makes the strike-through line visible.
    """
    count = rows * columns
    predictions = predict_texts(model, dataset.images[:count], vocabulary)

    figure, axes = plt.subplots(rows, columns, figsize=(columns * 3.2, rows * 1.35))
    for axis, path, truth, prediction in zip(
        axes.ravel(), dataset.paths[:count], dataset.texts[:count], predictions
    ):
        axis.imshow(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), cmap="gray")
        axis.set_xticks([])
        axis.set_yticks([])
        correct = prediction == truth
        label = prediction if correct else f"{prediction}  (actual {truth})"
        axis.set_title(
            label,
            color=CORRECT_COLOUR if correct else WRONG_COLOUR,
            fontsize=11,
            fontfamily="monospace",
        )
    figure.suptitle("Held-out captchas and the model's reading", fontsize=13)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return output_path


def training_curve(history: dict[str, list[float]], output_path: Path) -> Path:
    """Save total training and validation loss per epoch."""
    figure, axis = plt.subplots(figsize=(6.5, 3.6))
    axis.plot(history["loss"], label="train")
    axis.plot(history["val_loss"], label="validation")
    axis.set_xlabel("epoch")
    axis.set_ylabel("total loss (sum over 5 heads)")
    axis.set_title("Training loss")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return output_path


def write_metrics(scores: Scores, output_path: Path) -> Path:
    """Write the scores next to the figures so the README numbers are traceable."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "per_character_accuracy": list(scores.per_character),
                "mean_character_accuracy": scores.mean_character,
                "whole_captcha_accuracy": scores.exact_match,
            },
            indent=2,
        )
        + "\n"
    )
    return output_path


def write_predictions_csv(
    model: keras.Model, dataset: Dataset, vocabulary: Vocabulary, output_path: Path
) -> Path:
    """Write every held-out prediction beside its ground truth.

    Regenerated on every run so the file always describes the model that
    produced the accompanying metrics.
    """
    predictions = predict_texts(model, dataset.images, vocabulary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["predicted", "actual", "correct"])
        for prediction, truth in zip(predictions, dataset.texts):
            writer.writerow([prediction, truth, int(prediction == truth)])
    return output_path


__all__ = [
    "prediction_grid",
    "score",
    "training_curve",
    "write_metrics",
    "write_predictions_csv",
]
