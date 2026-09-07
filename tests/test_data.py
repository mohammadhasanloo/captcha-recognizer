"""Tests for loading, encoding and splitting the sample set."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from captcha_recognizer.data import (
    CAPTCHA_LENGTH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    Vocabulary,
    denoise,
    load_dataset,
    train_test_split,
)

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def test_vocabulary_is_independent_of_insertion_order():
    """The mapping must depend only on which characters appear, not their order."""
    assert Vocabulary("bca").characters == Vocabulary("acb").characters
    assert Vocabulary("bca").encode("abc") == Vocabulary("acb").encode("abc")


def test_vocabulary_round_trips():
    vocabulary = Vocabulary("2345678bcdefgmnpwxy")
    assert vocabulary.decode(vocabulary.encode("226md")) == "226md"


def test_denoise_preserves_shape_and_binarises():
    image = np.random.default_rng(0).integers(0, 256, (IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    cleaned = denoise(image)
    assert cleaned.shape == image.shape
    assert set(np.unique(cleaned)).issubset({0, 255})


def test_load_dataset_is_reproducible_for_a_seed():
    first, _ = load_dataset(SAMPLES, seed=7)
    second, _ = load_dataset(SAMPLES, seed=7)
    assert first.texts == second.texts
    assert np.array_equal(first.labels, second.labels)


def test_load_dataset_shapes_and_labels_agree():
    dataset, vocabulary = load_dataset(SAMPLES, seed=0)
    assert dataset.images.shape == (len(dataset), IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    assert dataset.labels.shape == (len(dataset), CAPTCHA_LENGTH)
    assert dataset.images.min() >= 0.0 and dataset.images.max() <= 1.0
    # Every label decodes back to the filename it came from.
    for text, row in zip(dataset.texts[:50], dataset.labels[:50]):
        assert vocabulary.decode(row) == text


def test_train_test_split_is_disjoint_and_covers_everything():
    dataset, _ = load_dataset(SAMPLES, seed=0)
    train_set, test_set = train_test_split(dataset, train_fraction=0.8)
    assert len(train_set) + len(test_set) == len(dataset)
    assert not set(train_set.paths) & set(test_set.paths)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5, 2.0])
def test_train_test_split_rejects_bad_fractions(fraction):
    dataset, _ = load_dataset(SAMPLES, seed=0)
    with pytest.raises(ValueError):
        train_test_split(dataset, train_fraction=fraction)


def test_head_labels_split_by_position():
    dataset, _ = load_dataset(SAMPLES, seed=0)
    heads = dataset.head_labels()
    assert len(heads) == CAPTCHA_LENGTH
    assert all(head.shape == (len(dataset),) for head in heads)
