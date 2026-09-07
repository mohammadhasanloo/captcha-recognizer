"""Loading and preprocessing for the captcha sample set.

Every sample is a 200x50 image whose filename is its ground truth, e.g.
``226md.png``. Both PNG and JPG samples are present.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_HEIGHT = 50
IMAGE_WIDTH = 200
CAPTCHA_LENGTH = 5
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


class Vocabulary:
    """Stable character/index mapping.

    Characters are sorted, so the mapping is a property of the dataset rather
    than of the order files happened to be read in. A model trained in one run
    therefore decodes correctly in the next.
    """

    def __init__(self, characters: str) -> None:
        self.characters: tuple[str, ...] = tuple(sorted(set(characters)))
        self._index: dict[str, int] = {c: i for i, c in enumerate(self.characters)}

    def __len__(self) -> int:
        return len(self.characters)

    def __repr__(self) -> str:
        return f"Vocabulary({''.join(self.characters)!r})"

    def encode(self, text: str) -> list[int]:
        return [self._index[character] for character in text]

    def decode(self, indices: "np.ndarray | list[int]") -> str:
        return "".join(self.characters[int(i)] for i in indices)


@dataclass(frozen=True)
class Dataset:
    """Images with their labels, index-aligned."""

    images: np.ndarray  # (n, H, W, 1) float32 in [0, 1]
    labels: np.ndarray  # (n, CAPTCHA_LENGTH) int32 vocabulary indices
    texts: tuple[str, ...]  # ground-truth strings
    paths: tuple[Path, ...]  # source files, kept so demos can show the raw image

    def __len__(self) -> int:
        return len(self.images)

    def head_labels(self) -> list[np.ndarray]:
        """Labels split per character position, the shape Keras wants for 5 heads."""
        return [self.labels[:, position] for position in range(CAPTCHA_LENGTH)]


def denoise(image: np.ndarray) -> np.ndarray:
    """Remove the strike-through line the captcha generator draws over the text.

    Blur to soften the one-pixel line, threshold to black and white, then dilate
    vertically and erode to reconnect the character strokes the line cut through.
    """
    image = cv2.blur(image, (3, 3))
    _, image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)
    image = cv2.dilate(image, np.ones((3, 1), np.uint8))
    image = cv2.erode(image, np.ones((2, 2), np.uint8))
    return image


def _sample_paths(samples_dir: Path) -> list[Path]:
    paths = [p for p in samples_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    if not paths:
        raise FileNotFoundError(f"no captcha images found in {samples_dir}")
    # Sort before shuffling so the shuffle depends only on the seed, not on the
    # order the filesystem happens to hand back.
    return sorted(paths)


def load_dataset(samples_dir: Path | str, seed: int = 0) -> tuple[Dataset, Vocabulary]:
    """Load, denoise and encode every sample under ``samples_dir``."""
    samples_dir = Path(samples_dir)
    paths = _sample_paths(samples_dir)
    random.Random(seed).shuffle(paths)

    texts = [p.stem for p in paths]
    wrong_length = {t for t in texts if len(t) != CAPTCHA_LENGTH}
    if wrong_length:
        raise ValueError(
            f"expected {CAPTCHA_LENGTH}-character filenames, got {sorted(wrong_length)[:5]}"
        )

    vocabulary = Vocabulary("".join(texts))

    images = np.zeros((len(paths), IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)
    labels = np.zeros((len(paths), CAPTCHA_LENGTH), dtype=np.int32)
    for i, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"could not read {path}")
        if image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError(f"{path} is {image.shape}, expected {(IMAGE_HEIGHT, IMAGE_WIDTH)}")
        images[i, :, :, 0] = denoise(image) / 255.0
        labels[i] = vocabulary.encode(texts[i])

    dataset = Dataset(
        images=images, labels=labels, texts=tuple(texts), paths=tuple(paths)
    )
    return dataset, vocabulary


def train_test_split(dataset: Dataset, train_fraction: float = 0.8) -> tuple[Dataset, Dataset]:
    """Split off the tail of an already-shuffled dataset as the test set."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    cut = int(len(dataset) * train_fraction)
    return (
        Dataset(
            dataset.images[:cut], dataset.labels[:cut], dataset.texts[:cut], dataset.paths[:cut]
        ),
        Dataset(
            dataset.images[cut:], dataset.labels[cut:], dataset.texts[cut:], dataset.paths[cut:]
        ),
    )
