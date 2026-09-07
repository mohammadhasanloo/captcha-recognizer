"""The multi-head CNN.

One convolutional trunk reads the whole 200x50 image; five independent heads each
predict the character at one position. Positions are learned jointly, so the trunk
never has to be told where one character ends and the next begins.
"""

from __future__ import annotations

import keras
from keras import layers

from captcha_recognizer.data import CAPTCHA_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH

TRUNK_FILTERS = (16, 32, 64, 128, 256)


def _trunk(inputs: keras.KerasTensor) -> keras.KerasTensor:
    x = inputs
    for depth, filters in enumerate(TRUNK_FILTERS):
        # A wider receptive field on the first block, where the strike-through
        # line and character strokes are still the same thickness.
        kernel_size = 5 if depth == 0 else 3
        x = layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            activation="relu",
            name=f"conv{depth + 1}",
        )(x)
        x = layers.MaxPool2D(pool_size=(2, 2), padding="same", name=f"pool{depth + 1}")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(0.2, name="trunk_dropout")(x)
    return layers.BatchNormalization(name="trunk_norm")(x)


def _head(features: keras.KerasTensor, num_classes: int, position: int) -> keras.KerasTensor:
    x = layers.Dense(64, activation="relu", name=f"char{position + 1}_dense")(features)
    x = layers.Dropout(0.2, name=f"char{position + 1}_dropout")(x)
    x = layers.BatchNormalization(name=f"char{position + 1}_norm")(x)
    return layers.Dense(num_classes, activation="softmax", name=f"char{position + 1}")(x)


def build_model(num_classes: int, learning_rate: float = 1e-3) -> keras.Model:
    """Compile the trunk-and-five-heads model.

    Labels are integer class indices, so sparse categorical cross-entropy reads
    them directly and no one-hot expansion is needed.
    """
    inputs = keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name="captcha")
    features = _trunk(inputs)
    outputs = [_head(features, num_classes, position) for position in range(CAPTCHA_LENGTH)]

    model = keras.Model(inputs=inputs, outputs=outputs, name="captcha_recognizer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        # Keras 3 needs one metric entry per output; a bare ["accuracy"] is
        # rejected for a multi-head model.
        metrics=[["accuracy"] for _ in range(CAPTCHA_LENGTH)],
    )
    return model
