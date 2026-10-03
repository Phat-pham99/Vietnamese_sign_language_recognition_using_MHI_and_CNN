"""Lightweight 2D-CNN architecture definition."""

from __future__ import annotations

from keras import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from keras.optimizers import RMSprop


def build_model(
    input_shape: tuple[int, int, int] = (160, 160, 3),
    num_classes: int = 6,
    learning_rate: float = 2e-4,
) -> Sequential:
    model = Sequential(
        [
            Conv2D(16, (19, 19), padding="valid", activation="relu", input_shape=input_shape),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(16, (11, 11), padding="valid", activation="relu"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(32, (7, 7), padding="valid", activation="relu"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(64, (3, 3), padding="valid", activation="relu"),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            Flatten(),
            Dense(1066, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    optimizer = RMSprop(learning_rate=learning_rate, rho=0.9, epsilon=1e-8)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
