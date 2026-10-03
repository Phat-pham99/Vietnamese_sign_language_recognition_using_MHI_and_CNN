"""Dataset loading and motion masking utilities."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

GESTURE_PREFIXES = {
    "uo": "uong",
    "do": "doi",
    "vu": "vui",
    "gi": "gian",
    "to": "toi",
    "no": "none",
}

NUM_CLASSES = 6


def process_image(path: str | os.PathLike, image_size: tuple[int, int] = (160, 160)) -> np.ndarray:
    img = Image.open(path)
    img = img.resize(image_size)
    return np.asarray(img)


def walk_file_tree(
    root: str | os.PathLike,
    gestures_map: dict[str, int],
    image_size: tuple[int, int] = (160, 160),
) -> tuple[np.ndarray, np.ndarray]:
    x_data: list[np.ndarray] = []
    y_data: list[int] = []
    for directory, _, files in os.walk(root):
        for file in sorted(files):
            if file.startswith(".") or file.startswith("C_"):
                continue
            gesture_name = GESTURE_PREFIXES.get(file[0:2])
            if gesture_name is None or gesture_name not in gestures_map:
                continue
            path = Path(directory) / file
            y_data.append(gestures_map[gesture_name])
            x_data.append(process_image(path, image_size))
    if not x_data:
        raise FileNotFoundError(f"No usable MHI images found under {root}")
    x = np.array(x_data, dtype="float32") / 255.0
    y = np.array(y_data)
    return x, y


def load_dataset(
    root: str | os.PathLike,
    gestures_map: dict[str, int],
    image_size: tuple[int, int] = (160, 160),
    num_classes: int = NUM_CLASSES,
) -> tuple[np.ndarray, np.ndarray]:
    from keras.utils import to_categorical

    x, y = walk_file_tree(root, gestures_map, image_size)
    return x, to_categorical(y, num_classes=num_classes)
