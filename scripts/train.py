"""Train the MHI + 2D-CNN gesture classifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import load_dataset
from src.model import build_model
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MHI-CNN gesture classifier")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default="models/cnn_mhi_model.h5")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    gestures_map = cfg["gestures"]
    data_dir = args.data_dir or train_cfg["data_dir"]
    image_size = tuple(train_cfg["image_size"])

    x, y = load_dataset(data_dir, gestures_map, image_size=image_size)
    print(f"x_data shape: {x.shape}, y_data shape: {y.shape}")

    y_idx = np.argmax(y, axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"], stratify=y_idx,
    )
    X_tr, X_val, Y_tr, Y_val = train_test_split(
        X_train, Y_train, test_size=train_cfg["val_size"],
        random_state=train_cfg["random_state"], stratify=np.argmax(Y_train, axis=1),
    )

    input_shape = tuple(x.shape[1:]) if x.ndim == 4 else (image_size[1], image_size[0], 3)
    model = build_model(input_shape=input_shape,
        num_classes=len(gestures_map),
        learning_rate=train_cfg["learning_rate"],
    )
    model.summary()

    import keras

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", patience=3, verbose=1, factor=0.5, min_lr=5e-5),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ]
    history = model.fit(
        X_tr, Y_tr, batch_size=train_cfg["batch_size"], epochs=train_cfg["epochs"],
        validation_data=(X_val, Y_val), shuffle=True, callbacks=callbacks,
    )
    model.evaluate(X_test, Y_test)
    model.save(args.output)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_title("Model accuracy"); axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_title("Model loss"); axes[1].legend()
    plt.tight_layout()
    plt.savefig("models/training_curves.png")

    labels = list(gestures_map.keys())
    preds = np.argmax(model.predict(X_test), axis=1)
    truth = np.argmax(Y_test, axis=1)
    cm = confusion_matrix(truth, preds)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, ax=ax)
    plt.savefig("models/confusion_matrix.png")


if __name__ == "__main__":
    main()
