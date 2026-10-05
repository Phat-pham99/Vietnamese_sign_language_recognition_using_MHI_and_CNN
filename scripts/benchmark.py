"""Latency and memory footprint profiler for the MHI + CNN pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mhi import MHITransformer
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark preprocessing and inference latency")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--frames", type=int, default=200)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mhi_cfg = cfg["mhi"]

    from keras.models import load_model

    model = load_model(args.model or cfg["model"]["weights_path"])
    transformer = MHITransformer(duration=mhi_cfg["duration"], threshold=mhi_cfg["threshold"])

    width, height = cfg["camera"]["width"], cfg["camera"]["height"]
    in_shape = tuple(cfg["model"]["input_shape"])

    mhi_times, infer_times = [], []
    for i in range(args.frames):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        mhi = transformer.update(frame, time.perf_counter())
        mhi_times.append((time.perf_counter() - t0) * 1e3)

        inp = cv.resize(mhi, (in_shape[1], in_shape[0])).astype("float32") / 255.0
        inp = inp.reshape(1, *in_shape)
        t1 = time.perf_counter()
        model.predict(inp, verbose=0)
        infer_times.append((time.perf_counter() - t1) * 1e3)

    print(f"Frames benchmarked : {args.frames}")
    print(f"MHI preprocessing  : {np.mean(mhi_times):.2f} ms/frame")
    print(f"Model inference    : {np.mean(infer_times):.2f} ms/frame")
    print(f"End-to-end FPS     : {1000.0 / (np.mean(mhi_times) + np.mean(infer_times)):.1f}")


if __name__ == "__main__":
    main()
