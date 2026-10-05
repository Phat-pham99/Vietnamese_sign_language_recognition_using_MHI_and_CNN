"""Profiling and visualization helpers."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2 as cv
import numpy as np
import yaml


def load_config(path: str | Path = "configs/default_config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class FPSMeter:
    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


def draw_grid(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    out = frame.copy()
    for x in (width // 4, width // 2, 3 * width // 4):
        cv.line(out, (x, 0), (x, height), (255, 0, 0), 2)
    for y in (height // 4, height // 2, 3 * height // 4):
        cv.line(out, (0, y), (width, y), (255, 0, 0), 2)
    return out


def overlay_info(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame.copy()
    for i, text in enumerate(lines):
        cv.putText(
            out, text, (12, 30 + i * 25), cv.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 0, 0), 2, lineType=cv.LINE_AA,
        )
    return out
