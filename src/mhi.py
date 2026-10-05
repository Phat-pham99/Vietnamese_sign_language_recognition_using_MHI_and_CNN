"""Motion History Image (MHI) transformation engine."""

from __future__ import annotations

import cv2 as cv
import numpy as np


class MHITransformer:
    """Accumulates frame-difference motion into a decaying motion history image."""

    def __init__(self, duration: float = 5.0, threshold: int = 50) -> None:
        self.duration = float(duration)
        self.threshold = int(threshold)
        self.motion_history: np.ndarray | None = None
        self._prev_frame: np.ndarray | None = None

    def reset(self) -> None:
        self.motion_history = None
        self._prev_frame = None

    def motion_mask(self, frame: np.ndarray) -> np.ndarray | None:
        """Binary mask Psi(x, y, t) = 1 where |I(t) - I(t-1)| > xi."""
        if self._prev_frame is None:
            self._prev_frame = frame.copy()
            return None
        diff = cv.absdiff(frame, self._prev_frame)
        gray_diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(gray_diff, self.threshold, 1, cv.THRESH_BINARY)
        self._prev_frame = frame.copy()
        return mask

    def update(self, frame: np.ndarray, timestamp: float) -> np.ndarray:
        """Update internal MHI buffer with a new frame and return the rendered MHI."""
        if self.motion_history is None:
            h, w = frame.shape[:2]
            self.motion_history = np.zeros((h, w), np.float32)
        mask = self.motion_mask(frame)
        if mask is not None:
            cv.motempl.updateMotionHistory(mask, self.motion_history, timestamp, self.duration)
        return self.render(timestamp)

    def render(self, timestamp: float) -> np.ndarray:
        """Convert the internal float MHI buffer into a uint8 BGR image."""
        if self.motion_history is None:
            raise RuntimeError("MHITransformer has not received any frames yet.")
        vis = np.uint8(
            np.clip((self.motion_history - (timestamp - self.duration)) / self.duration, 0, 1) * 255
        )
        return cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
