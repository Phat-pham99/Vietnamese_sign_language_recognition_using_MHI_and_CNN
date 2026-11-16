"""Motion History Image (MHI) transformation engine."""

from __future__ import annotations

import cv2 as cv
import numpy as np


class MHITransformer:
    """Frame-decay Motion History Image (Bobick & Davis).

    H(x, y, t) = tau                         if Psi(x, y, t) = 1
               = max(0, H(x, y, t - 1) - 1)  otherwise

    with Psi(x, y, t) = |I(t) - I(t-1)| > xi. ``tau`` is measured in frames.
    """

    def __init__(self, tau: int = 150, threshold: int = 50) -> None:
        if not 1 <= tau <= 255:
            raise ValueError(f"tau must be in [1, 255] frames, got {tau}")
        self.tau = int(tau)
        self.threshold = int(threshold)
        self.motion_history: np.ndarray | None = None
        self._prev_frame: np.ndarray | None = None

    def reset(self) -> None:
        self.motion_history = None
        self._prev_frame = None

    def motion_mask(self, frame: np.ndarray) -> np.ndarray | None:
        """Boolean mask Psi(x, y, t); ``None`` for the first frame."""
        if self._prev_frame is None:
            self._prev_frame = frame.copy()
            return None
        gray_diff = cv.cvtColor(cv.absdiff(frame, self._prev_frame), cv.COLOR_BGR2GRAY)
        self._prev_frame = frame.copy()
        return gray_diff > self.threshold

    def update(self, frame: np.ndarray) -> np.ndarray:
        """Advance the MHI by one frame and return the rendered 3-channel uint8 image."""
        if self.motion_history is None:
            self.motion_history = np.zeros(frame.shape[:2], np.uint8)
        mask = self.motion_mask(frame)
        # Saturating uint8 subtraction implements max(0, H - 1).
        cv.subtract(self.motion_history, 1, dst=self.motion_history)
        if mask is not None:
            self.motion_history[mask] = self.tau
        return self.render()

    def render(self) -> np.ndarray:
        """Scale H to I = H / tau * 255, replicated to 3 channels for the CNN input."""
        if self.motion_history is None:
            raise RuntimeError("MHITransformer has not received any frames yet.")
        vis = cv.convertScaleAbs(self.motion_history, alpha=255.0 / self.tau)
        return cv.cvtColor(vis, cv.COLOR_GRAY2BGR)
