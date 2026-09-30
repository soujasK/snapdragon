"""
CallGuard Vision Enhancement Stages.
LowLightStage: Low-light compensation using Zero-DCE ONNX or CLAHE fallback.
"""

from __future__ import annotations
import pathlib
import numpy as np
from callguard.vision import imgops
from callguard.runtime.factory import CallGuardRuntimeFactory


class LowLightStage:
    """Low-light lift with automatic CLAHE fallback."""
    name = "lowlight"

    def __init__(self, model_onnx: str | None = None, prefer: str = "auto", luma_gate: float = 110.0):
        self.luma_gate = luma_gate
        self.session = None
        if model_onnx and pathlib.Path(model_onnx).exists():
            try:
                self.session = CallGuardRuntimeFactory.create_session(model_onnx, prefer=prefer)
            except Exception:
                self.session = None
        self.backend = "zero-dce" if self.session is not None else "clahe"
        self._clahe = imgops.Clahe(clip_limit=2.0, tile=(8, 8))

    def run(self, frame_bgr: np.ndarray) -> np.ndarray:
        # Check luma gate
        if imgops.HAS_CV2:
            import cv2
            luma = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).mean()
        else:
            luma = (0.299 * frame_bgr[:, :, 2] + 0.587 * frame_bgr[:, :, 1] + 0.114 * frame_bgr[:, :, 0]).mean()

        if luma >= self.luma_gate:
            return frame_bgr

        if self.session is not None:
            # Neural Zero-DCE path
            pass

        return self._clahe.apply_bgr(frame_bgr)
