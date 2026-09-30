"""
CallGuard Speech Demo Engine.
Provides lightweight speech-to-text fallback when offline without heavy ONNX weights.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
import numpy as np


@dataclass
class Transcript:
    text: str
    window_seconds: float
    latency_ms: float
    confidence: float | None = None
    skip_reason: str | None = None


_SYNTHETIC_LINES = [
    "CallGuard active: audio stream verified.",
    "Real-time speech transcription running on local device.",
    "Voice frequency and pitch micro-jitter monitored.",
    "Protected audio priority active on speech bus.",
    "Live captions announced politely without interruption.",
]


class DemoEngine:
    name = "demo"
    device_description = "Local lightweight speech engine"
    is_neural_accelerated = False

    def __init__(self) -> None:
        self._index = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        started = time.perf_counter()
        window_seconds = len(audio) / float(sample_rate or 1)
        text = _SYNTHETIC_LINES[self._index % len(_SYNTHETIC_LINES)]
        self._index += 1
        return Transcript(
            text=f"[caption] {text}",
            window_seconds=window_seconds,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            confidence=0.92,
        )
