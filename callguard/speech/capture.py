"""
CallGuard Audio Capture & VAD Energy Utilities.
Provides RMS calculation, silence detection, and live audio capture.
"""

from __future__ import annotations
import numpy as np


def rms(window: np.ndarray) -> float:
    """Computes Root-Mean-Square energy of an audio signal."""
    if window.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))


def is_silent(window: np.ndarray, threshold: float = 1e-3) -> bool:
    """Returns True if the window RMS is below the silence threshold."""
    return rms(window) < threshold


def open_stream(device: int | None = None, sample_rate: int = 16000, block_ms: int = 50):
    """Opens a mono 16 kHz input stream via sounddevice if available."""
    try:
        import sounddevice as sd
        return sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=int(sample_rate * block_ms / 1000),
            dtype="float32",
            channels=1,
            device=device,
        )
    except Exception as exc:
        raise RuntimeError(f"Audio device open failed: {exc}")
