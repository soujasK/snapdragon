"""
CallGuard Evidence Contract.

Architectural Design:
    Defines the canonical, normalized Evidence data structure that encapsulates raw biometric,
    acoustic, and forensic detection metrics along with environmental context telemetry.
    Links quantitative signal values and detection confidences to contextual flags
    (low_light, bitrate_drop, stutter, multi_face) for adversarial verification.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Evidence:
    """
    Normalized multi-modal evidence contract linking raw biometric/acoustic observations
    to environmental and channel quality context flags.
    """
    source: str               # e.g., "raw_tap.rppg", "raw_tap.audio_flatness", "raw_tap.pitch_jitter"
    signal: str               # e.g., "pulse_snr_db", "spectral_flatness", "jitter_variance"
    value: float              # Measured numerical signal value
    confidence: float         # Confidence in measurement [0.0, 1.0]
    t_start: float            # Monotonic window start timestamp in seconds
    t_end: float              # Monotonic window end timestamp in seconds
    ctx: Dict[str, Any] = field(default_factory=lambda: {
        "low_light": False,
        "bitrate_drop": False,
        "stutter": False,
        "multi_face": False,
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "signal": self.signal,
            "value": round(float(self.value), 4),
            "confidence": round(float(self.confidence), 4),
            "t_start": round(float(self.t_start), 3),
            "t_end": round(float(self.t_end), 3),
            "ctx": dict(self.ctx),
        }
