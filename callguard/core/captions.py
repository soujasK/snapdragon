"""
Live Speech Captions Pipeline: VAD-Gated Whisper / ASR Pipeline.

Architectural Design:
    Provides live speech-to-text captions on the captured audio bus, gated by Voice Activity
    Detection (VAD) / RMS energy to avoid running expensive Whisper ASR inference during silent
    pauses, thereby preserving NPU and CPU duty-cycle budgets.
"""

import time
from typing import Optional, Tuple
import numpy as np

# Native CallGuard speech modules
from callguard.speech.capture import rms, is_silent
from callguard.speech.demo_engine import DemoEngine

WhisperOnnxEngine = None


class CaptionsPipeline:
    """
    VAD-gated speech transcription engine interfacing with CallGuard audio stream.
    Prioritizes real Whisper ONNX inference, gracefully falling back to DemoEngine.
    """

    def __init__(self, sample_rate: int = 16000, vad_silence_threshold: float = 1e-3, prefer_demo: bool = False):
        self.sample_rate = sample_rate
        self.vad_silence_threshold = vad_silence_threshold
        self.engine = None
        self.engine_name = "demo"

        # Attempt to load Whisper ONNX if available and not explicitly preferring demo
        if not prefer_demo and WhisperOnnxEngine is not None:
            try:
                self.engine = WhisperOnnxEngine()
                self.engine_name = "whisper_onnx"
            except Exception as exc:
                # Transparent fallback
                self.engine = DemoEngine()
                self.engine_name = "demo_fallback"
        else:
            self.engine = DemoEngine()
            self.engine_name = "demo"

        self.last_caption: str = ""
        self.last_latency_ms: float = 0.0

    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Tuple[Optional[str], float, bool]:
        """
        Processes an audio chunk.
        Returns:
            - caption text (or None if gated by VAD silence)
            - latency in ms
            - was_vad_active (True if speech detected, False if silenced)
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return None, 0.0, False

        # VAD Gate: Compute RMS energy using callguard.speech.capture.is_silent
        if is_silent(audio_chunk, threshold=self.vad_silence_threshold):
            return None, 0.0, False

        t0 = time.perf_counter()
        try:
            transcript = self.engine.transcribe(audio_chunk, sample_rate=self.sample_rate)
            text = transcript.text
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.last_caption = text
            self.last_latency_ms = latency_ms
            return text, latency_ms, True
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return f"[Caption Error: {e}]", latency_ms, True
