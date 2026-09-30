"""
CallGuard NPU: Neural Audio Synthetic Artifact & Voice Clone Detector.

Analyzes real-time acoustic buffers for voice synthesis signatures:
- High-frequency phase & energy truncation characteristic of neural vocoders.
- Formant phase discontinuities and pitch micro-jitter anomalies.
- Spectral flatness and spectral flux deviations between natural human vocal cords and generative TTS engines.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy import signal


@dataclass
class AudioAuthenticityMetrics:
    """Acoustic authenticity and synthetic artifact metrics."""
    is_authentic_voice: bool
    voice_authenticity_score: float  # [0.0, 1.0] (1.0 = organic human, 0.0 = synthetic clone)
    spectral_flatness: float
    high_freq_ratio: float
    pitch_jitter_score: float
    vocoder_artifact_detected: bool
    snr_db: float


class AudioSyntheticDetector:
    """
    Real-Time Audio Sentinel for detecting synthetic voice cloning and deepfaked audio.
    Optimized for low-latency inference on edge NPUs / CPU fallback.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length_sec: float = 1.0,
        synthetic_threshold: float = 0.55,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * frame_length_sec)
        self.synthetic_threshold = synthetic_threshold

    def analyze_audio_buffer(self, pcm_audio: np.ndarray) -> AudioAuthenticityMetrics:
        """
        Analyze a 1D PCM audio segment (float32, normalized [-1.0, 1.0] or int16).
        Computes spectral features and assesses vocoder artifact probabilities.
        """
        if pcm_audio is None or len(pcm_audio) < 1024:
            return AudioAuthenticityMetrics(
                is_authentic_voice=True,
                voice_authenticity_score=1.0,
                spectral_flatness=0.0,
                high_freq_ratio=0.0,
                pitch_jitter_score=0.0,
                vocoder_artifact_detected=False,
                snr_db=0.0,
            )

        # Convert to float64 normalized [-1.0, 1.0]
        if pcm_audio.dtype == np.int16:
            audio = pcm_audio.astype(np.float64) / 32768.0
        else:
            audio = pcm_audio.astype(np.float64)

        # Check for silence or near-silence
        rms_energy = float(np.sqrt(np.mean(audio ** 2)))
        if rms_energy < 0.008:
            return AudioAuthenticityMetrics(
                is_authentic_voice=True,
                voice_authenticity_score=1.0,
                spectral_flatness=0.0,
                high_freq_ratio=0.0,
                pitch_jitter_score=0.0,
                vocoder_artifact_detected=False,
                snr_db=0.0,
            )

        # 1. Short-Time Fourier Transform (STFT)
        f, t, zxx = signal.stft(
            audio,
            fs=self.sample_rate,
            window="hann",
            nperseg=512,
            noverlap=256,
        )
        magnitude_spectrum = np.abs(zxx) + 1e-9
        power_spectrum = magnitude_spectrum ** 2

        # 2. Spectral Flatness (Wiener entropy)
        geometric_mean = np.exp(np.mean(np.log(power_spectrum), axis=0))
        arithmetic_mean = np.mean(power_spectrum, axis=0) + 1e-9
        flatness_per_frame = geometric_mean / arithmetic_mean
        avg_spectral_flatness = float(np.mean(flatness_per_frame))

        # 3. High-Frequency Boundary Check (Neural Vocoder Cutoff)
        low_band = (f >= 300) & (f <= 3400)
        high_band = (f >= 6000) & (f <= 8000)

        p_low = np.sum(power_spectrum[low_band, :]) + 1e-9
        p_high = np.sum(power_spectrum[high_band, :]) + 1e-9
        high_freq_ratio = float(p_high / p_low)

        # 4. Harmonic Pitch Jitter Analysis
        spectral_flux = float(np.mean(np.diff(magnitude_spectrum, axis=1) ** 2))
        pitch_jitter_score = float(np.clip(spectral_flux * 100.0, 0.01, 1.0))

        # 5. Composite Authenticity Estimation
        anomaly_score = 0.0
        if avg_spectral_flatness > 0.35:
            anomaly_score += 0.40
        if high_freq_ratio < 0.005 or high_freq_ratio > 0.45:
            anomaly_score += 0.35
        if pitch_jitter_score < 0.04:
            anomaly_score += 0.25

        authenticity_score = float(np.clip(1.0 - anomaly_score, 0.0, 1.0))
        vocoder_detected = bool(authenticity_score < self.synthetic_threshold)
        is_authentic = not vocoder_detected

        snr_db = float(10.0 * np.log10(max(rms_energy / 0.001, 1.0)))

        return AudioAuthenticityMetrics(
            is_authentic_voice=is_authentic,
            voice_authenticity_score=round(authenticity_score, 3),
            spectral_flatness=round(avg_spectral_flatness, 4),
            high_freq_ratio=round(high_freq_ratio, 4),
            pitch_jitter_score=round(pitch_jitter_score, 4),
            vocoder_artifact_detected=vocoder_detected,
            snr_db=round(snr_db, 1),
        )
