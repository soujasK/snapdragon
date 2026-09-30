"""
CallGuard NPU: Remote Photoplethysmography (rPPG) Biometric Engine.

Extracts sub-dermal capillary hemoglobin blood-volume pulse (BVP) signals
from facial region-of-interest (ROI) video frames using the Plane-Orthogonal-to-Skin (POS)
and Chrominance (CHROM) algorithms.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy import signal


@dataclass
class RPPGMetrics:
    """Telemetry metrics extracted from facial rPPG analysis."""
    heart_rate_bpm: float
    snr_db: float
    pulse_confidence: float  # [0.0, 1.0]
    is_biologically_authentic: bool
    bvp_signal: np.ndarray  # Filtered temporal pulse signal
    psd_frequencies: np.ndarray
    psd_power: np.ndarray
    inter_beat_intervals_ms: List[float]


class RPPGEngine:
    """
    Real-Time Physiological Blood Volume Pulse (BVP) Extractor and Liveness Sentinel.
    Optimized for execution on low-power edge accelerators (Qualcomm Hexagon NPU / CPU fallback).
    """

    def __init__(
        self,
        sampling_rate_fps: float = 30.0,
        buffer_window_seconds: float = 5.0,
        min_bpm: float = 45.0,
        max_bpm: float = 160.0,
        liveness_snr_threshold_db: float = 1.5,
    ) -> None:
        self.fps = sampling_rate_fps
        self.buffer_size = int(sampling_rate_fps * buffer_window_seconds)
        self.min_freq = min_bpm / 60.0  # ~0.75 Hz
        self.max_freq = max_bpm / 60.0  # ~2.67 Hz
        self.liveness_snr_threshold = liveness_snr_threshold_db

        # Rolling buffers for spatial average RGB skin reflections
        self.r_buffer: List[float] = []
        self.g_buffer: List[float] = []
        self.b_buffer: List[float] = []

        # Pre-compute Butterworth bandpass filter coefficients
        nyquist = 0.5 * self.fps
        low = self.min_freq / nyquist
        high = min(self.max_freq / nyquist, 0.99)
        self.b_filter, self.a_filter = signal.butter(3, [low, high], btype="bandpass")

    def reset(self) -> None:
        """Reset historical temporal buffers."""
        self.r_buffer.clear()
        self.g_buffer.clear()
        self.b_buffer.clear()

    def update_frame_roi(self, skin_roi_bgr: np.ndarray) -> Optional[RPPGMetrics]:
        """
        Process a single video frame's facial skin ROI.
        Extracts spatial average RGB, updates rolling temporal buffer,
        and computes rPPG metrics when buffer is primed.
        """
        if skin_roi_bgr is None or skin_roi_bgr.size == 0:
            return None

        # Calculate spatial mean of R, G, B channels across skin pixels
        mean_b = float(np.mean(skin_roi_bgr[:, :, 0]))
        mean_g = float(np.mean(skin_roi_bgr[:, :, 1]))
        mean_r = float(np.mean(skin_roi_bgr[:, :, 2]))

        self.r_buffer.append(mean_r)
        self.g_buffer.append(mean_g)
        self.b_buffer.append(mean_b)

        if len(self.r_buffer) > self.buffer_size:
            self.r_buffer.pop(0)
            self.g_buffer.pop(0)
            self.b_buffer.pop(0)

        if len(self.r_buffer) < int(self.fps * 2.0):
            # Need at least 2.0 seconds of contiguous frames for frequency resolution
            return None

        return self._compute_liveness_metrics()

    def _compute_liveness_metrics(self) -> RPPGMetrics:
        """Applies POS projection, temporal filtering, FFT, and SNR estimation."""
        r = np.array(self.r_buffer, dtype=np.float64)
        g = np.array(self.g_buffer, dtype=np.float64)
        b = np.array(self.b_buffer, dtype=np.float64)

        # 1. Temporal normalization against baseline illumination
        eps = 1e-6
        mean_r = np.mean(r) + eps
        mean_g = np.mean(g) + eps
        mean_b = np.mean(b) + eps

        r_norm = r / mean_r
        g_norm = g / mean_g
        b_norm = b / mean_b

        # 2. Plane-Orthogonal-to-Skin (POS) projection
        # S1 = G - B, S2 = G + B - 2R
        s1 = g_norm - b_norm
        s2 = g_norm + b_norm - (2.0 * r_norm)

        std_s1 = np.std(s1) + eps
        std_s2 = np.std(s2) + eps
        alpha = std_s1 / std_s2
        raw_bvp = s1 + (alpha * s2)

        # 3. Detrending and bandpass filtering
        detrended_bvp = signal.detrend(raw_bvp)
        filtered_bvp = signal.filtfilt(self.b_filter, self.a_filter, detrended_bvp)

        # Normalize signal to [-1.0, 1.0] range
        max_abs = np.max(np.abs(filtered_bvp)) + eps
        normalized_bvp = filtered_bvp / max_abs

        # 4. Power Spectral Density via Welch / FFT
        n_samples = len(filtered_bvp)
        n_fft = max(512, 1 << (n_samples - 1).bit_length())
        freqs, psd = signal.welch(filtered_bvp, fs=self.fps, nperseg=min(n_samples, 256), nfft=n_fft)

        # Mask frequencies strictly within physiological heart rate range (45 - 160 BPM)
        valid_mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        valid_freqs = freqs[valid_mask]
        valid_psd = psd[valid_mask]

        if len(valid_psd) == 0:
            return RPPGMetrics(
                heart_rate_bpm=0.0,
                snr_db=-10.0,
                pulse_confidence=0.0,
                is_biologically_authentic=False,
                bvp_signal=normalized_bvp,
                psd_frequencies=freqs,
                psd_power=psd,
                inter_beat_intervals_ms=[],
            )

        # Peak frequency corresponds to the fundamental cardiac pulse
        peak_idx = int(np.argmax(valid_psd))
        f_peak = float(valid_freqs[peak_idx])
        heart_rate_bpm = float(np.round(f_peak * 60.0, 1))

        # 5. Signal-to-Noise Ratio (SNR) Calculation
        delta_f = 0.15
        signal_band = (valid_freqs >= f_peak - delta_f) & (valid_freqs <= f_peak + delta_f)
        p_signal = np.sum(valid_psd[signal_band])
        p_noise = np.sum(valid_psd[~signal_band]) + eps

        snr_linear = p_signal / p_noise
        snr_db = float(10.0 * np.log10(max(snr_linear, 1e-4)))

        # 6. Peak-to-peak interval regularity (cardiac rhythmicity)
        peaks, _ = signal.find_peaks(normalized_bvp, distance=int(self.fps / 2.5), prominence=0.3)
        ibi_ms: List[float] = []
        if len(peaks) > 1:
            ibi_ms = [(float(peaks[i] - peaks[i - 1]) / self.fps) * 1000.0 for i in range(1, len(peaks))]

        confidence = float(np.clip((snr_db - self.liveness_snr_threshold) / 6.0 + 0.5, 0.0, 1.0))
        is_authentic = bool(snr_db >= self.liveness_snr_threshold and 48.0 <= heart_rate_bpm <= 150.0)

        return RPPGMetrics(
            heart_rate_bpm=heart_rate_bpm if (is_authentic or snr_db > 0.0) else 0.0,
            snr_db=round(snr_db, 2),
            pulse_confidence=round(confidence, 3),
            is_biologically_authentic=is_authentic,
            bvp_signal=normalized_bvp,
            psd_frequencies=valid_freqs,
            psd_power=valid_psd,
            inter_beat_intervals_ms=ibi_ms,
        )
