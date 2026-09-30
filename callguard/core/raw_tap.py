"""
RAW Tap Pipeline: Biometric Hemodynamic Liveness & Voice Authenticity Detection.

Architectural Design:
    Enforces the architectural constraint that deepfake detection and voice authenticity
    analysis are performed STRICTLY on pristine, unenhanced video frames and raw audio chunks
    tapped directly from the input ring buffers, never on processed or enhanced display frames.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# Native CallGuard biometrics modules
from callguard.biometrics.rppg_engine import RPPGEngine, RPPGMetrics
from callguard.biometrics.audio_detector import AudioSyntheticDetector, AudioAuthenticityMetrics
from callguard.biometrics.face_tracker import FaceMeshTracker, FaceTrackingResult

from callguard.core.evidence import Evidence


class RawTapPipeline:
    """
    Executes biometric hemodynamic liveness and voice clone detection on RAW captured media.
    Guaranteed to run before and completely independent of any enhancement filter.
    """

    def __init__(
        self,
        fps: float = 30.0,
        sample_rate: int = 16000,
        min_bpm: float = 45.0,
        max_bpm: float = 180.0,
        snr_threshold_db: float = 1.5,
    ):
        self.fps = fps
        self.sample_rate = sample_rate
        self.rppg_engine = RPPGEngine(
            sampling_rate_fps=fps,
            buffer_window_seconds=3.0,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            liveness_snr_threshold_db=snr_threshold_db,
        )
        self.audio_detector = AudioSyntheticDetector(
            sample_rate=sample_rate,
            frame_length_sec=1.0,
        )
        self.tracker = FaceMeshTracker()

    def process_raw(
        self,
        raw_video_frame: Optional[np.ndarray],
        raw_audio_chunk: Optional[np.ndarray],
        t_start: float,
        t_end: float,
        ctx: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Evidence], Dict[str, Any]]:
        """
        Analyze pristine raw frame and audio.
        Returns:
            - List of normalized Evidence objects
            - Raw metrics dictionary for telemetry HUD
        """
        evidences: List[Evidence] = []
        metrics: Dict[str, Any] = {
            "face_detected": False,
            "heart_rate_bpm": 0.0,
            "pulse_snr_db": 0.0,
            "pulse_confidence": 0.0,
            "is_biologically_authentic": True,
            "voice_authenticity_score": 1.0,
            "spectral_flatness": 0.0,
            "high_freq_ratio": 0.0,
            "pitch_jitter_score": 0.0,
        }

        context = dict(ctx or {
            "low_light": False,
            "bitrate_drop": False,
            "stutter": False,
            "multi_face": False,
        })

        # --- 1. Raw Video Hemodynamic Analysis ---
        if raw_video_frame is not None and raw_video_frame.size > 0:
            track_res: Optional[FaceTrackingResult] = self.tracker.process_frame(raw_video_frame)
            if track_res is not None and track_res.composite_skin_roi is not None:
                metrics["face_detected"] = True
                rppg_res: Optional[RPPGMetrics] = self.rppg_engine.update_frame_roi(track_res.composite_skin_roi)
                if rppg_res is not None:
                    metrics["heart_rate_bpm"] = float(rppg_res.heart_rate_bpm)
                    metrics["pulse_snr_db"] = float(rppg_res.snr_db)
                    metrics["pulse_confidence"] = float(rppg_res.pulse_confidence)
                    metrics["is_biologically_authentic"] = bool(rppg_res.is_biologically_authentic)

                    evidences.append(Evidence(
                        source="raw_tap.rppg",
                        signal="pulse_snr_db",
                        value=float(rppg_res.snr_db),
                        confidence=float(rppg_res.pulse_confidence),
                        t_start=t_start,
                        t_end=t_end,
                        ctx=context,
                    ))
            else:
                # If face not tracked, or skin ROI absent
                # If unprimed, push fallback neutral evidence if low light
                pass

        # --- 2. Raw Audio Forensic Acoustic Analysis ---
        if raw_audio_chunk is not None and len(raw_audio_chunk) >= 512:
            audio_res: AudioAuthenticityMetrics = self.audio_detector.analyze_audio_buffer(raw_audio_chunk)
            metrics["voice_authenticity_score"] = float(audio_res.voice_authenticity_score)
            metrics["spectral_flatness"] = float(audio_res.spectral_flatness)
            metrics["high_freq_ratio"] = float(audio_res.high_freq_ratio)
            metrics["pitch_jitter_score"] = float(audio_res.pitch_jitter_score)

            conf = 0.85 if len(raw_audio_chunk) >= 8000 else 0.65

            evidences.append(Evidence(
                source="raw_tap.audio_flatness",
                signal="spectral_flatness",
                value=float(audio_res.spectral_flatness),
                confidence=conf,
                t_start=t_start,
                t_end=t_end,
                ctx=context,
            ))

            evidences.append(Evidence(
                source="raw_tap.audio_cutoff",
                signal="high_freq_ratio",
                value=float(audio_res.high_freq_ratio),
                confidence=conf,
                t_start=t_start,
                t_end=t_end,
                ctx=context,
            ))

            evidences.append(Evidence(
                source="raw_tap.audio_jitter",
                signal="pitch_jitter",
                value=float(audio_res.pitch_jitter_score),
                confidence=conf,
                t_start=t_start,
                t_end=t_end,
                ctx=context,
            ))

        return evidences, metrics
