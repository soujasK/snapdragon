"""
CallGuard Master Orchestration Pipeline.

Architectural Design:
    Provides the central orchestrator that wires together all subsystems:
    Timestamped Ring Buffers, RAW Tap (deepfake/voice detection), Enhance Path (display-only),
    VAD-gated Captions, Claim Adapter, Adversarial Debate Engine, Hysteresis Filter,
    Cryptographic Audit DB, Priority NPU Scheduler, and Overlay HUD.
"""

import time
import os
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

from callguard.config import CallGuardConfig, load_config
from callguard.capture.ring_buffer import TimestampedRingBuffer, TimestampedSample
from callguard.capture.sources import CaptureSource, FileCaptureSource, SyntheticCaptureSource, LiveCaptureSource
from callguard.core.evidence import Evidence
from callguard.core.claim_adapter import Claim, ClaimAdapter
from callguard.core.raw_tap import RawTapPipeline
from callguard.core.enhance_path import EnhancePathPipeline
from callguard.core.captions import CaptionsPipeline
from callguard.core.debate import CallGuardDebateEngine, DebateResult
from callguard.core.hysteresis import HysteresisFilter, HysteresisState
from callguard.core.audit import CallGuardAuditDB
from callguard.runtime.factory import CallGuardRuntimeFactory
from callguard.runtime.scheduler import PriorityScheduler
from callguard.runtime.telemetry import TelemetryTracker
from callguard.ui.overlay import CallGuardOverlayHUD


class CallGuardPipeline:
    """
    Unified CallGuard NPU Orchestrator.
    Executes the end-to-end multi-modal verification loop.
    """

    def __init__(self, config: Optional[CallGuardConfig] = None, source: Optional[CaptureSource] = None):
        self.config = config or load_config()

        # Hardware & runtime detection
        hw_info = CallGuardRuntimeFactory.get_hardware_info()
        self.hardware_label = hw_info["default_backend"]

        # Ring buffers with drop-oldest policy and monotonic clock
        self.video_buffer: TimestampedRingBuffer[np.ndarray] = TimestampedRingBuffer(
            capacity=self.config.capture.ring_buffer_video_capacity,
            name="VideoRingBuffer"
        )
        self.audio_buffer: TimestampedRingBuffer[np.ndarray] = TimestampedRingBuffer(
            capacity=self.config.capture.ring_buffer_audio_capacity,
            name="AudioRingBuffer"
        )

        # Ingest capture source
        self.source = source or SyntheticCaptureSource(fps=self.config.capture.video_fps)

        # Core pipelines
        self.enhance_pipeline = EnhancePathPipeline(target_fps=self.config.capture.video_fps)
        self.raw_tap = RawTapPipeline(
            fps=self.config.capture.video_fps,
            sample_rate=self.config.capture.audio_sample_rate,
            min_bpm=self.config.detection.rppg_min_bpm,
            max_bpm=self.config.detection.rppg_max_bpm,
            snr_threshold_db=self.config.detection.rppg_snr_threshold_db,
        )
        self.captions_pipeline = CaptionsPipeline(sample_rate=self.config.capture.audio_sample_rate)
        self.claim_adapter = ClaimAdapter(
            rppg_snr_threshold=self.config.detection.rppg_snr_threshold_db,
            audio_flatness_threshold=self.config.detection.audio_flatness_threshold,
            high_freq_cutoff_ratio=self.config.detection.audio_high_freq_cutoff_ratio,
            min_jitter_threshold=self.config.detection.audio_min_jitter_threshold,
        )
        self.debate_engine = CallGuardDebateEngine(
            prior_synthetic_prob=0.10,
            low_light_discount=self.config.detection.low_light_discount,
            bitrate_drop_discount=self.config.detection.bitrate_drop_discount,
            stutter_discount=self.config.detection.stutter_discount,
            multi_face_discount=self.config.detection.multi_face_discount,
        )
        self.hysteresis = HysteresisFilter(
            window_size=self.config.hysteresis.window_size,
            enter_threshold=self.config.hysteresis.enter_synthetic_threshold,
            enter_min_windows=self.config.hysteresis.enter_synthetic_min_windows,
            exit_threshold=self.config.hysteresis.exit_synthetic_threshold,
            exit_min_windows=self.config.hysteresis.exit_synthetic_min_windows,
        )
        self.audit_db = CallGuardAuditDB(db_path=self.config.audit.db_path) if self.config.audit.enabled else None
        self.scheduler = PriorityScheduler(max_duty_cycle=self.config.runtime.duty_cycle_budget)
        self.telemetry = TelemetryTracker()
        self.hud = CallGuardOverlayHUD()

        self._window_counter = 0
        self._is_running = False

    def start(self):
        """Start media capture ingest."""
        self._is_running = True
        self.source.start(self.video_buffer, self.audio_buffer)

    def stop(self):
        """Stop media capture ingest."""
        self._is_running = False
        self.source.stop()

    def process_step(self, apply_context_discount: bool = True) -> Dict[str, Any]:
        """
        Executes one full synchronous iteration of the CallGuard pipeline.
        Returns a rich execution summary.
        """
        t_cycle_start = time.monotonic()
        self._window_counter += 1
        seq = self._window_counter

        # 1. Fetch latest raw video frame and audio chunk from ring buffers
        video_sample = self.video_buffer.pop_oldest(timeout=0.05)
        raw_frame = video_sample.data if video_sample is not None else None
        t_capture = video_sample.timestamp if video_sample is not None else t_cycle_start

        # Collect audio accumulated in buffer (e.g. ~1 second of samples for FFT resolution)
        audio_window_samples = int(self.config.capture.audio_sample_rate * 1.0)
        audio_chunks = []
        collected_len = 0
        while collected_len < audio_window_samples:
            s = self.audio_buffer.pop_oldest(timeout=0.005)
            if s is None:
                break
            audio_chunks.append(s.data)
            collected_len += len(s.data)

        if audio_chunks:
            raw_audio = np.concatenate(audio_chunks)
        else:
            raw_audio = np.zeros(audio_window_samples, dtype=np.float32)

        # 2. ENHANCE PATH: Display copy only + Environmental Context Extraction
        t0 = time.perf_counter()
        run_superres = self.scheduler.should_run_stage("super_res", seq)
        run_lowlight = self.scheduler.should_run_stage("low_light", seq)

        display_frame, ctx = self.enhance_pipeline.process_display_frame(
            raw_frame=raw_frame,
            frame_timestamp=t_capture,
            apply_superres=run_superres,
            apply_lowlight=run_lowlight,
            apply_blur=False,
        )
        enhance_ms = (time.perf_counter() - t0) * 1000.0
        self.scheduler.record_stage_latency("super_res" if run_superres else "low_light", enhance_ms)
        self.telemetry.record_stage("enhance_path", enhance_ms)

        # 3. RAW TAP: Deepfake & Acoustic Voice Clone Detection on RAW media ONLY
        t0 = time.perf_counter()
        run_deepfake = self.scheduler.should_run_stage("deepfake_cadence", seq)
        evidences = []
        raw_metrics = {}

        if run_deepfake:
            evidences, raw_metrics = self.raw_tap.process_raw(
                raw_video_frame=raw_frame,
                raw_audio_chunk=raw_audio,
                t_start=t_capture - 1.0,
                t_end=t_capture,
                ctx=ctx,
            )
        raw_tap_ms = (time.perf_counter() - t0) * 1000.0
        self.scheduler.record_stage_latency("deepfake_cadence", raw_tap_ms)
        self.telemetry.record_stage("raw_tap", raw_tap_ms)

        # 4. CAPTIONS PATH: VAD-Gated Whisper (Never throttled)
        t0 = time.perf_counter()
        caption_text, caption_ms, vad_active = self.captions_pipeline.process_audio_chunk(raw_audio)
        if vad_active:
            self.scheduler.record_stage_latency("captions", caption_ms)
            self.telemetry.record_stage("captions_vad", caption_ms)

        # 5. CLAIM ADAPTER: Evidence -> Polarized Forensic Claims
        t0 = time.perf_counter()
        claims: List[Claim] = self.claim_adapter.adapt(evidences)
        claim_ms = (time.perf_counter() - t0) * 1000.0
        self.telemetry.record_stage("claim_adapter", claim_ms)

        # 6. ADVERSARIAL DEBATE & FUSION: Defender Context Discounting & Log-Odds
        t0 = time.perf_counter()
        debate_res: DebateResult = self.debate_engine.debate_and_fuse(
            claims=claims,
            ctx=ctx,
            apply_context_discount=apply_context_discount,
        )
        debate_ms = (time.perf_counter() - t0) * 1000.0
        self.telemetry.record_stage("debate_fusion", debate_ms)

        # 7. HYSTERESIS STATE MACHINE: Zero-Trust Badge Transitions
        h_state: HysteresisState = self.hysteresis.update(debate_res.posterior_probability)

        # 8. AUDIT DB: Cryptographic Hash-Chain Logging
        t0 = time.perf_counter()
        row_id, curr_hash = -1, "DISABLED"
        if self.audit_db is not None:
            claims_summary = [
                {"text": c.text, "polarity": c.polarity, "weight": c.weight, "confidence": c.confidence}
                for c in claims
            ]
            row_id, curr_hash = self.audit_db.record_window(
                window_id=seq,
                claims_summary=claims_summary,
                prior_prob=debate_res.prior_probability,
                posterior_prob=debate_res.posterior_probability,
                verdict=h_state.badge,
                confidence=debate_res.confidence,
            )
        audit_ms = (time.perf_counter() - t0) * 1000.0
        self.telemetry.record_stage("audit_hash_chain", audit_ms)

        # 9. END-TO-END TIMING & SCHEDULER UPDATE
        t_end = time.monotonic()
        frame_to_badge_ms = (t_end - t_capture) * 1000.0
        self.telemetry.record_frame_to_badge(frame_to_badge_ms)
        sched_state = self.scheduler.update_budget()

        # 10. OVERLAY HUD
        dropped = self.video_buffer.get_stats().get("total_dropped", 0)
        telemetry_summary = self.telemetry.get_summary()

        hud_text = self.hud.format_terminal_hud(
            badge=h_state.badge,
            probability=debate_res.posterior_probability,
            confidence=debate_res.confidence,
            captions=caption_text or self.captions_pipeline.last_caption,
            duty_cycle=sched_state["duty_cycle"],
            provider_name=self.hardware_label,
            stage_latencies=telemetry_summary["stages"],
            dropped_frames=dropped,
        )

        annotated_frame = self.hud.draw_on_frame(
            frame_bgr=display_frame,
            badge=h_state.badge,
            probability=debate_res.posterior_probability,
            captions=caption_text or self.captions_pipeline.last_caption,
            duty_cycle=sched_state["duty_cycle"],
            provider_name=self.hardware_label,
        )

        return {
            "window_id": seq,
            "badge": h_state.badge,
            "posterior_probability": debate_res.posterior_probability,
            "confidence": debate_res.confidence,
            "context_flags": ctx,
            "claims": [c.text for c in claims],
            "discounts_triggered": debate_res.context_discounts_triggered,
            "audit_row_id": row_id,
            "audit_hash": curr_hash,
            "duty_cycle": sched_state["duty_cycle"],
            "frame_to_badge_ms": frame_to_badge_ms,
            "terminal_hud": hud_text,
            "display_frame": annotated_frame,
            "telemetry": telemetry_summary,
        }
