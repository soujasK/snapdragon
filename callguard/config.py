"""
CallGuard Configuration Loader.

Architectural Design:
    Provides a centralized, validated configuration object loaded from callguard.yaml
    or falling back to sensible system defaults. Governs multi-modal ring buffers,
    NPU duty cycles, debate discount weights, and hysteresis states.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class CaptureConfig:
    video_fps: int = 30
    audio_sample_rate: int = 16000
    ring_buffer_video_capacity: int = 120
    ring_buffer_audio_capacity: int = 160000
    drop_oldest: bool = True


@dataclass
class RuntimeConfig:
    provider_order: List[str] = field(default_factory=lambda: ["QNNExecutionProvider", "CPUExecutionProvider"])
    qnn_options: Dict[str, Any] = field(default_factory=lambda: {
        "backend_path": "QnnHtp.dll",
        "htp_performance_mode": "burst",
        "htp_precision": "float16",
    })
    duty_cycle_budget: float = 0.60


@dataclass
class SchedulerConfig:
    throttle_stages_order: List[str] = field(default_factory=lambda: [
        "super_res", "low_light", "deepfake_cadence"
    ])
    protected_stages: List[str] = field(default_factory=lambda: [
        "landmarks", "captions"
    ])


@dataclass
class DetectionConfig:
    rppg_min_bpm: float = 45.0
    rppg_max_bpm: float = 180.0
    rppg_snr_threshold_db: float = 1.5
    audio_flatness_threshold: float = 0.25
    audio_high_freq_cutoff_ratio: float = 0.08
    audio_min_jitter_threshold: float = 0.005
    low_light_discount: float = 0.25
    bitrate_drop_discount: float = 0.30
    stutter_discount: float = 0.30
    multi_face_discount: float = 0.40


@dataclass
class HysteresisConfig:
    window_size: int = 5
    enter_synthetic_threshold: float = 0.75
    enter_synthetic_min_windows: int = 3
    exit_synthetic_threshold: float = 0.50
    exit_synthetic_min_windows: int = 4


@dataclass
class AuditConfig:
    db_path: str = "callguard_audit.db"
    enabled: bool = True


@dataclass
class CallGuardConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)


def load_config(config_path: str = "callguard.yaml") -> CallGuardConfig:
    """Load configuration from YAML file or return defaults if file not found."""
    if not os.path.exists(config_path):
        return CallGuardConfig()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        cap_data = raw.get("capture", {})
        run_data = raw.get("runtime", {})
        sched_data = raw.get("scheduler", {})
        det_data = raw.get("detection", {})
        rppg_data = det_data.get("rppg", {})
        audio_data = det_data.get("audio", {})
        disc_data = det_data.get("context_discounting", {})
        hyst_data = raw.get("hysteresis", {})
        aud_data = raw.get("audit", {})

        return CallGuardConfig(
            capture=CaptureConfig(
                video_fps=cap_data.get("video_fps", 30),
                audio_sample_rate=cap_data.get("audio_sample_rate", 16000),
                ring_buffer_video_capacity=cap_data.get("ring_buffer_video_capacity", 120),
                ring_buffer_audio_capacity=cap_data.get("ring_buffer_audio_capacity", 160000),
                drop_oldest=cap_data.get("drop_oldest", True),
            ),
            runtime=RuntimeConfig(
                provider_order=run_data.get("provider_order", ["QNNExecutionProvider", "CPUExecutionProvider"]),
                qnn_options=run_data.get("qnn_options", {}),
                duty_cycle_budget=run_data.get("duty_cycle_budget", 0.60),
            ),
            scheduler=SchedulerConfig(
                throttle_stages_order=sched_data.get("throttle_stages_order", ["super_res", "low_light", "deepfake_cadence"]),
                protected_stages=sched_data.get("protected_stages", ["landmarks", "captions"]),
            ),
            detection=DetectionConfig(
                rppg_min_bpm=rppg_data.get("min_bpm", 45.0),
                rppg_max_bpm=rppg_data.get("max_bpm", 180.0),
                rppg_snr_threshold_db=rppg_data.get("snr_threshold_db", 1.5),
                audio_flatness_threshold=audio_data.get("flatness_threshold", 0.25),
                audio_high_freq_cutoff_ratio=audio_data.get("high_freq_cutoff_ratio", 0.08),
                audio_min_jitter_threshold=audio_data.get("min_jitter_threshold", 0.005),
                low_light_discount=disc_data.get("low_light_discount", 0.25),
                bitrate_drop_discount=disc_data.get("bitrate_drop_discount", 0.30),
                stutter_discount=disc_data.get("stutter_discount", 0.30),
                multi_face_discount=disc_data.get("multi_face_discount", 0.40),
            ),
            hysteresis=HysteresisConfig(
                window_size=hyst_data.get("window_size", 5),
                enter_synthetic_threshold=hyst_data.get("enter_synthetic_threshold", 0.75),
                enter_synthetic_min_windows=hyst_data.get("enter_synthetic_min_windows", 3),
                exit_synthetic_threshold=hyst_data.get("exit_synthetic_threshold", 0.50),
                exit_synthetic_min_windows=hyst_data.get("exit_synthetic_min_windows", 4),
            ),
            audit=AuditConfig(
                db_path=aud_data.get("db_path", "callguard_audit.db"),
                enabled=aud_data.get("enabled", True),
            ),
        )
    except Exception as exc:
        print(f"[CallGuard Config] Warning: Failed to parse {config_path} ({exc}), using defaults.")
        return CallGuardConfig()
