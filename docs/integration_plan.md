# CallGuard NPU - Subsystem Architecture & Specification

This document details the functional specifications and implementation breakdown for all subsystems comprising **CallGuard NPU**.

---

## 1. Subsystem Breakdown

| Subsystem / Capability | Module Path | Purpose & Architectural Role |
|---|---|---|
| **Capture & Synchronized Buffering** | `callguard/capture/` | Provides thread-safe, bounded, timestamped circular ring buffers (`TimestampedRingBuffer`) governed by a single monotonic clock, with a drop-oldest policy. Ingest drivers (`sources.py`) support synchronized live Windows devices, media file playback, and synthetic streaming. |
| **RAW Tap Biometric Pipeline** | `callguard/biometrics/`, `callguard/core/raw_tap.py` | Extracts physiological capillary pulse via Remote Photoplethysmography (POS & CHROM rPPG), inspects acoustic Wiener entropy / spectral flatness, and analyzes vocal pitch micro-jitter. Executes **strictly on raw, unenhanced sensor inputs**. |
| **Enhancement & Environmental Telemetry** | `callguard/vision/`, `callguard/core/enhance_path.py` | Runs CLAHE low-light enhancement strictly on display frames, and continuously extracts environmental context flags (`low_light`, `bitrate_drop`, `stutter`, `multi_face`) that feed into the Defender discount logic. |
| **VAD-Gated Live Speech Captions** | `callguard/speech/`, `callguard/core/captions.py` | Performs live speech-to-text transcription. Integrates Voice Activity Detection (VAD) / RMS energy gating to avoid running inference during silent pauses, conserving NPU duty cycles. |
| **Normalized Evidence Structure** | `callguard/core/evidence.py` | Defines the canonical schema linking quantitative sensor readings with timestamped windows and environmental context flags. |
| **Forensic Claim Adapter** | `callguard/core/claim_adapter.py` | Parameterized templates translating numerical Evidence into polarized linguistic claims (Prosecution vs. Defense) with calibrated evidence weights. |
| **Adversarial Debate & Context Discounting** | `callguard/core/debate.py` | Triadic verification engine where the Prosecutor argues synthetic manipulation, the Defender discounts anomalies explained by environmental context (e.g. low-light noise or network drops), and the Judge calculates calibrated log-odds Bayesian fusion. |
| **Cryptographic Audit Trail** | `callguard/core/audit.py` | SQLite audit database implementing SHA-256 cryptographic hash-chaining across successive evaluation windows to guarantee tamper-evident verification. |
| **Decision Hysteresis** | `callguard/core/hysteresis.py` | Asymmetric temporal state machine preventing UI flickering across transient network dips (enters alert on $P \ge 0.75$ in 3 of 5 windows; exits on $P \le 0.50$ in 4 of 5 windows). |
| **Unified Runtime & Provider Factory** | `callguard/runtime/factory.py` | Central factory managing ONNX Runtime sessions, enforcing Qualcomm `QNNExecutionProvider` (Hexagon NPU) with transparent `CPUExecutionProvider` fallback, and logging exact active providers. |
| **Priority NPU Queue Scheduler** | `callguard/runtime/scheduler.py` | Enforces a strict compute budget ($\sum (\text{cost} \times \text{rate}) \le 60\%/\text{sec}$) with intelligent progressive throttling under thermal load (super-res $\to$ low-light $\to$ deepfake cadence; never throttling captions or landmarks). |
| **Live Overlay HUD & Telemetry** | `callguard/ui/overlay.py` | Renders the real-time HUD with trust status badges (CONSISTENT, UNCERTAIN, LIKELY SYNTHETIC), live speech captions, empirical p50/p95 stage latencies, and the mandatory advisory legal disclaimer. |
| **Configuration Engine** | `callguard/config.py`, `callguard.yaml` | Validated configuration management for all system parameters, thresholds, and execution modes. |
