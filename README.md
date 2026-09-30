# CallGuard NPU

> **Zero-Trust Multi-Modal Biometric & Acoustic Verification Shield for Snapdragon-Powered HP PCs**  
> *Developed, Tested, and Profiled on Physical Snapdragon HP Hardware (HP OmniBook X) with Portable CPU Fallback*

---

## 1. Executive Summary

**CallGuard NPU** is an edge-first, zero-trust assistive security and accessibility application engineered specifically for next-generation Snapdragon-powered HP PCs (such as the **HP OmniBook X** and **HP EliteBook Ultra**). It protects personal and enterprise video calls against real-time deepfake face synthesis, neural voice cloning, and audio impersonation.

By running entirely on-device, CallGuard delivers continuous physiological and acoustic verification with **zero cloud dependencies** and **zero network egress**, preserving user privacy while offloading compute-intensive models to the Qualcomm Hexagon NPU.

```
+-----------------------------------------------------------------------------------+
|                                  CallGuard NPU                                    |
|                                                                                   |
|  +--------------------+   +---------------------+   +--------------------------+  |
|  | Multi-Modal Ingest |-->|    RAW Tap Engine   |-->|   Adversarial Verifier   |  |
|  | - Camera / WASAPI  |   | - rPPG Pulse (POS)  |   | - Prosecutor (Anomalies) |  |
|  | - File / Synthetic |   | - Voice Flatness    |   | - Defender (Discounts)   |  |
|  | - Monotonic Clock  |   | - Pitch Micro-Jitter|   | - Judge (Bayesian Odds)  |  |
|  +--------------------+   +---------------------+   +--------------------------+  |
|             |                        |                           |                |
|             v                        v                           v                |
|  +--------------------+   +---------------------+   +--------------------------+  |
|  | Enhance Path       |   | NPU Queue Scheduler |   | Tamper-Evident Audit DB  |  |
|  | - CLAHE Low-Light  |   | - 60% Duty Budget   |   | - SHA-256 Hash Chaining  |  |
|  | - Context Telemetry|   | - Graceful Throttle |   | - SQLite Local Storage   |  |
|  +--------------------+   +---------------------+   +--------------------------+  |
|             \                        |                           /                |
|              +-----------------------+--------------------------+                 |
|                                      v                                            |
|                        +---------------------------+                              |
|                        | Real-Time HUD Overlay     |                              |
|                        | - Trust Badge / Captions  |                              |
|                        +---------------------------+                              |
+-----------------------------------------------------------------------------------+
```

---

## 2. Key Capabilities & Subsystems

CallGuard is built as a modular native application:

1. **Biometric & Acoustic Liveness (`callguard.biometrics`)**:
   - **Hemodynamic rPPG Engine**: Plane-Orthogonal-to-Skin (POS) and CHROM physiological pulse extraction using Butterworth bandpass filtering and Welch power spectral density (PSD) estimation.
   - **Synthetic Voice Detector**: Acoustic Wiener entropy / spectral flatness estimation, high-frequency cutoff ratio inspection, and vocal pitch micro-jitter analysis to expose neural vocoders.
   - **Face Mesh Tracker**: Isolates facial skin chrominance regions of interest (ROI) across changing poses.

2. **Video Studio & Environmental Telemetry (`callguard.vision`)**:
   - **Environmental Context Extractor**: Analyzes frames for low illumination (`low_light`), compression macro-blocking (`bitrate_drop`), frame delivery stutter (`stutter`), and multiple faces (`multi_face`).
   - **CLAHE Low-Light Boost**: Compensates for dark capture conditions to improve user video quality.
   - **Dual-Backend Image Operations**: Seamlessly switches between OpenCV and a cv2-free backend (PIL + SciPy + NumPy) ensuring native Windows ARM64 compatibility.

3. **Speech & Live Captions (`callguard.speech`)**:
   - **VAD-Gated Transcription**: Voice Activity Detection (VAD) monitors RMS audio energy to gate speech-to-text inference during silence, preventing idle compute waste.
   - **Speech Pipeline**: Connects Whisper ONNX inference with zero-dependency algorithmic demo fallbacks.

4. **Multi-Modal Evidence & Adversarial Debate (`callguard.core`)**:
   - **Evidence Contract**: Normalizes quantitative sensor readings with timestamped windows and environmental context flags.
   - **Claim Adapter**: Translates numerical telemetry into polarized forensic propositions (Prosecution vs. Defense).
   - **Adversarial Debate Engine**: Triadic verification where the Prosecutor argues synthetic manipulation, the Defender discounts anomalies explained by environmental degradation (e.g., poor lighting or packet loss), and the Judge computes calibrated log-odds Bayesian posterior probabilities.
   - **Decision Hysteresis**: Asymmetric state machine preventing UI flickering across transient network dips.
   - **Tamper-Evident Audit Database**: SQLite database with SHA-256 cryptographic hash-chaining linking every evaluation record to its predecessor.

5. **Runtime Scheduling & Hardware Abstraction (`callguard.runtime`)**:
   - **Unified Provider Factory**: Automatically targets Qualcomm `QNNExecutionProvider` (Hexagon NPU) when running on Snapdragon X platforms, with transparent fallback to `CPUExecutionProvider` on x64 systems.
   - **Priority Queue Scheduler**: Enforces a strict compute budget ($\le 60\%$ duty cycle per second) and dynamic progressive throttling under thermal load.
   - **Honest Latency Telemetry**: Tracks empirical p50 and p95 distributions; never fabricates unmeasured metrics.

---

## 3. Quick Start Guide

### 3.1 Installation

#### Prerequisites
- Python 3.10+ (Tested on Python 3.11.9)
- Windows 11 (ARM64 Snapdragon X or x64 AMD/Intel)

#### Setup
```powershell
# Clone or navigate to the CallGuard project folder
cd sna

# Install dependencies
pip install -r requirements.txt
```

*Note for Snapdragon X Devices:* To activate direct Qualcomm Hexagon NPU offloading, install `onnxruntime-qnn`:
```powershell
pip install onnxruntime-qnn
```

---

### 3.2 Running CallGuard

#### A. Synthetic Stream Mode (Headless / Self-Contained)
Runs the entire multi-modal pipeline with simulated video frames and audio signals:
```powershell
python run_callguard.py --synthetic --iterations 20
```

#### B. Simulating Deepfake Manipulation
Simulates cardiac pulse absence and acoustic vocoder high-frequency truncation to demonstrate alert escalation:
```powershell
python run_callguard.py --synthetic --simulate-deepfake --iterations 15
```

#### C. Testing Context Discounting (Low-Light False-Positive Mitigation)
Simulates dim lighting conditions to verify that the Defender Agent discounts missing pulse signals:
```powershell
python run_callguard.py --synthetic --low-light --iterations 15
```

#### D. Local Media File Playback Mode
Process pre-recorded video and audio files:
```powershell
python run_callguard.py --video sample.mp4 --audio sample.wav
```

#### E. Live Camera & Audio Mode
Connects directly to your system webcam and WASAPI audio stream:
```powershell
python run_callguard.py --video 0
```

#### F. Tamper-Evident Audit Trail Verification
Audits the cryptographic SHA-256 hash chain in the SQLite database to verify data integrity:
```powershell
python run_callguard.py --verify-audit
```

---

## 4. Verification & Testing

Execute the comprehensive automated test suite (19 tests covering pipeline stages, memory buffers, cryptographic hash chains, scheduling, and network isolation):

```powershell
python -m pytest -v tests/
```

### Test Suite Structure
| Test File | Verification Scope |
| :--- | :--- |
| `tests/test_ring_buffer.py` | Monotonic timestamps, drop-oldest overflow, and thread concurrency. |
| `tests/test_claim_adapter.py` | Translation of raw biometric telemetry into polarized forensic propositions. |
| `tests/test_fusion_hysteresis.py` | Calibrated log-odds Bayesian fusion and asymmetric window hysteresis. |
| `tests/test_audit_hash_chain.py` | Tamper-evident SHA-256 cryptographic linkage and audit integrity validation. |
| `tests/test_network_egress.py` | Strictly enforces zero non-loopback network calls during runtime. |
| `tests/test_scheduler.py` | Enforces 60% NPU duty-cycle budget and progressive task throttling order. |
| `tests/test_smoke.py` | End-to-end multi-modal pipeline execution with CPU fallback. |
| `tests/test_ablation.py` | Mathematical verification of Defender environmental context discounting. |
| `tests/test_benchmarks.py` | Empirical p50 and p95 latency tracking across pipeline configurations. |

---

## 5. Architectural Principles & Security Posture

1. **Zero-Trust RAW Tap Isolation**:
   Biometric and voice liveness detection is performed strictly on raw, pristine sensor inputs before any video enhancement filter is applied. This prevents video enhancements (such as super-resolution or face smoothing) from corrupting subtle physiological capillary signals.

2. **Environmental Context Discounting**:
   Adversarial debate leverages environmental telemetry (low-light noise, network packet drops, frame stutter) to discount anomalous readings, dramatically reducing false alarm rates during challenging call conditions.

3. **Zero Network Egress**:
   All inference, verification, and audit logging runs entirely on the local device. No user frames, audio snippets, or telemetry are transmitted externally.

4. **Snapdragon Hardware Verification**:
   Verified on physical Snapdragon-powered HP PCs (HP OmniBook X) with Qualcomm Hexagon NPU offloading via `QNNExecutionProvider`, alongside portable CPU fallback for cross-platform validation. Metric fields reflect empirical measurements across all execution stages.

---

## 6. Project Layout

```
CallGuard/
├── callguard/                  # Core application package
│   ├── biometrics/             # rPPG pulse extraction, audio analysis, face tracking
│   ├── vision/                 # Image ops (cv2 + cv2-free), CLAHE stages, video I/O
│   ├── speech/                 # Audio capture, RMS VAD gating, speech engine
│   ├── core/                   # Evidence, claims, debate engine, audit DB, hysteresis
│   ├── runtime/                # Hardware provider factory, priority scheduler, telemetry
│   ├── capture/                # Timestamped ring buffers, multi-modal sources
│   ├── ui/                     # Overlay HUD & advisory trust badges
│   ├── config.py               # Central configuration loader
│   └── pipeline.py             # Master orchestration loop
├── aihub/                      # Qualcomm AI Hub cloud profiling automation
│   └── profile_all.py
├── eval/                       # Dataset evaluation hooks (FF++, ASVspoof)
│   └── eval_hooks.py
├── docs/                       # Technical architecture, decisions, benchmarks, HP proposal
│   ├── HP_SNAPDRAGON_PROPOSAL.md
│   ├── architecture.md
│   ├── benchmarks.md
│   ├── decisions.md
│   └── limitations.md
├── tests/                      # Automated unit, integration, and ablation tests
├── callguard.yaml              # Default configuration
├── requirements.txt            # System dependencies
└── run_callguard.py            # CLI entry point
```

---

## 7. License & Compliance Notice

*Notice: Verdict text generated in the CallGuard UI is advisory only.*  
Engineered for deployment on Qualcomm Snapdragon X series platforms and Snapdragon-powered HP PCs.
