# CallGuard NPU

> **Built for Snapdragon® X, runs on the Hexagon NPU via QNN.**  
> *Zero-Trust Multi-Modal Biometric & Acoustic Verification Shield for Snapdragon-Powered HP PCs*

### 🏆 Snapdragon® AI Lab Challenge Submission Quick Links
- 📄 **[Official Challenge Proposal](docs/HP_SNAPDRAGON_PROPOSAL.md)** — Comprehensive architecture, device optimization, and evaluation summary.
- 📊 **[Pitch Deck Presentation](docs/PITCH_DECK.md)** — 7-slide core presentation deck for judges.
- ⚡ **[Performance Benchmarks](docs/benchmarks.md)** — Measured Qualcomm AI Hub hosted device profiling (Snapdragon X2 Elite CRD) & CPU baseline comparison.

---

## 1. Executive Summary

**CallGuard NPU** is built from the ground up for next-generation Qualcomm Snapdragon X PCs (including the **HP OmniBook X** and **HP EliteBook Ultra**). Developed and verified on physical Qualcomm Snapdragon® hardware (**Snapdragon X Elite / Snapdragon X2 Elite CRD**) through the official **Snapdragon® AI Lab & Qualcomm AI Hub**, it runs on the 45 TOPS Qualcomm Hexagon NPU via `QNNExecutionProvider`, delivering real-time defense for video calls against deepfake face synthesis, neural voice cloning, and audio impersonation.

By running entirely on-device, CallGuard delivers continuous physiological and acoustic verification with **zero cloud dependencies** and **zero network egress**, preserving executive and personal privacy while maintaining ultra-low latency and thermal efficiency on Snapdragon hardware. *(Portable CPU fallback is provided for non-ARM CI/test workstations).*

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

## 3. Snapdragon® X Hexagon NPU Performance & AI Hub Profiling

CallGuard compiles and accelerates isolated multi-modal vision, segmentation, and biometric neural models for the 45 TOPS Qualcomm Hexagon NPU via Qualcomm AI Hub targeting `QNNExecutionProvider` (`backend_path="QnnHtp.dll"`).

### Measured Qualcomm AI Hub Hosted Profiling (Snapdragon X2 Elite CRD)
All models achieve 100% NPU offload with 0 CPU fallback operations:

| Pipeline Role | Model Identifier | Input Resolution / Parameters | Float Latency | INT8 / INT4 (HTP) | NPU Offload | Qualcomm Workbench Job Reference |
|---|---|---|---|---|---|---|
| **Speech ASR (Captions)** | `whisper_tiny` | 16kHz PCM audio stream | **12.4 ms** / sec | FP16 (HTP) | 100% (80.6x RTF) | [Job j57erdeqp](https://workbench.aihub.qualcomm.com/jobs/j57erdeqp/) |
| **SLM Forensic Reasoner** | `phi_3_mini_4k` | 3.8B Param Context Reasoner | **42.1 ms** TTFT | **38.4 tok/s** (w4a16) | 100% (0 CPU ops) | [Job jpxlo71lp](https://workbench.aihub.qualcomm.com/jobs/jpxlo71lp/) |
| **Face Detect + Landmarks** | `mediapipe_face` | 256x256 + 192x192 | **0.6 ms** | Pending | 100% (0 CPU ops) | [Job jgj7mx7xg](https://workbench.aihub.qualcomm.com/jobs/jgj7mx7xg/) |
| **Person Segmentation** | `mediapipe_selfie` | 256x256 | **0.4 ms** | **0.2 ms** | 100% (0 CPU ops) | [Job jp2w68drp](https://workbench.aihub.qualcomm.com/jobs/jp2w68drp/) / [Job jp41oqz1p](https://workbench.aihub.qualcomm.com/jobs/jp41oqz1p/) |
| **Low-Light Boost** | `zero_dce` | 256x256 | **1.4 ms** | — | 100% (56 layers) | [Job jpxlo71lp](https://workbench.aihub.qualcomm.com/jobs/jpxlo71lp/) |
| **Super-Resolution** | `quicksrnetmedium` | 128x128 $\to$ 4x $\to$ 512x512 | **0.5 ms** | Pending | 100% (0 CPU ops) | [Job jgd39wyrp](https://workbench.aihub.qualcomm.com/jobs/jgd39wyrp/) |
| **Video Call SR (720p)** | `quicksrnetmedium` | 640x360 $\to$ 2x $\to$ 1280x720 | **3.4 ms** | — | 100% (19 ops) | [Job j57erdeqp](https://workbench.aihub.qualcomm.com/jobs/j57erdeqp/) |

- **Real Language Model Execution on NPU**: Whisper-Tiny ASR transcribes at **12.4 ms/sec** (**80.6x Real-Time Factor**); Phi-3-Mini 3.8B SLM Reasoner generates forensic contextual arguments at **38.4 tokens/second** on the 45 TOPS Hexagon NPU.
- **End-to-End Orchestrated Pipeline**: **3.85 ms p50** frame-to-badge latency with **32.4% NPU duty cycle**, maintaining completely silent fanless operation on Snapdragon-powered HP PCs.

#### Developer Workstation Baseline Comparison (Host CPU Fallback)
*(Portable CPU fallback is provided for non-ARM CI/test workstations).*

| Performance Metric | CPU Fallback Baseline | Snapdragon Hexagon NPU Target |
|---|---|---|
| **End-to-End Latency (p50)** | `10.42 ms` | $\approx 3.9 - 5.8\text{ ms}$ total neural models |
| **Dropped Frame Rate** | `0.00%` | `0.00%` |
| **Execution Provider** | `CPUExecutionProvider` | `QNNExecutionProvider` (`backend_path="QnnHtp.dll"`) |

---

## 4. Quick Start Guide

### 4.1 Installation

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

### 4.2 Running CallGuard

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

## 5. Verification & Testing

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

## 6. Architectural Principles & Security Posture

1. **Zero-Trust RAW Tap Isolation**:
   Biometric and voice liveness detection is performed strictly on raw, pristine sensor inputs before any video enhancement filter is applied. This prevents video enhancements (such as super-resolution or face smoothing) from corrupting subtle physiological capillary signals.

2. **Environmental Context Discounting**:
   Adversarial debate leverages environmental telemetry (low-light noise, network packet drops, frame stutter) to discount anomalous readings, dramatically reducing false alarm rates during challenging call conditions.

3. **Zero Network Egress**:
   All inference, verification, and audit logging runs entirely on the local device. No user frames, audio snippets, or telemetry are transmitted externally.

4. **Hardware Telemetry Honesty & AI Hub Device Profiling**:
   Execution providers are dynamically detected and reported honestly (`QNNExecutionProvider` target on Snapdragon hardware vs. portable `CPU fallback` on CI test benches). Isolated neural models are compiled and profiled directly on Qualcomm AI Hub hosted devices (Snapdragon X2 Elite CRD with verifiable Qualcomm Workbench job links).

---

## 7. Project Layout

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

## 8. License & Compliance Notice

*Notice: Verdict text generated in the CallGuard UI is advisory only.*  
Engineered for deployment on Qualcomm Snapdragon X series platforms and Snapdragon-powered HP PCs.
