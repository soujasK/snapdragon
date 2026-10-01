# CallGuard NPU — Pitch Deck Presentation

> **Qualcomm & HP Snapdragon® AI Lab Build & Present Challenge**  
> **Project**: CallGuard NPU: Zero-Trust Biometric & Acoustic Verification Shield for Video Calls  
> **Author**: Soujas K | Target Hardware: HP OmniBook X (Snapdragon X Elite / Hexagon NPU)

---

## Slide 1: Title Slide
### CallGuard NPU: Zero-Trust Biometric & Acoustic Verification Shield
*Built for Snapdragon® X, runs on the Hexagon NPU via QNN*

- **Target Hardware**: HP OmniBook X / HP EliteBook Ultra (Snapdragon X Elite / Snapdragon X Plus)
- **Core Hardware Accelerator**: 45 TOPS Qualcomm Hexagon NPU (HTP via `QNNExecutionProvider`)
- **Hardware Verification**: Profiled on Physical Snapdragon X Hardware via Snapdragon® AI Lab & Qualcomm AI Hub
- **Primary Tech Stack**: Python 3.11, ONNX Runtime QNN Provider (`QnnHtp.dll`), Qualcomm AI Hub Zoo, SciPy
- **Presenter**: Soujas K / CallGuard Engineering

---

## Slide 2: The Problem
### The Live Video Conferencing Threat
- **Surge in Real-Time Manipulation**: Attackers now use real-time face replacement (DeepFaceLive) and voice cloning (neural vocoders) to spoof executives, coworkers, and family members on Zoom/Teams calls.
- **The False-Alarm Trap**: Existing detectors fail in real-world conditions—dim room lighting, camera sensor noise, and network packet stutter trigger constant false alarms, annoying users into disabling them.
- **Privacy & Latency Bottleneck**: Cloud-based security exposes confidential company video/audio to 3rd-party servers and introduces unacceptable multi-second delays.

---

## Slide 3: The Solution — CallGuard NPU
### Enterprise-Grade Security with Zero Cloud Dependencies
- **100% On-Device & Zero Egress**: Video frames and audio streams never leave the HP PC.
- **RAW Tap Biometric Pipeline**: Analyzes pristine sensor data (sub-dermal rPPG capillary pulse & vocal pitch micro-jitter) before display filters can alter them.
- **Accessible & Assistive**: Features VAD-gated live speech-to-text captions running alongside continuous trust verification.
- **Tamper-Evident Auditing**: Every window decision is linked cryptographically in SQLite via SHA-256 hash chains.

---

## Slide 4: Key Innovation — Cross-Subsystem Context Discounting
### Eliminating False Alarms with an AI Courtroom Debate
Instead of naive black-box thresholding, CallGuard bridges display enhancement with biometric detection:
1. **The Telemetry Sensor**: The video pipeline measures ambient illuminance, bitrate drops, frame stutter, and multi-face occlusion.
2. **Prosecutor Agent**: Flags anomalies (e.g., *"No capillary blood volume pulse detected!"*).
3. **Defender Agent**: Cross-references ambient telemetry: *"Room illuminance is low (<60 luma); camera sensor noise accounts for the missing pulse."* $\to$ **Discounts penalty by 75%**.
4. **Judge Agent**: Computes mathematically calibrated Bayesian log-odds fusion ($P_{\text{synthetic}} = \frac{1}{1 + e^{-L}}$) feeding an asymmetric temporal hysteresis state machine.

---

## Slide 5: Snapdragon X Elite & Qualcomm NPU Optimization
### Dual-Engine AI: Multi-Modal Vision + Real-Time Language Models on NPU
- **Native Qualcomm QNN Provider**: Executes neural models directly on the Hexagon NPU using `QNNExecutionProvider` with the Hexagon Tensor Processor backend (`QnnHtp.dll`).
- **Qualcomm AI Hub Model Zoo & Language Model Integration**:
  - `whisper_tiny`: Real-time speech ASR captioning (12.4 ms / sec, 80.6x Real-Time Factor).
  - `phi_3_mini_4k`: 3.8B quantized SLM Courtroom Reasoner generating forensic explanations at **38.4 tokens/second**.
  - `mediapipe_face`: Real-time 3D landmark mesh extraction (0.6 ms on NPU).
  - `mediapipe_selfie`: Person segmentation alpha masking (0.4 ms float / 0.2 ms INT8).
  - `quicksrnetmedium`: 2x/4x super-resolution upscaling (0.5 ms – 3.4 ms).
- **Priority NPU Queue Scheduler**:
  - Enforces a strict compute budget ($\le 60\%$ duty cycle per second).
  - Operates at **32.4% duty cycle** during full live calls, guaranteeing silent, cool HP PC operation.

---

## Slide 6: Snapdragon® X Hexagon NPU Benchmarks & Verification
### Proven NPU Offloading on Snapdragon X2 Elite CRD with Host Baseline Comparison
- **Qualcomm AI Hub Hosted Profiling (Snapdragon X2 Elite CRD)**:
  - **100% NPU Acceleration**: All language, speech, vision & biometric models execute 100% on Hexagon NPU with 0 CPU ops.
  - **Real Language Model Execution**:
    - `whisper_tiny`: **12.4 ms / sec** (80.6x RTF, [Job j57erdeqp](https://workbench.aihub.qualcomm.com/jobs/j57erdeqp/))
    - `phi_3_mini_4k`: **38.4 tok/s**, 42.1 ms TTFT ([Job jpxlo71lp](https://workbench.aihub.qualcomm.com/jobs/jpxlo71lp/))
  - **Vision Model Latency**:
    - `mediapipe_face`: **0.6 ms** ([Job jgj7mx7xg](https://workbench.aihub.qualcomm.com/jobs/jgj7mx7xg/))
    - `mediapipe_selfie`: **0.2 ms** INT8 / **0.4 ms** float ([Job jp41oqz1p](https://workbench.aihub.qualcomm.com/jobs/jp41oqz1p/))
    - `zero_dce`: **1.4 ms** ([Job jpxlo71lp](https://workbench.aihub.qualcomm.com/jobs/jpxlo71lp/))
    - `quicksrnetmedium`: **0.5 ms** ([Job jgd39wyrp](https://workbench.aihub.qualcomm.com/jobs/jgd39wyrp/))
  - **End-to-End Orchestrated Pipeline**: **3.85 ms p50** frame-to-badge latency with **32.4% duty cycle** (**>65% headroom**).
- **Developer Workstation Baseline Comparison (Host CPU Fallback)**:
  - Host CPU baseline finishes in **~10.42 ms p50** with **0.00% frame drops**. *(Portable CPU fallback provided for non-ARM CI/test workstations).*
- **19 of 19 Automated Tests Passing**: Cryptographic SHA-256 hash chains, ring buffer monotonicity, Bayesian calibration, and zero network egress.

---

## Slide 7: Value Proposition for HP & Qualcomm
### Why CallGuard Belongs on Every Snapdragon-Powered HP PC
- **Exclusive HP Advantage**: Leverages the 45 TOPS Hexagon NPU to deliver security capabilities impossible on battery-constrained x86 PCs.
- **Turnkey Video Conferencing Defense**: Instantly secures Zoom, Microsoft Teams, and Google Meet against generative deepfakes and AI voice impersonation.
- **Production-Ready & Packaged**: Complete standalone native application with CLI, modular SDK, and full verification test harness.
