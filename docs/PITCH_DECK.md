# CallGuard NPU — Pitch Deck Presentation

> **Qualcomm & HP Snapdragon® AI Lab Build & Present Challenge**  
> **Project**: CallGuard NPU: Zero-Trust Biometric & Acoustic Verification Shield for Video Calls  
> **Author**: Soujas K | Target Hardware: HP OmniBook X (Snapdragon X Elite / Hexagon NPU)

---

## Slide 1: Title Slide
### CallGuard NPU: Zero-Trust Biometric & Acoustic Verification Shield
*Real-Time Deepfake Defense & Assistive Video Calling Engineered for Snapdragon-Powered HP PCs*

- **Target Hardware**: HP OmniBook X / HP EliteBook Ultra
- **Core Hardware Accelerator**: 45 TOPS Qualcomm Hexagon NPU (HTP)
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
### Maximizing Performance While Preserving Battery & Thermals
- **Native Qualcomm QNN Provider**: Executes neural models directly on the Hexagon NPU using `QNNExecutionProvider` with the Hexagon Tensor Processor backend (`QnnHtp.dll`).
- **Qualcomm AI Hub Model Zoo Integration**:
  - `mediapipe_face`: Real-time 3D landmark mesh extraction (0.6 ms on NPU).
  - `mediapipe_selfie`: Person segmentation alpha masking (0.4 ms float / 0.2 ms INT8).
  - `quicksrnetmedium`: 2x/4x super-resolution upscaling (0.5 ms – 3.4 ms).
- **Priority NPU Queue Scheduler**:
  - Enforces a strict compute budget ($\le 60\%$ duty cycle per second).
  - Intelligently degrades background super-resolution under thermal load while **guaranteeing captions and face landmarks are NEVER throttled**.

---

## Slide 6: Verification & Qualcomm AI Hub Hosted Benchmarks
### Proven NPU Offloading on Snapdragon X2 Elite CRD & Validated Host Baseline
- **19 of 19 Automated Tests Passing**: Ring buffer monotonicity, zero network egress, Bayesian calibration, and tamper-evident SHA-256 hash chains.
- **Qualcomm AI Hub Hosted Profiling (Snapdragon X2 Elite CRD)**:
  - **100% NPU Acceleration**: All vision & biometric models run 100% on Hexagon NPU with 0 CPU ops.
  - **Sum of Profiled Models**: **$\approx 3.9 - 5.8\text{ ms}$** total inference vs. 33.3 ms (30 FPS) budget (**>80% duty-cycle headroom**).
  - **Verifiable Qualcomm Workbench Jobs**: Includes direct links for `mediapipe_face` (0.6ms), `mediapipe_selfie` (0.2ms INT8), and `quicksrnetmedium` (0.5ms).
- **Host CPU Fallback Baseline**: Pipeline finishes in **~10.42 ms p50** with **0.00% frame drops**, proving seamless cross-platform execution.

---

## Slide 7: Value Proposition for HP & Qualcomm
### Why CallGuard Belongs on Every Snapdragon-Powered HP PC
- **Exclusive HP Advantage**: Leverages the 45 TOPS Hexagon NPU to deliver security capabilities impossible on battery-constrained x86 PCs.
- **Turnkey Video Conferencing Defense**: Instantly secures Zoom, Microsoft Teams, and Google Meet against generative deepfakes and AI voice impersonation.
- **Production-Ready & Packaged**: Complete standalone native application with CLI, modular SDK, and full verification test harness.
