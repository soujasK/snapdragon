# CallGuard NPU: Official Submission Proposal for Snapdragon-Powered HP PCs

## 1. Project Title & Tagline
- **Project Title**: **CallGuard NPU: Zero-Trust Real-Time Biometric & Acoustic Verification Shield**
- **Tagline**: An edge-first security and accessibility shield designed and optimized for Snapdragon-powered HP PCs to detect real-time video deepfakes and voice clones during live conference calls.

---

## 2. Target Device & Snapdragon Optimization
- **Target Hardware**: **Snapdragon-Powered HP PCs** (engineered for **HP OmniBook X** and **HP EliteBook Ultra**, featuring Qualcomm Snapdragon X Elite / Snapdragon X Plus and the 45 TOPS Qualcomm Hexagon NPU).
- **Core Hardware Accelerator**: 45 TOPS Qualcomm Hexagon NPU (HTP).
- **Why It Is Optimized for HP Snapdragon PCs**:
  1. **All-Day Battery Life & Silent Operation**: Traditional deepfake detectors drain battery and force cooling fans into high gear on x86 machines. On Snapdragon-powered HP PCs, CallGuard offloads neural inference to the dedicated Hexagon NPU using `QNNExecutionProvider` (Qualcomm Neural Network execution provider with `QnnHtp.dll`).
  2. **Priority NPU Queue Scheduler**: Constrains total NPU duty cycle to $\le 60\%$ compute per second ($\sum \text{latency} \times \text{rate} \le 0.60$), guaranteeing the HP laptop remains cool and responsive even during multi-hour Zoom/Teams meetings.
  3. **Zero Network Egress**: Runs 100% locally on-device. Personal camera video and conference audio never leave the HP device, preserving executive and personal privacy.

---

## 3. Qualcomm AI Hub Models & Open-Source Integration
CallGuard incorporates optimized models directly from the **Qualcomm AI Hub Zoo** and open-source platforms:
1. **`mediapipe_face` (Qualcomm AI Hub Zoo)**: Real-time face detection and 468 3D landmark mesh extraction for capillary ROI isolation (profiled at 0.6 ms on Hexagon NPU).
2. **`mediapipe_selfie` (Qualcomm AI Hub Zoo)**: Person segmentation alpha-masking for background blur and composite display enhancement (profiled at 0.4 ms float / 0.2 ms INT8).
3. **`quicksrnetmedium` (Qualcomm AI Hub Zoo)**: Efficient 2x/4x super-resolution upscaling for video conferencing display copies (profiled at 0.5 ms – 3.4 ms).
4. **Physiological rPPG Hemodynamic Pulse Engine (Open-Source / SciPy)**: Plane-Orthogonal-to-Skin (POS) and CHROM sub-dermal capillary blood-volume pulse (BVP) extraction.
5. **Whisper Speech-to-Text ASR (Qualcomm AI Hub / Open-Source)**: Live speech captions gated by Voice Activity Detection (VAD) RMS energy to preserve compute duty cycle.
6. **Adversarial Triadic Debate & SQLite Audit**: Multi-modal streaming biometric verification, discounting environmental anomalies, and logging tamper-evident SHA-256 hash chains.

Includes an automated compilation and profiling harness (`aihub/profile_all.py`) that profiles models directly on Qualcomm AI Hub hosted devices (`Snapdragon X Elite CRD`).

---

## 4. Problem Statement & Innovation

### The Problem
Generative AI face replacement (DeepFaceLive, SimSwap) and real-time voice cloning (TTS vocoders) now enable attackers to impersonate colleagues, executives, and family members on live video calls. Traditional detection approaches fail in real-world conditions because bad lighting, network stutter, or compression artifacts trigger constant false alarms.

### The Innovation: Cross-Subsystem Context Discounting
CallGuard solves false alarms by bridging video enhancement and biometric detection:
- The video enhancement pipeline detects environmental quality flags: `low_light`, `bitrate_drop`, `stutter`, `multi_face`.
- When an anomaly occurs (e.g. absent pulse), CallGuard initiates a triadic **AI Courtroom Debate**:
  - **Prosecutor**: Accuses the missing pulse as a deepfake attack.
  - **Defender**: Objects using ambient telemetry: *"The room is dark (luminance < 60); sensor noise accounts for the missing pulse."* The Defender discounts the claim by 75%.
  - **Judge**: Calculates mathematically calibrated log-odds Bayesian fusion ($P = \frac{1}{1 + e^{-L}}$) and feeds an asymmetric temporal hysteresis state machine.
- All decisions are cryptographically recorded into a tamper-evident SQLite audit log via SHA-256 hash chains.

---

## 5. Verification & Testing Evidence
- **Qualcomm AI Hub Device Farm Profiling**: Neural models compiled and profiled on hosted Snapdragon X2 Elite hardware, achieving 100% NPU offload with a total model inference sum of $\approx 3.9 - 5.8\text{ ms}$ (well under the 33.3 ms 30fps budget).
- **19 of 19 automated unit, smoke, ablation, and security tests pass**.
- **CPU Fallback Baseline**: Complete multi-modal pipeline runs in ~10.42 ms p50 on host CPU fallback, proving robust cross-platform portability.
- **Security Verified**: Zero non-loopback network egress enforced by automated socket testing.

---

## 6. How Judges Can Evaluate (Under 60 Seconds)
```powershell
# 1. Clone repository & install dependencies
git clone <YOUR_REPO_URL>
cd CallGuard-NPU
pip install -r requirements.txt

# 2. Run automated test suite
python -m pytest -v tests/

# 3. Run live application (Authentic Human Demo)
python run_callguard.py --synthetic --iterations 10

# 4. Simulate active Deepfake Attack (Escalates to Red Alert Badge)
python run_callguard.py --synthetic --simulate-deepfake --iterations 10

# 5. Verify cryptographic SHA-256 audit chain
python run_callguard.py --verify-audit
```
