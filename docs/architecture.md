# CallGuard NPU - System Architecture

CallGuard NPU is a local, privacy-first zero-trust verification and assistive video calling system developed and verified for Snapdragon-powered HP PCs (Windows ARM64 / Qualcomm Hexagon NPU) with seamless CPU fallback for portable testing.

It integrates five specialized native capabilities into a unified real-time pipeline:
1. **Biometric & Acoustic Liveness Verification** (`callguard.biometrics`)
2. **Video Enhancement & Environmental Quality Telemetry** (`callguard.vision`)
3. **Real-Time VAD-Gated Speech-to-Text Captions** (`callguard.speech`)
4. **Adversarial Triadic Debate & Bayesian Fusion** (`callguard.core`)
5. **Cryptographic SHA-256 Audit Trail & Priority Hardware Scheduling** (`callguard.core`, `callguard.runtime`)

---

## 1. High-Level Data Flow Diagram

```mermaid
flowchart TD
    subgraph INGEST ["Multi-Modal Ingest Subsystem"]
        WIN_CAP["Windows Graphics Capture / Camera"] -->|Raw Video Frames| VRING["Timestamped Ring Buffer (Video)\n(Drop-Oldest, Monotonic Clock)"]
        WASAPI["WASAPI Loopback / Mic"] -->|16kHz PCM Audio| ARING["Timestamped Ring Buffer (Audio)\n(Drop-Oldest, Monotonic Clock)"]
    end

    subgraph ENHANCE ["Display Enhancement Path (Display Copy Only)"]
        VRING -->|Display Copy| ENH["EnhancePathPipeline\n(Face, Blur, Low-Light, Super-Res)"]
        ENH -->|Enhanced BGR| DISPLAY["Local UI Display Copy"]
        ENH -.->|Telemetry Flags:\nlow_light, bitrate_drop,\nstutter, multi_face| CTX["Environmental Quality Context (ctx)"]
    end

    subgraph RAWTAP ["RAW Tap Pipeline (Strictly Unenhanced Media)"]
        VRING -->|Raw Frames ONLY| RPPG["FaceMesh & rPPG Engine\n(BVP, PSD, Pulse SNR)"]
        ARING -->|Raw Audio ONLY| ACOUSTIC["Acoustic Voice Detector\n(Spectral Flatness, Cutoff, Jitter)"]
        RPPG --> EV_RAW["Evidence Generator"]
        ACOUSTIC --> EV_RAW
        CTX -.->|Attach Context| EV_RAW
        EV_RAW -->|Evidence Contract| CLAIMS["Claim Adapter\n(Polarized Forensic Propositions)"]
    end

    subgraph CAPTIONS ["Assistive Captions Subsystem"]
        ARING -->|Audio Stream| VAD{"VAD Gate\n(RMS / Silence)"}
        VAD -->|Speech Detected| WHISPER["Whisper ONNX / DemoEngine\n(Protected Priority)"]
        VAD -->|Silence| IDLE["Skip Inference\n(Preserve Duty Cycle)"]
        WHISPER --> HUD_CAP["Live Caption Stream"]
    end

    subgraph DEBATE ["Adversarial Debate & Decision Logic"]
        CLAIMS --> PROS["Prosecutor Agent\n(Argues Anomaly / Manipulation)"]
        CLAIMS --> DEF["Defender Agent\n(Discounts Artifacts Explained by ctx)"]
        CTX -.->|Explain Anomaly| DEF
        PROS --> JUDGE["Judge Agent\n(Calibrated Bayesian Log-Odds Fusion)"]
        DEF --> JUDGE
        JUDGE -->|P(Synthetic)| HYST["Asymmetric Hysteresis State Machine\n(Enter >=0.75 in 3/5, Exit <=0.50 in 4/5)"]
        HYST --> BADGE["Trust Badge:\n[CONSISTENT] / [UNCERTAIN] / [LIKELY SYNTHETIC]"]
    end

    subgraph AUDIT ["Audit Trail Subsystem"]
        JUDGE --> AUDIT_DB["SQLite Audit DB (WAL Mode)"]
        BADGE --> AUDIT_DB
        AUDIT_DB --> HASH_CHAIN["SHA-256 Hash Chain\n(Tamper-Evident Row Linkage)"]
    end

    subgraph SCHEDULER ["Priority NPU Scheduler & Telemetry"]
        SCHED["Priority NPU Scheduler\nBudget: sum(cost * rate) <= 60%"]
        SCHED -->|1. Throttle Super-Res| ENH
        SCHED -->|2. Throttle Low-Light| ENH
        SCHED -->|3. Throttle Deepfake Cadence| RAWTAP
        SCHED -.->|PROTECTED: Never Throttle| WHISPER
        SCHED -.->|PROTECTED: Never Throttle| RPPG
    end

    subgraph UI ["User Interface"]
        BADGE --> OVERLAY["CallGuard Overlay HUD"]
        HUD_CAP --> OVERLAY
        DISPLAY --> OVERLAY
        HASH_CHAIN -.->|Verification Status| OVERLAY
        OVERLAY --> ADVISORY["Advisory Notice:\n'Verdict text in UI is advisory'"]
    end
```

---

## 2. Core Architectural Principles

### 2.1 Isolation of RAW Tap vs. Display Enhancement
- **The RAW Tap Constraint**: The deepfake detection and acoustic voice authenticity algorithms (`callguard.biometrics`) read **pristine, unenhanced media buffers only**.
- Enhancement filters (Zero-DCE/CLAHE low-light boost, background segmentation blur, and QuickSRNet super-resolution) modify pixel distributions, destroy capillary hemodynamic micro-flushes, and alter frequency bands. Consequently, enhancement runs strictly on a secondary display copy.

### 2.2 Environmental Context Extraction & Defender Discounting
A critical innovation of CallGuard is cross-subsystem feedback:
- The enhancement path measures camera luminance, frame arrival jitter, and video compression artifacts.
- It exports four boolean context flags:
  1. `low_light`: Low ambient illumination (mean luminance < 60).
  2. `bitrate_drop`: Compression artifacts / sudden loss of spatial high frequencies.
  3. `stutter`: Network packet loss / temporal frame delta irregularities ($\Delta t > 1.6 \times \text{expected}$).
  4. `multi_face`: Multiple faces in the frame.
- When an anomaly is detected (e.g. low rPPG SNR or missing high-frequency vocal formants), the **Defender Agent** checks these flags. If the ambient context explains the anomaly, the claim's severity is discounted, eliminating false alarms.

### 2.3 Calibrated Log-Odds Bayesian Fusion
Rather than naive thresholding or unweighted averaging:
- Baseline prior log-odds: $L_0 = \ln\left(\frac{P_0}{1 - P_0}\right)$ with $P_0 = 0.10$.
- Each claim adds or subtracts weighted log-odds: $\Delta L_i = \text{polarity} \times \text{weight} \times \text{confidence} \times \text{discount}$.
- Final posterior probability: $P_{\text{synthetic}} = \frac{1}{1 + e^{-L}}$.

### 2.4 Asymmetric Temporal Hysteresis
To prevent disorienting badge twitching:
- **Enter Alert (`LIKELY SYNTHETIC`)**: Requires $P \ge 0.75$ in at least 3 of the last 5 evaluation windows.
- **Exit Alert (`CONSISTENT`)**: Requires $P \le 0.50$ in at least 4 of the last 5 evaluation windows.
- **Transitional (`UNCERTAIN`)**: Displayed while in intermediate confidence bands.

### 2.5 Cryptographic SHA-256 Hash Chaining
Every evaluation window appends a record to `callguard_audit.db` with a cryptographic link:
$$\text{current\_hash} = \text{SHA256}(\text{row\_id} \parallel \text{timestamp} \parallel \text{window\_id} \parallel \text{claims} \parallel \text{prior\_p} \parallel \text{post\_p} \parallel \text{verdict} \parallel \text{confidence} \parallel \text{prev\_hash})$$
Any retroactive modification to prior audit entries invalidates the hash chain and is flagged immediately.

### 2.6 NPU Duty-Cycle Budgeting & Progressive Throttling
The Hexagon NPU is constrained to a maximum 60% compute budget per second:
$$\sum_{s \in \text{stages}} (\text{latency}_s \times \text{rate}_s) \le 0.60$$
When load spikes:
1. **Super-Resolution** throttles from 30Hz $\to$ 10Hz $\to$ 0Hz.
2. **Low-Light Enhancement** throttles from 30Hz $\to$ 15Hz $\to$ 0Hz.
3. **Deepfake Cadence** throttles to every 2nd or 4th frame.
4. **Speech Captions and Face Landmarks are PROTECTED and NEVER throttled.**
