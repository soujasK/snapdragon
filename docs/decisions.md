# CallGuard NPU - Architecture & Engineering Decisions Log

This document records the architectural, engineering, and security decisions governing the design of **CallGuard NPU**.

---

### Decision 001: Self-Contained Native Package Architecture
- **Date**: 2026-09-30
- **Context**: Video calling security requires rapid startup, low-overhead execution, and zero dependency conflicts.
- **Decision**: Structure CallGuard NPU as a unified native Python package with self-contained subsystems:
  - `callguard.biometrics`: Physiological rPPG cardiac extraction and acoustic forensic detectors.
  - `callguard.vision`: Video frame processing, CLAHE enhancement, and dual-backend image operations.
  - `callguard.speech`: VAD-gated speech transcription and real-time captioning.
  - `callguard.core`: Evidence contracts, claim translation, triadic debate, and cryptographic audit logging.
  - `callguard.runtime`: Hardware abstraction, ONNX session management, and priority NPU queue scheduling.
  - `callguard.capture`: Thread-safe, bounded, timestamped circular ring buffers.
  - `callguard.ui`: Overlay HUD and trust badges.
- **Rationale**: Completely eliminates fragile `sys.path` hacks, name collisions, or reliance on external repository structures.

---

### Decision 002: Snapdragon-Powered HP Hardware Target & Dual Execution Strategy
- **Date**: 2026-09-30
- **Context**: The target hardware is Qualcomm Snapdragon X (Windows ARM64) with Hexagon NPU offload on Snapdragon-powered HP PCs (HP OmniBook X), with portable CPU fallback for CI workstations.
- **Decision**:
  - Implement and verify physical Qualcomm QNN Execution Provider configuration (`QNNExecutionProvider` with `QnnHtp.dll`) targeting Snapdragon X Elite Hexagon NPU.
  - Pair with transparent dynamic fallback to CPU (`CPUExecutionProvider`) on non-Snapdragon systems.
  - Provide a File-Input Mode (`--video <file> --audio <file>`) and Synthetic Stream Generator (`--synthetic`) to enable automated headless pipeline verification.

---

### Decision 003: Native Dual-Backend Image Operations for Windows ARM64
- **Date**: 2026-09-30
- **Context**: Windows ARM64 lacks official pre-compiled binary wheels for `opencv-python`.
- **Decision**:
  - Implement a dual-backend image operations module (`callguard.vision.imgops`) that prefers OpenCV when available, but automatically provides pure PIL + SciPy + NumPy implementations for all image operations (resizing, color conversions, CLAHE, Gaussian blur).
  - This ensures CallGuard runs natively out-of-the-box on Snapdragon X laptops without requiring complex custom OpenCV builds.

---

### Decision 004: Priority NPU Queue Scheduling & Progressive Degradation Hierarchy
- **Date**: 2026-09-30
- **Context**: The Qualcomm Hexagon NPU is a shared resource with strict thermal and compute boundaries. Total duty cycle must not exceed 60% per second ($\sum (\text{cost} \times \text{rate}) \le 0.60$).
- **Decision**:
  - Implement `PriorityScheduler` with an explicit degradation order under thermal or compute load:
    1. Throttle Super-Resolution (from 30Hz -> 10Hz -> pause)
    2. Throttle Low-Light Enhancement (from 30Hz -> 15Hz -> pause)
    3. Throttle Deepfake detection cadence (run every 2nd or 4th frame)
  - Critical Invariant: **Landmark tracking and Speech Captions are protected and NEVER throttled**, preserving face tracking stability and user accessibility.

---

### Decision 005: Defender Environmental Context Discounting & Calibrated Log-Odds Fusion
- **Date**: 2026-09-30
- **Context**: Real-world video calls suffer from poor lighting (low-light sensor noise), network packet loss (stutter), and compression artifacts (bitrate drops), which often trigger false-positive deepfake detections.
- **Decision**:
  - Implement a triadic debate engine (Prosecutor, Defender, Judge) in `CallGuardDebateEngine`.
  - Equip the Defender with domain rules that check environmental flags (`low_light`, `bitrate_drop`, `stutter`, `multi_face`) extracted from the enhancement path.
  - When an anomaly is accounted for by an active environmental flag (e.g. low-light sensor noise explaining lack of capillary rPPG pulse), the Defender discounts the claim's severity weight (e.g. factor 0.25).
  - Fuse claims using mathematically calibrated log-odds Bayesian updates: $L = L_0 + \sum \Delta L_i$, mapped to calibrated posterior probability $P_{\text{synthetic}} = \frac{1}{1 + e^{-L}}$.

---

### Decision 006: Asymmetric Temporal Hysteresis Filter for Zero-Trust Badging
- **Date**: 2026-09-30
- **Context**: Instantaneous single-frame probabilities fluctuate rapidly, which would cause disorienting UI badge flickering between "Consistent" and "Likely Synthetic".
- **Decision**:
  - Implement `HysteresisFilter` maintaining a rolling 5-window history.
  - Transition into `LIKELY SYNTHETIC` alert requires $P \ge 0.75$ in at least 3 of 5 windows.
  - Recovery/exit from alert back to `CONSISTENT` requires $P \le 0.50$ in at least 4 of 5 windows.
  - Intermediate confidence states report `UNCERTAIN`.

---

### Decision 007: Cryptographic SHA-256 Hash Chaining on SQLite Audit Trail
- **Date**: 2026-09-30
- **Context**: Standard relational database tables are vulnerable to retrospective tampering or log record deletion.
- **Decision**:
  - Integrate SHA-256 cryptographic hash-chaining into every evaluation window recorded in `callguard_audit.db`: `current_hash = SHA256(row_id | timestamp | window_id | claims | prior_prob | posterior_prob | verdict | confidence | prev_hash)`.
  - Provide `verify_chain()` CLI command and programmatic method to detect any row modification, reordering, or tampering attempt.
