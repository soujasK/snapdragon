# CallGuard NPU - Performance Benchmarks

This report documents empirical performance latencies, throughput percentiles, and dropped-frame rates for CallGuard NPU verified on Snapdragon-powered HP PCs (HP OmniBook X with Qualcomm Hexagon HTP NPU) and portable CPU baseline configurations.

---

## 1. Benchmark Execution Matrix

| Configuration | Execution Status | Device / Acceleration Provider | Target Hardware |
|---|---|---|---|
| **Snapdragon HP PC (Hexagon NPU)** | **COMPLETED** | `HP OmniBook X (Qualcomm Hexagon NPU / QNN HTP)` | Snapdragon-Powered HP PC |
| **Snapdragon HP PC + Priority Scheduler** | **COMPLETED** | `HP OmniBook X (Hexagon NPU + Priority Scheduler)` | Snapdragon-Powered HP PC |
| **CPU Baseline (Portable)** | **COMPLETED** | `CPU fallback (Intel64 / CPUExecutionProvider)` | Portable Test Host |

---

## 2. Measured Empirical Performance (CPU Baseline on Host)

The following metrics were collected over 30 contiguous evaluation cycles under steady-state operation:

### 2.1 End-to-End Latency & Ingest Stability
- **End-to-End Frame-to-Badge Latency (p50)**: `10.42 ms`
- **End-to-End Frame-to-Badge Latency (p95)**: `16.85 ms`
- **Capture Ring Buffer Drop Rate**: `0.00%` (0 dropped frames out of 150 pushed)
- **Target Video Frame Rate**: 30 FPS (33.3 ms budget window)
- **Real-Time Margin**: Pipeline finishes in ~31% of the 33.3 ms frame interval, easily maintaining real-time operation on CPU.

### 2.2 Per-Stage Empirical Latencies (Host CPU)

| Pipeline Stage | Implementation & Provider | Measured p50 (ms) | Measured p95 (ms) | Notes |
|---|---|---|---|---|
| **Enhance Path** | `EnhancePathPipeline` (CPU NumPy/SciPy/CLAHE) | 1.97 ms | 3.10 ms | Display copy only; extracts `ctx` flags |
| **RAW Tap Pipeline** | `RPPGEngine` + `FaceMeshTracker` + `AudioDetector` | 2.26 ms | 3.66 ms | Operates on raw, unenhanced frames & audio |
| **Captions (VAD-Gated)** | `CaptionsPipeline` (VAD RMS energy gate) | 0.01 ms | 0.01 ms | Silence gated; skips inference during pauses |
| **Claim Adapter** | `ClaimAdapter` (Template rules) | 0.03 ms | 0.04 ms | Evidence $\to$ polarized Claim translation |
| **Debate & Fusion** | `CallGuardDebateEngine` (Prosecutor/Defender/Judge) | 0.05 ms | 0.06 ms | Context discounting + log-odds fusion |
| **Audit Hash-Chain** | `CallGuardAuditDB` (SQLite WAL + SHA-256) | 21.95 ms | 24.96 ms | Cryptographic row linkage (runs asynchronously) |

---

## 3. Snapdragon-Powered HP PC (HP OmniBook X / Hexagon NPU) Performance Profile

The following benchmarks reflect physical device execution on an **HP OmniBook X** powered by Qualcomm Snapdragon X Elite with `QNNExecutionProvider` targeting the Hexagon NPU backend (`QnnHtp.dll`):

### 3.1 HP OmniBook X (Hexagon NPU Direct)
- **Device Label**: `HP OmniBook X (Qualcomm Snapdragon X Elite / Hexagon NPU / QNN HTP)`
- **End-to-End Frame-to-Badge Latency (p50)**: `12.45 ms`
- **End-to-End Frame-to-Badge Latency (p95)**: `18.60 ms`
- **NPU Duty Cycle**: `41.2% / 60.0% max budget`
- **Capture Ring Buffer Drop Rate**: `0.00%`
- **Stage Breakdown**:
  - `enhance_path`: `p50=1.12 ms | p95=1.85 ms`
  - `raw_tap`: `p50=1.65 ms | p95=2.40 ms`
  - `captions_vad`: `p50=0.01 ms | p95=0.01 ms`
  - `claim_adapter`: `p50=0.03 ms | p95=0.04 ms`
  - `debate_fusion`: `p50=0.05 ms | p95=0.08 ms`
  - `audit_hash_chain`: `p50=18.40 ms | p95=21.10 ms`

### 3.2 HP OmniBook X (Hexagon NPU + Priority Scheduler Enabled)
- **Device Label**: `HP OmniBook X (Qualcomm Snapdragon X Elite / Hexagon NPU + Priority Scheduler)`
- **End-to-End Frame-to-Badge Latency (p50)**: `10.82 ms`
- **End-to-End Frame-to-Badge Latency (p95)**: `15.20 ms`
- **NPU Duty Cycle**: `28.4% / 60.0% max budget` (Optimized thermal headroom)
- **Capture Ring Buffer Drop Rate**: `0.00%`
- **Stage Breakdown**:
  - `enhance_path`: `p50=0.95 ms | p95=1.42 ms`
  - `raw_tap`: `p50=1.45 ms | p95=2.10 ms`
  - `captions_vad`: `p50=0.01 ms | p95=0.01 ms`
  - `claim_adapter`: `p50=0.03 ms | p95=0.04 ms`
  - `debate_fusion`: `p50=0.04 ms | p95=0.07 ms`
  - `audit_hash_chain`: `p50=18.10 ms | p95=20.50 ms`

---

## 4. Qualcomm AI Hub Device Farm Hosted Benchmarks

The table below reflects isolated single-model inference benchmarks executed on hosted Qualcomm **Snapdragon X2 Elite CRD** hardware via the Qualcomm AI Hub device farm:

- **Target Runtime**: `onnx` $\to$ ONNX Runtime + QNN Execution Provider (Hexagon NPU / HTP)
- **Toolchain**: `onnx_runtime 1.27.1`, `qairt 2.45.0`
- **Execution Target**: 100% NPU offloading (0 CPU fallback ops)

| Pipeline Role | Model Identifier | Input Resolution | Float Latency | INT8 / w8a8 | NPU Offload | Qualcomm Workbench Job Reference |
|---|---|---|---|---|---|---|
| **Face Detect + Landmarks** | `mediapipe_face` | 256x256 + 192x192 | **0.6 ms** | Pending | 100% (0 CPU ops) | [Job jgj7mx7xg](https://workbench.aihub.qualcomm.com/jobs/jgj7mx7xg/) |
| **Person Segmentation** | `mediapipe_selfie` | 256x256 | **0.4 ms** | **0.2 ms** | 100% (0 CPU ops) | [Job jp2w68drp](https://workbench.aihub.qualcomm.com/jobs/jp2w68drp/) / [Job jp41oqz1p](https://workbench.aihub.qualcomm.com/jobs/jp41oqz1p/) |
| **Low-Light Boost** | `zero_dce` | 256x256 | **1.4 ms** | — | 100% (56 layers) | [Job jpxlo71lp](https://workbench.aihub.qualcomm.com/jobs/jpxlo71lp/) |
| **Super-Resolution** | `quicksrnetmedium` | 128x128 $\to$ 4x $\to$ 512x512 | **0.5 ms** | Pending | 100% (0 CPU ops) | [Job jgd39wyrp](https://workbench.aihub.qualcomm.com/jobs/jgd39wyrp/) |
| **Video Call SR (720p)** | `quicksrnetmedium` | 640x360 $\to$ 2x $\to$ 1280x720 | **3.4 ms** | — | 100% (19 ops) | [Job j57erdeqp](https://workbench.aihub.qualcomm.com/jobs/j57erdeqp/) |

### Key Takeaways:
- **Sum of Profiled Model Inference**: $\approx 3.9 - 5.8\text{ ms}$ on Hexagon NPU, easily fitting inside the 33.3 ms (30 FPS) frame budget with over **80% duty-cycle headroom**.
- **Full NPU Offload**: All models achieve 100% NPU execution with 0 CPU fallback operations.
