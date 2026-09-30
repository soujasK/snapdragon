# CallGuard NPU - Performance Benchmarks

This report documents empirical performance latencies, throughput percentiles, and dropped-frame rates for CallGuard NPU. In accordance with strict measurement integrity standards, all metrics reflect strictly empirical data: host CPU fallback measurements taken during local execution, and cloud-hosted device profiling conducted on the Qualcomm AI Hub device farm.

> [!IMPORTANT]
> **Measurement Integrity Policy**: No unmeasured or hypothetical local NPU performance numbers are printed or stored. All measurements are explicitly labeled with the executing device hardware backend (`CPU fallback` on test host vs. `AI Hub hosted-device profiling` on Qualcomm test racks). Physical local execution on consumer Snapdragon X laptops remains **UNTESTED ON DEVICE**. NEVER label any AI Hub cloud number as end-to-end NPU performance.

---

## 1. Benchmark Execution Matrix

| Configuration | Execution Status | Device / Acceleration Provider | Target Hardware |
|---|---|---|---|
| **CPU Baseline (Local Host)** | **COMPLETED** | `CPU fallback (Intel64 / CPUExecutionProvider)` | Development / CI Host |
| **Qualcomm AI Hub Hosted Profiling** | **COMPLETED** | `Snapdragon X2 Elite CRD (Qualcomm Hexagon NPU / QNN HTP)` | Qualcomm AI Hub Device Farm |
| **Physical Snapdragon Laptop** | **UNTESTED ON DEVICE** | `QNNExecutionProvider` (`backend_path="QnnHtp.dll"`) | Snapdragon-Powered HP PC |

---

## 2. Measured Empirical Performance (CPU Baseline on Host)

The following metrics were collected over 30 contiguous evaluation cycles under steady-state operation on the host system:

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

## 3. Qualcomm AI Hub Hosted-Device Profiling (Snapdragon X2 Elite CRD)

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
- **Full NPU Offload**: All models achieve 100% NPU execution with 0 CPU fallback operations on hosted Qualcomm hardware.
- **Architectural Scope**: This sum represents isolated model inference on Qualcomm AI Hub cloud test racks. It excludes local frame capture, pre/post-processing, and compositing, and is strictly **not** labeled as local end-to-end device performance.
