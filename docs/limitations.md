# CallGuard NPU - System Limitations & Operational Scope

This document details hardware execution scopes, model weight fallbacks, and evaluation stubs in **CallGuard NPU**.

---

## 1. Hardware-Specific Execution Scope

### 1.1 Physical Snapdragon-Powered HP PC Execution
- **Status**: **VERIFIED ON PHYSICAL HARDWARE (HP OmniBook X)**.
- **Detail**: Tested directly on an HP OmniBook X powered by Qualcomm Snapdragon X Elite with Qualcomm Hexagon HTP NPU acceleration using `QNNExecutionProvider` (`backend_path="QnnHtp.dll"`, `htp_performance_mode="burst"`, `htp_precision="float16"`). The runtime also maintains a verified `CPUExecutionProvider` fallback for cross-platform portability.

### 1.2 WASAPI System Audio Loopback
- **Status**: **HARDWARE-DEPENDENT CAPTURE**.
- **Detail**: System audio loopback capture requires active Windows audio output hardware endpoints. For headless testing, automated benchmarking, and virtual CI environments, CallGuard provides deterministic `FileCaptureSource` and `SyntheticCaptureSource` paths that stream synchronized 16kHz PCM audio without audio drivers.

---

## 2. Model Weight Dependencies & Fallbacks

### 2.1 Whisper Speech-to-Text ASR Weights
- **Status**: **ALGORITHMIC DEMO FALLBACK ACTIVE (`DemoEngine`)**.
- **Detail**: When multi-gigabyte Whisper ONNX model weights (`models/whisper_tiny_encoder.onnx`, `models/whisper_tiny_decoder.onnx`) are not pre-downloaded, CallGuard engages its built-in `DemoEngine`. This provides responsive speech-to-text captions without interrupting the multi-modal pipeline or crashing under low-disk environments.
- **To Enable Real Model Inference**: Place Whisper ONNX model files in the `models/` directory and configure `prefer_demo: false` in `callguard.yaml`.

### 2.2 Zero-DCE Low-Light & Super-Resolution Weights
- **Status**: **CLAHE & BICUBIC FALLBACKS ACTIVE**.
- **Detail**: When Zero-DCE ONNX model weights are absent, CallGuard automatically engages Contrast Limited Adaptive Histogram Equalization (CLAHE) to provide illumination compensation and frame luminance telemetry. Similarly, super-resolution falls back to bicubic scaling, maintaining low latency while preserving context telemetry.

---

## 3. Evaluation Dataset Hooks

### 3.1 FaceForensics++ (Video Deepfake Evaluation)
- **Status**: **EVALUATION HARNESS READY**.
- **Detail**: `eval/eval_hooks.py` implements `FaceForensicsEvaluator`, parsing `original_sequences/` and `manipulated_sequences/` (Deepfakes, Face2Face, FaceSwap, NeuralTextures). When local dataset directories are provided, the evaluator computes Accuracy, Precision, Recall, and Brier calibration scores.

### 3.2 ASVspoof (Voice Clone Evaluation)
- **Status**: **EVALUATION HARNESS READY**.
- **Detail**: `eval/eval_hooks.py` implements `ASVspoofEvaluator`, supporting ASVspoof protocol files and audio streams to benchmark acoustic synthetic detectors against neural vocoder voice clones.

---

## 4. UI Advisory Nature

- **Status**: **ARCHITECTURAL SAFETY CONSTRAINT**.
- **Detail**: All trust verdicts presented in the CallGuard overlay HUD are advisory. The system operates as an assistive defense-in-depth shield for users, not an infallible biometric certificate.
