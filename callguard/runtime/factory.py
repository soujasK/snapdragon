"""
Unified Runtime Provider Factory: QNN (Snapdragon NPU) to CPU Fallback.

Architectural Design:
    Provides a single centralized factory for ONNX Runtime session creation across all
    CallGuard stages, ensuring seamless execution on Snapdragon Hexagon NPUs with transparent
    CPU fallback on non-Snapdragon machines. Logs the exact execution provider actually
    assigned to each model at runtime and enforces strict telemetry honesty.
"""

import os
import sys
import platform
import time
from typing import Dict, List, Optional, Any, Tuple
import onnxruntime as ort

QNN_EP = "QNNExecutionProvider"
CPU_EP = "CPUExecutionProvider"


class CallGuardSession:
    """
    Thin wrapper over onnxruntime.InferenceSession tracking actual execution provider
    and execution latency without fabricating unmeasured performance metrics.
    """

    def __init__(self, model_path: str, sess: ort.InferenceSession, requested_provider: str, active_provider: str):
        self.model_path = model_path
        self.sess = sess
        self.requested_provider = requested_provider
        self.active_provider = active_provider
        self.last_latency_ms: float = 0.0

        # Cache input/output names
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def run(self, input_feed: Dict[str, Any]) -> List[Any]:
        t0 = time.perf_counter()
        outputs = self.sess.run(self.output_names, input_feed)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return outputs


class CallGuardRuntimeFactory:
    """
    Centralized execution provider factory.
    Attempts QNN (Hexagon HTP) on Snapdragon Windows ARM64 platforms, falling back
    cleanly to CPU on x64 test benches or when QNN drivers are absent.
    """

    _registry: Dict[str, CallGuardSession] = {}

    @classmethod
    def get_hardware_info(cls) -> Dict[str, Any]:
        """Inspect host architecture and available ONNX Runtime providers."""
        arch = platform.machine().lower()
        is_arm64 = "arm64" in arch or "aarch64" in arch
        avail = list(ort.get_available_providers())
        has_qnn = QNN_EP in avail

        return {
            "platform": sys.platform,
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "is_windows_arm64": (sys.platform == "win32" and is_arm64),
            "ort_version": ort.__version__,
            "available_providers": avail,
            "npu_accelerator_available": has_qnn,
            "default_backend": QNN_EP,
        }

    @classmethod
    def create_session(
        cls,
        model_path: str,
        prefer: str = "auto",
        qnn_backend_path: Optional[str] = None,
        intra_threads: int = 4,
    ) -> CallGuardSession:
        """
        Creates an InferenceSession trying QNN first, then CPU fallback.
        Logs the exact provider assigned by ONNX Runtime.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        avail_providers = ort.get_available_providers()
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = intra_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        chosen_providers = []
        provider_options = []
        requested = "CPUExecutionProvider"

        # QNNExecutionProvider path targeting Qualcomm Hexagon HTP
        # Configured for Snapdragon-powered HP PCs running Qualcomm ONNX Runtime QNN provider.
        if prefer in ("auto", "qnn") and QNN_EP in avail_providers:
            qnn_opts = {
                "backend_path": qnn_backend_path or "QnnHtp.dll",
                "htp_performance_mode": "burst",
                "htp_precision": "float16",
            }
            chosen_providers.append(QNN_EP)
            provider_options.append(qnn_opts)
            requested = QNN_EP

        # Always append CPU fallback
        chosen_providers.append(CPU_EP)
        provider_options.append({})

        # Instantiate ORT session
        try:
            sess = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=chosen_providers,
                provider_options=provider_options,
            )
        except Exception as exc:
            # Fall back to pure CPU
            print(f"[CallGuard Runtime] Provider {chosen_providers[0]} failed ({exc}), falling back to CPU.")
            sess = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=[CPU_EP],
            )
            requested = CPU_EP

        # Check what provider ORT actually assigned
        assigned = sess.get_providers()[0] if sess.get_providers() else "Unknown"
        model_name = os.path.basename(model_path)
        print(f"[CallGuard Runtime] Model '{model_name}' -> Assigned Provider: '{assigned}' (requested: '{requested}')")

        wrapper = CallGuardSession(
            model_path=model_path,
            sess=sess,
            requested_provider=requested,
            active_provider=assigned,
        )
        cls._registry[model_name] = wrapper
        return wrapper

    @classmethod
    def get_registered_sessions(cls) -> Dict[str, str]:
        """Returns map of {model_name: active_provider} for all active sessions."""
        return {name: s.active_provider for name, s in cls._registry.items()}
