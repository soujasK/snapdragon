"""
Benchmark Suite: Stage Latencies, End-to-End Latencies, and Dropped-Frame Rates.

Design:
    Collects measured, empirical performance metrics across CPU-only, NPU, and NPU+scheduler
    configurations. Strictly tags unsupported hardware configurations as 'not run' and enforces
    honest labeling ('measured on <device>' or 'CPU fallback').
"""

import time
import platform
import pytest
from callguard.pipeline import CallGuardPipeline
from callguard.capture.sources import SyntheticCaptureSource
from callguard.runtime.factory import CallGuardRuntimeFactory
from callguard.config import load_config


def run_benchmark_configuration(config_name: str, use_scheduler: bool = True, iterations: int = 30):
    hw_info = CallGuardRuntimeFactory.get_hardware_info()
    has_npu = hw_info["npu_accelerator_available"]

    # When running on physical Snapdragon hardware
    if has_npu:
        cfg = load_config()
        source = SyntheticCaptureSource(fps=30)
        pipeline = CallGuardPipeline(config=cfg, source=source)
        pipeline.start()
        try:
            for _ in range(iterations):
                pipeline.process_step()
                time.sleep(0.01)
        finally:
            pipeline.stop()
        summary = pipeline.telemetry.get_summary()
        ring_stats = pipeline.video_buffer.get_stats()
        return {
            "config": config_name,
            "status": "completed",
            "device_label": summary["device_label"],
            "stages": summary["stages"],
            "end_to_end_p50_ms": summary["end_to_end_frame_to_badge"]["p50_ms"],
            "end_to_end_p95_ms": summary["end_to_end_frame_to_badge"]["p95_ms"],
            "dropped_frame_rate": ring_stats["drop_rate"],
        }

    # CPU-only baseline on host
    if config_name == "CPU-only":
        cfg = load_config()
        source = SyntheticCaptureSource(fps=30)
        pipeline = CallGuardPipeline(config=cfg, source=source)
        pipeline.start()
        try:
            for _ in range(iterations):
                pipeline.process_step()
                time.sleep(0.01)
        finally:
            pipeline.stop()
        summary = pipeline.telemetry.get_summary()
        ring_stats = pipeline.video_buffer.get_stats()
        return {
            "config": config_name,
            "status": "completed",
            "device_label": summary["device_label"],
            "stages": summary["stages"],
            "end_to_end_p50_ms": summary["end_to_end_frame_to_badge"]["p50_ms"],
            "end_to_end_p95_ms": summary["end_to_end_frame_to_badge"]["p95_ms"],
            "dropped_frame_rate": ring_stats["drop_rate"],
        }

    # When physical Snapdragon QNN hardware is not present on this host
    return {
        "config": config_name,
        "status": "not run",
        "reason": f"Physical Snapdragon QNN HTP hardware not detected on this host ({platform.machine()}).",
        "device_label": "UNTESTED ON DEVICE",
        "stages": {},
        "end_to_end_p50_ms": None,
        "end_to_end_p95_ms": None,
        "dropped_frame_rate": None,
    }


def test_benchmarks_all_configs():
    # 1. CPU Baseline Configuration (Empirically measured on host)
    cpu_res = run_benchmark_configuration("CPU-only", use_scheduler=False, iterations=25)
    assert cpu_res["status"] == "completed"
    assert cpu_res["end_to_end_p50_ms"] is not None
    print(f"\n[Benchmark - CPU Baseline (Measured on Host)]:")
    print(f"  Device Label          : {cpu_res['device_label']}")
    print(f"  End-to-End p50 Latency: {cpu_res['end_to_end_p50_ms']} ms")
    print(f"  End-to-End p95 Latency: {cpu_res['end_to_end_p95_ms']} ms")
    print(f"  Dropped Frame Rate    : {cpu_res['dropped_frame_rate'] * 100:.2f}%")
    for st, data in cpu_res["stages"].items():
        print(f"    - {st:<18}: p50={data['p50_ms']} ms, p95={data['p95_ms']} ms")

    # 2. Physical Snapdragon NPU Configuration
    npu_res = run_benchmark_configuration("NPU (QNN HTP)", use_scheduler=False, iterations=10)
    print(f"\n[Benchmark - Physical Snapdragon NPU]: Status = {npu_res['status']} ({npu_res.get('reason', 'completed')})")

    # 3. Physical Snapdragon NPU + Priority Scheduler Configuration
    npu_sched_res = run_benchmark_configuration("NPU + Priority Scheduler", use_scheduler=True, iterations=10)
    print(f"[Benchmark - Snapdragon NPU + Scheduler]: Status = {npu_sched_res['status']} ({npu_sched_res.get('reason', 'completed')})")
    print("\n[Notice]: Isolated model benchmarks on Snapdragon X hardware are profiled via Qualcomm AI Hub hosted devices (see docs/benchmarks.md).")
