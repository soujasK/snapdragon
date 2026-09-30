"""
Qualcomm AI Hub Hosted Device Model Compilation & Profiling Tool.

Design:
    Provides an automated harness to compile and profile CallGuard's models on hosted Qualcomm
    Snapdragon X devices via the Qualcomm AI Hub cloud API, writing verified per-model metrics
    to docs/benchmarks.md.
"""

import os
import sys
import datetime as dt
import pathlib
import re
import subprocess
from typing import Dict, Any, Optional, List

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
BENCHMARKS_DOC = ROOT / "docs" / "benchmarks.md"

# Target Snapdragon X devices on Qualcomm AI Hub
TARGET_DEVICES = ["Snapdragon X Elite CRD", "Snapdragon X2 Elite CRD"]
DEFAULT_DEVICE = "Snapdragon X Elite CRD"

# Models to compile and profile via AI Hub Zoo
MODELS = {
    "face_detector": "mediapipe_face",
    "person_segmentation": "mediapipe_selfie",
    "super_resolution": "quicksrnetmedium",
}

_LAT_REGEX = re.compile(r"inference time \(ms\)\s*:\s*([\d.]+)", re.I)
_NPU_OPS_REGEX = re.compile(r"npu \((\d+) ops\) gpu \((\d+) ops\) cpu \((\d+) ops\)", re.I)
_JOB_URL_REGEX = re.compile(r"(https://\S*aihub\S*\.qualcomm\.com/\S+)", re.I)


def check_api_token() -> Optional[str]:
    """Retrieve QAI_HUB_API_TOKEN from environment."""
    token = os.environ.get("QAI_HUB_API_TOKEN", "").strip()
    if not token:
        return None
    return token


def parse_export_output(output_text: str) -> Dict[str, Any]:
    """Extract inference latency, compute unit ops, and AI hub job URLs from CLI output."""
    latencies = [float(x) for x in _LAT_REGEX.findall(output_text)]
    cus = [(int(n), int(g), int(c)) for n, g, c in _NPU_OPS_REGEX.findall(output_text)]
    total_lat = round(sum(latencies), 2) if latencies else None

    npu_frac = None
    if cus:
        npu_ops = sum(n for n, _, _ in cus)
        total_ops = sum(n + g + c for n, g, c in cus)
        npu_frac = round(100.0 * npu_ops / total_ops, 1) if total_ops > 0 else None

    job_urls = list(dict.fromkeys(_JOB_URL_REGEX.findall(output_text)))
    return {
        "latency_ms": total_lat,
        "npu_percent": npu_frac,
        "job_urls": job_urls,
    }


def update_benchmarks_doc(results: List[Dict[str, Any]], skipped_reason: Optional[str] = None):
    """Appends or updates the AI Hub cloud-profiled section in docs/benchmarks.md."""
    if not BENCHMARKS_DOC.exists():
        print(f"[AI Hub Profile] Warning: {BENCHMARKS_DOC} not found.")
        return

    content = BENCHMARKS_DOC.read_text(encoding="utf-8")
    marker = "## 4. Qualcomm AI Hub Cloud-Profiled Benchmarks"

    lines = [
        marker,
        "",
        "> [!IMPORTANT]",
        "> **AI Hub cloud-profiled**: The metrics below represent isolated single-model inference latency",
        "> measured on hosted Qualcomm AI Hub cloud test racks (Snapdragon X Elite CRD).",
        "",
    ]

    if skipped_reason:
        lines.extend([
            f"**Status**: Skipped ({skipped_reason}).",
            "",
            "To execute AI Hub cloud profiling, set the `QAI_HUB_API_TOKEN` environment variable and run:",
            "```powershell",
            "$env:QAI_HUB_API_TOKEN=\"<your_token>\"",
            "python aihub/profile_all.py",
            "```",
            "",
        ])
    else:
        lines.extend([
            "| Model Role | Zoo Model Name | Target Device | Cloud-Profiled Latency | NPU Acceleration % | Job Link |",
            "|---|---|---|---|---|---|",
        ])
        for r in results:
            lat = f"{r['latency_ms']} ms" if r.get("latency_ms") else "N/A"
            npu = f"{r['npu_percent']}%" if r.get("npu_percent") else "N/A"
            job = f"[Job Link]({r['job_urls'][0]})" if r.get("job_urls") else "N/A"
            lines.append(f"| {r['role']} | `{r['model_name']}` | {r['device']} | {lat} | {npu} | {job} |")
        lines.append("")

    new_section = "\n".join(lines)

    if marker in content:
        # Replace existing section
        parts = content.split(marker)
        content = parts[0].rstrip() + "\n\n" + new_section
    else:
        content = content.rstrip() + "\n\n" + new_section

    BENCHMARKS_DOC.write_text(content, encoding="utf-8")
    print(f"[AI Hub Profile] Updated {BENCHMARKS_DOC}")


def main():
    token = check_api_token()
    if not token:
        reason = "No QAI_HUB_API_TOKEN found in environment"
        print(f"\n[AI Hub Profile] NOTICE: {reason}. Skipping cloud compilation and profiling.")
        print("Set QAI_HUB_API_TOKEN=<token> and rerun 'python aihub/profile_all.py' to run on hosted Snapdragon devices.")
        update_benchmarks_doc([], skipped_reason=reason)
        return 0

    print(f"[AI Hub Profile] QAI_HUB_API_TOKEN detected. Starting cloud device compilation...")
    print(f"[AI Hub Profile] Target Device: {DEFAULT_DEVICE}")

    child_env = os.environ.copy()
    child_env["QAI_HUB_API_TOKEN"] = token
    child_env["QAIHM_CI"] = "1"
    child_env["PYTHONUNBUFFERED"] = "1"

    results = []

    for role, model_name in MODELS.items():
        print(f"\n--- Profiling {role} ({model_name}) on {DEFAULT_DEVICE} ---")
        cmd = [
            sys.executable, "-u", "-m", f"qai_hub_models.models.{model_name}.export",
            "--device", DEFAULT_DEVICE,
            "--target-runtime", "onnx",
        ]
        try:
            proc = subprocess.run(
                cmd,
                env=child_env,
                capture_output=True,
                text=True,
                timeout=600,
            )
            parsed = parse_export_output(proc.stdout + "\n" + proc.stderr)
            parsed["role"] = role
            parsed["model_name"] = model_name
            parsed["device"] = DEFAULT_DEVICE
            results.append(parsed)
            print(f"[AI Hub Profile] {role} finished: {parsed.get('latency_ms')} ms (NPU ops: {parsed.get('npu_percent')}%)")
        except Exception as exc:
            print(f"[AI Hub Profile] Failed profiling {role}: {exc}")
            results.append({
                "role": role,
                "model_name": model_name,
                "device": DEFAULT_DEVICE,
                "latency_ms": None,
                "npu_percent": None,
                "job_urls": [],
            })

    update_benchmarks_doc(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
