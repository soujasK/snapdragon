"""
CallGuard NPU CLI Application Runner.

Design:
    Provides the primary command-line interface to launch CallGuard NPU in live camera/mic mode,
    file-input mode, or synthetic test mode.
"""

import sys
import os
import time
import argparse
import signal

from callguard.config import load_config
from callguard.pipeline import CallGuardPipeline
from callguard.capture.sources import FileCaptureSource, SyntheticCaptureSource, LiveCaptureSource


def parse_args():
    parser = argparse.ArgumentParser(description="CallGuard NPU - Multi-Modal Biometric & Acoustic Zero-Trust Sentinel")
    parser.add_argument("--config", type=str, default="callguard.yaml", help="Path to configuration file")
    parser.add_argument("--video", type=str, default=None, help="Video file path or camera index (e.g. 0)")
    parser.add_argument("--audio", type=str, default=None, help="Audio wav file path")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic multi-modal source")
    parser.add_argument("--simulate-deepfake", action="store_true", help="Inject synthetic deepfake anomalies")
    parser.add_argument("--low-light", action="store_true", help="Simulate low-light environment")
    parser.add_argument("--no-discount", action="store_true", help="Disable Defender context discounting (ablation mode)")
    parser.add_argument("--iterations", type=int, default=0, help="Number of windows to process (0 = infinite)")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display window")
    parser.add_argument("--verify-audit", action="store_true", help="Verify integrity of SQLite audit hash chain")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Audit verification mode
    if args.verify_audit:
        from callguard.core.audit import CallGuardAuditDB
        db = CallGuardAuditDB(db_path=cfg.audit.db_path)
        valid, msg = db.verify_chain()
        print(f"[CallGuard Audit Verification] Result: {'VALID' if valid else 'TAMPERED'} | {msg}")
        return 0 if valid else 1

    # Select Capture Source
    if args.video and not args.video.isdigit() and os.path.exists(args.video):
        source = FileCaptureSource(
            video_path=args.video,
            audio_path=args.audio,
            fps=cfg.capture.video_fps,
            sample_rate=cfg.capture.audio_sample_rate
        )
        print(f"[CallGuard] Source: File mode ({args.video})")
    elif args.synthetic or (args.video is None and args.audio is None):
        source = SyntheticCaptureSource(
            fps=cfg.capture.video_fps,
            sample_rate=cfg.capture.audio_sample_rate,
            simulate_deepfake=args.simulate_deepfake,
            low_light=args.low_light
        )
        print(f"[CallGuard] Source: Synthetic generator (deepfake={args.simulate_deepfake}, low_light={args.low_light})")
    else:
        cam_idx = int(args.video) if args.video and args.video.isdigit() else 0
        source = LiveCaptureSource(
            camera_index=cam_idx,
            fps=cfg.capture.video_fps,
            sample_rate=cfg.capture.audio_sample_rate
        )
        print(f"[CallGuard] Source: Live hardware device (camera {cam_idx})")

    # Instantiate pipeline
    pipeline = CallGuardPipeline(config=cfg, source=source)
    qnn_backend = cfg.runtime.qnn_options.get("backend_path", "QnnHtp.dll") if isinstance(cfg.runtime.qnn_options, dict) else getattr(cfg.runtime.qnn_options, "backend_path", "QnnHtp.dll")
    qnn_mode = cfg.runtime.qnn_options.get("htp_performance_mode", "burst") if isinstance(cfg.runtime.qnn_options, dict) else getattr(cfg.runtime.qnn_options, "htp_performance_mode", "burst")
    print(f"[CallGuard] Target Architecture: Qualcomm Snapdragon X / Hexagon NPU")
    print(f"[CallGuard] Execution Provider : {pipeline.hardware_label} (backend_path={qnn_backend}, mode={qnn_mode})")
    pipeline.start()

    print("[CallGuard] Pipeline active. Press Ctrl+C to terminate.")

    # Graceful shutdown handler
    running = True
    def _sig_handler(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _sig_handler)

    iteration_count = 0
    try:
        while running:
            iteration_count += 1
            res = pipeline.process_step(apply_context_discount=not args.no_discount)

            # Print terminal HUD
            print(res["terminal_hud"])
            print()

            if args.iterations > 0 and iteration_count >= args.iterations:
                break

            time.sleep(0.05)
    finally:
        pipeline.stop()
        print("[CallGuard] Shutdown complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
