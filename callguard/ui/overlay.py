"""
CallGuard Overlay HUD & Advisory Trust Badge Display.

Architectural Design:
    Renders real-time biometric trust badges (CONSISTENT / UNCERTAIN / LIKELY SYNTHETIC),
    live speech captions, per-stage latency percentiles (p50/p95), and hardware execution provider
    telemetry. Enforces the mandatory disclaimer: "Verdict text in UI is advisory."
"""

import time
from typing import Dict, Any, Optional
import numpy as np


class CallGuardOverlayHUD:
    """
    Renders the CallGuard overlay HUD both as formatted terminal status blocks
    and as visual overlays onto display video frames.
    """

    ADVISORY_NOTICE = "NOTICE: Verdict text in UI is advisory."

    def __init__(self):
        self.last_render_time = time.monotonic()

    def format_terminal_hud(
        self,
        badge: str,
        probability: float,
        confidence: float,
        captions: str,
        duty_cycle: float,
        provider_name: str,
        stage_latencies: Dict[str, Dict[str, float]],
        dropped_frames: int,
    ) -> str:
        """Generates a rich, human-readable terminal HUD block."""
        badge_color = {
            "CONSISTENT": "\033[92m[ CONSISTENT - AUTHENTIC HUMAN ]\033[0m",
            "UNCERTAIN": "\033[93m[ UNCERTAIN - MONITORING ]\033[0m",
            "LIKELY SYNTHETIC": "\033[91m[ LIKELY SYNTHETIC - ALERT ]\033[0m",
        }.get(badge, f"[{badge}]")

        lines = [
            "=" * 78,
            f" CALLGUARD NPU  |  TRUST SENTINEL HUD  |  {self.ADVISORY_NOTICE}",
            "=" * 78,
            f" Trust Badge        : {badge_color}",
            f" Synthetic Prob     : {probability:.4f}  |  Confidence: {confidence:.2f}",
            f" Hardware Backend   : {provider_name}",
            f" NPU Duty Cycle     : {duty_cycle * 100.0:.1f}% / 60.0% max budget",
            f" Dropped Frames     : {dropped_frames}",
            "-" * 78,
            f" Live Captions (ASR): {captions if captions else '(listening...)'}",
            "-" * 78,
            " Stage Latency Telemetry (Measured empirical p50 / p95):",
        ]

        for st_name, stats in stage_latencies.items():
            lines.append(f"   - {st_name:<18}: p50={stats.get('p50_ms', 0.0):>6.2f} ms | p95={stats.get('p95_ms', 0.0):>6.2f} ms")

        lines.append("=" * 78)
        return "\n".join(lines)

    def draw_on_frame(
        self,
        frame_bgr: np.ndarray,
        badge: str,
        probability: float,
        captions: str,
        duty_cycle: float,
        provider_name: str,
    ) -> np.ndarray:
        """
        Draws visual HUD banner and captions directly onto display frame (cv2 if available, else numpy slice).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr

        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]

        # Draw a semi-transparent HUD banner at top
        banner_height = min(60, h // 4)
        banner_color = (30, 120, 30) if badge == "CONSISTENT" else (
            (30, 30, 180) if badge == "LIKELY SYNTHETIC" else (30, 130, 180)
        )

        annotated[:banner_height, :] = (
            annotated[:banner_height, :].astype(np.float32) * 0.4 +
            np.array(banner_color, dtype=np.float32) * 0.6
        ).astype(np.uint8)

        # Draw captions bar at bottom
        caption_height = min(40, h // 6)
        annotated[-caption_height:, :] = (
            annotated[-caption_height:, :].astype(np.float32) * 0.3 +
            np.array((20, 20, 20), dtype=np.float32) * 0.7
        ).astype(np.uint8)

        try:
            import cv2
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Badge text
            badge_text = f"CALLGUARD: {badge} (P={probability:.2f}) - {provider_name}"
            cv2.putText(annotated, badge_text, (10, 25), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated, self.ADVISORY_NOTICE, (10, 48), font, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
            # Captions text
            cap_text = f"ASR: {captions[:70]}" if captions else "ASR: (listening...)"
            cv2.putText(annotated, cap_text, (10, h - 12), font, 0.50, (100, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            pass

        return annotated
