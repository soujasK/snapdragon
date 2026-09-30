"""
Priority NPU Queue Scheduler with Duty-Cycle Budgeting & Progressive Throttling.

Architectural Design:
    Enforces a strict real-time compute budget (sum(cost * rate) <= 60% per second) across
    all stages competing for the Qualcomm Hexagon NPU / host compute pipeline.
    Implements intelligent progressive throttling under load while protecting critical
    accessibility and landmark tracking stages to maintain real-time responsiveness.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class StageProfile:
    name: str
    target_rate_hz: float
    current_rate_hz: float
    avg_latency_ms: float = 5.0
    is_protected: bool = False
    throttle_level: int = 0  # 0 = full speed, 1 = reduced, 2 = minimal, 3 = paused


class PriorityScheduler:
    """
    Priority scheduler managing a single NPU queue with duty-cycle budget constraint:
        sum(latency_sec * rate_hz) <= max_duty_cycle (0.60 = 60% per second).

    Configured for Snapdragon-powered HP PCs with physical Hexagon NPU offloading
    and CPU fallback support.

    Throttling Order under load:
        1. super_res (e.g. 30Hz -> 10Hz -> 0Hz)
        2. low_light (e.g. 30Hz -> 15Hz -> 0Hz)
        3. deepfake_cadence (e.g. every frame -> every 2nd -> every 4th)
        PROTECTED (Never throttled):
        - captions (accessibility)
        - landmarks (face tracking stability)
    """

    def __init__(self, max_duty_cycle: float = 0.60):
        self.max_duty_cycle = max_duty_cycle

        # Register stages with nominal baseline latencies (~42% duty cycle at 30fps)
        self.stages: Dict[str, StageProfile] = {
            "captions": StageProfile(name="captions", target_rate_hz=2.0, current_rate_hz=2.0, avg_latency_ms=20.0, is_protected=True),
            "landmarks": StageProfile(name="landmarks", target_rate_hz=30.0, current_rate_hz=30.0, avg_latency_ms=2.0, is_protected=True),
            "deepfake_cadence": StageProfile(name="deepfake_cadence", target_rate_hz=30.0, current_rate_hz=30.0, avg_latency_ms=4.0, is_protected=False),
            "low_light": StageProfile(name="low_light", target_rate_hz=30.0, current_rate_hz=30.0, avg_latency_ms=3.0, is_protected=False),
            "super_res": StageProfile(name="super_res", target_rate_hz=30.0, current_rate_hz=30.0, avg_latency_ms=4.0, is_protected=False),
        }

        # Execution cadence counters
        self._frame_counter = 0

    def record_stage_latency(self, stage_name: str, latency_ms: float):
        """Update exponential moving average latency for a stage."""
        if stage_name in self.stages:
            st = self.stages[stage_name]
            alpha = 0.35
            st.avg_latency_ms = (1.0 - alpha) * st.avg_latency_ms + alpha * max(0.1, latency_ms)

    def calculate_duty_cycle(self) -> float:
        """
        Calculate current projected duty cycle: sum((latency_ms / 1000.0) * current_rate_hz).
        """
        total = 0.0
        for st in self.stages.values():
            cost_sec = st.avg_latency_ms / 1000.0
            total += cost_sec * st.current_rate_hz
        return total

    def update_budget(self) -> Dict[str, Any]:
        """
        Evaluates duty cycle and adjusts throttling states to enforce <= 60% budget.
        Returns telemetry summary of scheduler state.
        """
        current_duty = self.calculate_duty_cycle()

        # If exceeding budget, throttle in order: super_res -> low_light -> deepfake_cadence
        if current_duty > self.max_duty_cycle:
            # 1. Throttle super_res
            sr = self.stages["super_res"]
            if sr.throttle_level == 0:
                sr.throttle_level = 1
                sr.current_rate_hz = 10.0
            elif sr.throttle_level == 1 and self.calculate_duty_cycle() > self.max_duty_cycle:
                sr.throttle_level = 2
                sr.current_rate_hz = 0.0 # Pause super_res

            # 2. Throttle low_light
            if self.calculate_duty_cycle() > self.max_duty_cycle:
                ll = self.stages["low_light"]
                if ll.throttle_level == 0:
                    ll.throttle_level = 1
                    ll.current_rate_hz = 15.0
                elif ll.throttle_level == 1 and self.calculate_duty_cycle() > self.max_duty_cycle:
                    ll.throttle_level = 2
                    ll.current_rate_hz = 0.0 # Pause low-light

            # 3. Throttle deepfake_cadence
            if self.calculate_duty_cycle() > self.max_duty_cycle:
                df = self.stages["deepfake_cadence"]
                if df.throttle_level == 0:
                    df.throttle_level = 1
                    df.current_rate_hz = 15.0 # Every 2nd frame
                elif df.throttle_level == 1 and self.calculate_duty_cycle() > self.max_duty_cycle:
                    df.throttle_level = 2
                    df.current_rate_hz = 7.5 # Every 4th frame

        # If duty cycle is safely low (< 42%), gradually recover
        elif current_duty < (self.max_duty_cycle * 0.70):
            # Recover deepfake first
            df = self.stages["deepfake_cadence"]
            if df.throttle_level > 0:
                df.throttle_level -= 1
                df.current_rate_hz = 15.0 if df.throttle_level == 1 else 30.0
            else:
                # Recover low-light next
                ll = self.stages["low_light"]
                if ll.throttle_level > 0:
                    ll.throttle_level -= 1
                    ll.current_rate_hz = 15.0 if ll.throttle_level == 1 else 30.0
                else:
                    # Recover super-res last
                    sr = self.stages["super_res"]
                    if sr.throttle_level > 0:
                        sr.throttle_level -= 1
                        sr.current_rate_hz = 10.0 if sr.throttle_level == 1 else 30.0

        return {
            "duty_cycle": round(self.calculate_duty_cycle(), 4),
            "max_budget": self.max_duty_cycle,
            "super_res_rate": self.stages["super_res"].current_rate_hz,
            "low_light_rate": self.stages["low_light"].current_rate_hz,
            "deepfake_rate": self.stages["deepfake_cadence"].current_rate_hz,
            "captions_rate": self.stages["captions"].current_rate_hz,
            "landmarks_rate": self.stages["landmarks"].current_rate_hz,
        }

    def should_run_stage(self, stage_name: str, frame_seq: int) -> bool:
        """
        Determines whether the specified stage should run on the given frame sequence.
        Guarantees protected stages (captions, landmarks) are never dropped.
        """
        if stage_name not in self.stages:
            return True

        st = self.stages[stage_name]
        if st.is_protected:
            return True

        if st.current_rate_hz <= 0.0:
            return False

        # Calculate stride based on target 30fps
        stride = max(1, int(round(30.0 / st.current_rate_hz)))
        return (frame_seq % stride == 0)
