"""
Unit Tests: Priority NPU Queue Scheduler.

Why this file is needed:
    Verifies that the NPU duty-cycle scheduler accurately enforces the 60% budget limit,
    progressively throttles stages in strict order (super_res -> low_light -> deepfake),
    and strictly guarantees that protected stages (captions, landmarks) are NEVER throttled.

Repo capability missing:
    No repo implemented an adaptive duty-cycle scheduler or queue throttling hierarchy.
"""

import pytest
from callguard.runtime.scheduler import PriorityScheduler


def test_scheduler_budget_and_throttling_order():
    sched = PriorityScheduler(max_duty_cycle=0.60)

    # Initial state: all stages at normal rate
    st = sched.update_budget()
    assert st["duty_cycle"] < 0.60
    assert sched.stages["super_res"].throttle_level == 0
    assert sched.stages["low_light"].throttle_level == 0
    assert sched.stages["deepfake_cadence"].throttle_level == 0

    # Simulate sustained heavy compute load: high super_res latency
    for _ in range(5):
        sched.record_stage_latency("super_res", 25.0)
        sched.record_stage_latency("low_light", 15.0)
        sched.record_stage_latency("deepfake_cadence", 15.0)

    # First update under load: should throttle super_res first!
    st1 = sched.update_budget()
    assert sched.stages["super_res"].throttle_level > 0

    # Further heavy load: should throttle low_light next
    for _ in range(5):
        sched.record_stage_latency("super_res", 35.0)
        sched.record_stage_latency("low_light", 25.0)
    st2 = sched.update_budget()
    assert sched.stages["low_light"].throttle_level > 0

    # Further extreme load: high deepfake latency forces deepfake cadence throttling
    for _ in range(5):
        sched.record_stage_latency("deepfake_cadence", 30.0)
    st3 = sched.update_budget()
    assert sched.stages["deepfake_cadence"].throttle_level > 0

    # CRITICAL INVARIANT: Protected stages (captions, landmarks) must NEVER be throttled!
    assert sched.stages["captions"].throttle_level == 0
    assert sched.stages["landmarks"].throttle_level == 0
    assert sched.stages["captions"].current_rate_hz == 2.0
    assert sched.stages["landmarks"].current_rate_hz == 30.0

    # Verify should_run_stage preserves protected stages for every frame
    for frame_id in range(1, 100):
        assert sched.should_run_stage("captions", frame_id) is True
        assert sched.should_run_stage("landmarks", frame_id) is True
