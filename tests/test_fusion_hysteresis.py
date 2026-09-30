"""
Unit Tests: Calibrated Log-Odds Fusion & Hysteresis State Machine.

Why this file is needed:
    Verifies calibrated Bayesian log-odds calculation and asymmetric hysteresis state
    transitions (enter P>=0.75 in 3 of 5, exit P<=0.50 in 4 of 5).

Repo capability missing:
    No repo implemented log-odds evidence aggregation or multi-window asymmetric hysteresis.
"""

import pytest
from callguard.core.evidence import Evidence
from callguard.core.claim_adapter import Claim
from callguard.core.debate import CallGuardDebateEngine
from callguard.core.hysteresis import HysteresisFilter


def test_calibrated_log_odds_fusion_authentic():
    engine = CallGuardDebateEngine(prior_synthetic_prob=0.10)
    ev = Evidence("test", "test", 1.0, 0.9, 0.0, 1.0)
    claim = Claim("c1", "Strong organic pulse", "authentic", 2.0, 0.9, ev, [])

    res = engine.debate_and_fuse([claim], ctx={})
    # Authentic claim should decrease synthetic probability below prior 0.10
    assert res.posterior_probability < 0.10
    assert res.verdict == "CONSISTENT"


def test_calibrated_log_odds_fusion_synthetic():
    engine = CallGuardDebateEngine(prior_synthetic_prob=0.10)
    ev = Evidence("test", "test", 1.0, 0.9, 0.0, 1.0)
    c1 = Claim("c1", "Anomalous flat pulse", "synthetic", 2.5, 0.95, ev, [])
    c2 = Claim("c2", "Vocoder high-freq cutoff", "synthetic", 2.0, 0.90, ev, [])

    res = engine.debate_and_fuse([c1, c2], ctx={})
    # Multiple strong synthetic claims should raise P towards 1.0
    assert res.posterior_probability > 0.85
    assert res.verdict == "LIKELY SYNTHETIC"


def test_hysteresis_enter_alert():
    hf = HysteresisFilter(window_size=5, enter_threshold=0.75, enter_min_windows=3, exit_threshold=0.50, exit_min_windows=4)

    # 1. Start clean
    s1 = hf.update(0.10)
    assert s1.badge == "CONSISTENT"

    # 2. Window 1 high prob (1/5)
    s2 = hf.update(0.80)
    assert s2.badge == "UNCERTAIN"

    # 3. Window 2 high prob (2/5)
    s3 = hf.update(0.85)
    assert s3.badge == "UNCERTAIN"

    # 4. Window 3 high prob (3/5 -> triggers LIKELY SYNTHETIC)
    s4 = hf.update(0.88)
    assert s4.badge == "LIKELY SYNTHETIC"
    assert s4.transition_occurred is True


def test_hysteresis_exit_alert():
    hf = HysteresisFilter(window_size=5, enter_threshold=0.75, enter_min_windows=3, exit_threshold=0.50, exit_min_windows=4)

    # Prime into LIKELY SYNTHETIC
    hf.update(0.80)
    hf.update(0.85)
    hf.update(0.90)
    assert hf._current_state == "LIKELY SYNTHETIC"

    # 1 drop below 0.50 should NOT immediately exit alert
    s1 = hf.update(0.20)
    assert s1.badge == "LIKELY SYNTHETIC"

    # 2 drops below 0.50
    s2 = hf.update(0.25)
    assert s2.badge == "LIKELY SYNTHETIC"

    # 3 drops below 0.50
    s3 = hf.update(0.30)
    assert s3.badge == "LIKELY SYNTHETIC"

    # 4 drops below 0.50 out of 5 -> EXIT to CONSISTENT!
    s4 = hf.update(0.15)
    assert s4.badge == "CONSISTENT"
    assert s4.transition_occurred is True
