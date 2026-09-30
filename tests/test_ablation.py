"""
Ablation Tests: Fusion With vs Without Defender Environmental Context Discounting.

Why this file is needed:
    Demonstrates the quantitative and qualitative impact of the Defender's context-discounting
    logic in preventing false-positive synthetic deepfake classifications when video/audio
    quality degrades due to ambient low-light or network bitrate drops.

Repo capability missing:
    No repo provided ablation testing evaluating biometric verification accuracy with vs
    without environmental degradation discounting.
"""

import pytest
from callguard.core.evidence import Evidence
from callguard.core.claim_adapter import ClaimAdapter
from callguard.core.debate import CallGuardDebateEngine


def test_ablation_low_light_discount():
    """
    Scenario: Low-light environment causes poor camera SNR, making face capillary
    pulse undetectable (pulse_snr_db = 0.5 dB).
    Ablation compares posterior probability With vs Without Defender context discount.
    """
    adapter = ClaimAdapter(rppg_snr_threshold=1.5)
    engine = CallGuardDebateEngine(prior_synthetic_prob=0.10, low_light_discount=0.25)

    ctx_low_light = {"low_light": True, "bitrate_drop": False, "stutter": False, "multi_face": False}

    # Low rPPG SNR evidence
    ev = Evidence(
        source="raw_tap.rppg",
        signal="pulse_snr_db",
        value=0.5,
        confidence=0.90,
        t_start=1.0,
        t_end=2.0,
        ctx=ctx_low_light,
    )
    claims = adapter.adapt([ev])

    # 1. RUN WITHOUT DISCOUNT (Ablated Baseline)
    res_no_discount = engine.debate_and_fuse(claims, ctx=ctx_low_light, apply_context_discount=False)

    # 2. RUN WITH DEFENDER CONTEXT DISCOUNT (CallGuard Architecture)
    res_with_discount = engine.debate_and_fuse(claims, ctx=ctx_low_light, apply_context_discount=True)

    # Assertions
    # Without discount: Prosecutor's anomaly claim acts at full weight -> high synthetic probability
    assert res_no_discount.posterior_probability > 0.35

    # With discount: Defender discounts the anomaly because low_light is active -> probability stays low
    assert res_with_discount.posterior_probability < res_no_discount.posterior_probability
    assert len(res_with_discount.context_discounts_triggered) > 0
    assert any("low_light" in d for d in res_with_discount.context_discounts_triggered)

    print(
        f"\n[Ablation Result - Low-Light]:\n"
        f"  Without Discount: P(Synthetic) = {res_no_discount.posterior_probability:.4f} (False Alarm Risk)\n"
        f"  With Discount   : P(Synthetic) = {res_with_discount.posterior_probability:.4f} (Suppressed via Defender)\n"
        f"  Discounts Triggered: {res_with_discount.context_discounts_triggered}"
    )


def test_ablation_bitrate_drop_discount():
    """
    Scenario: Network compression / bitrate drop causes high-frequency audio cutoff.
    Ablation verifies that Defender discounts the vocoder false alarm when bitrate_drop is active.
    """
    adapter = ClaimAdapter()
    engine = CallGuardDebateEngine(prior_synthetic_prob=0.10, bitrate_drop_discount=0.30)

    ctx_drop = {"low_light": False, "bitrate_drop": True, "stutter": False, "multi_face": False}

    # Cutoff audio evidence
    ev = Evidence(
        source="raw_tap.audio",
        signal="high_freq_ratio",
        value=0.03, # Below 0.08 cutoff
        confidence=0.85,
        t_start=1.0,
        t_end=2.0,
        ctx=ctx_drop,
    )
    claims = adapter.adapt([ev])

    res_no_discount = engine.debate_and_fuse(claims, ctx=ctx_drop, apply_context_discount=False)
    res_with_discount = engine.debate_and_fuse(claims, ctx=ctx_drop, apply_context_discount=True)

    assert res_with_discount.posterior_probability < res_no_discount.posterior_probability
    assert any("bitrate_drop" in d for d in res_with_discount.context_discounts_triggered)
