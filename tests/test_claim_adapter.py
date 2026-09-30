"""
Unit Tests: Claim Adapter.

Why this file is needed:
    Verifies that quantitative Evidence objects are accurately transformed into
    polarized linguistic Claims with correct evidentiary weights and context discount tags.

Repo capability missing:
    No repo implemented an automated bridge from raw multi-modal biometric signals to
    polarized claims for adversarial verification.
"""

import pytest
from callguard.core.evidence import Evidence
from callguard.core.claim_adapter import ClaimAdapter, Claim


def test_claim_adapter_rppg_authentic():
    adapter = ClaimAdapter(rppg_snr_threshold=1.5)
    ev = Evidence(
        source="raw_tap.rppg",
        signal="pulse_snr_db",
        value=3.2,
        confidence=0.9,
        t_start=1.0,
        t_end=2.0,
    )
    claims = adapter.adapt([ev])
    assert len(claims) == 1
    c = claims[0]
    assert c.polarity == "authentic"
    assert "Physiological blood volume pulse" in c.text
    assert c.confidence == 0.9


def test_claim_adapter_rppg_synthetic_anomaly():
    adapter = ClaimAdapter(rppg_snr_threshold=1.5)
    ev = Evidence(
        source="raw_tap.rppg",
        signal="pulse_snr_db",
        value=0.4,
        confidence=0.85,
        t_start=1.0,
        t_end=2.0,
    )
    claims = adapter.adapt([ev])
    assert len(claims) == 1
    c = claims[0]
    assert c.polarity == "synthetic"
    assert "Absence of authentic hemodynamic" in c.text
    assert "low_light" in c.discountable_by


def test_claim_adapter_audio_anomalies():
    adapter = ClaimAdapter(audio_flatness_threshold=0.25, high_freq_cutoff_ratio=0.08, min_jitter_threshold=0.005)
    ev_flat = Evidence("raw_tap.audio", "spectral_flatness", 0.40, 0.8, 1.0, 2.0)
    ev_cutoff = Evidence("raw_tap.audio", "high_freq_ratio", 0.03, 0.8, 1.0, 2.0)
    ev_jitter = Evidence("raw_tap.audio", "pitch_jitter", 0.001, 0.8, 1.0, 2.0)

    claims = adapter.adapt([ev_flat, ev_cutoff, ev_jitter])
    assert len(claims) == 3
    for c in claims:
        assert c.polarity == "synthetic"
    assert "bitrate_drop" in claims[1].discountable_by
