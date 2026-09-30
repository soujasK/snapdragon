"""
Claim Adapter: Translating Multi-Modal Evidence to Polarized Forensic Claims.

Architectural Design:
    Provides parameterized templates that translate raw numerical Evidence into linguistic,
    polarized forensic propositions (claims) suitable for adversarial debate in the CallGuard verifier.
"""

from dataclasses import dataclass
from typing import List, Optional
from callguard.core.evidence import Evidence


@dataclass
class Claim:
    """
    A polarized linguistic proposition derived from biometric/acoustic evidence.
    """
    claim_id: str
    text: str
    polarity: str            # "synthetic" (pro-attack / Prosecution) or "authentic" (pro-human / Defense)
    weight: float            # Base evidence weight / information gain [0.1, 3.0]
    confidence: float        # Measurement confidence [0.0, 1.0]
    source_evidence: Evidence
    discountable_by: List[str] # List of ctx flags that can explain away this anomaly


class ClaimAdapter:
    """
    Adapts streaming Evidence objects into forensic Claims using verified domain rules.
    """

    def __init__(self, rppg_snr_threshold: float = 1.5,
                 audio_flatness_threshold: float = 0.25,
                 high_freq_cutoff_ratio: float = 0.08,
                 min_jitter_threshold: float = 0.005):
        self.rppg_snr_threshold = rppg_snr_threshold
        self.audio_flatness_threshold = audio_flatness_threshold
        self.high_freq_cutoff_ratio = high_freq_cutoff_ratio
        self.min_jitter_threshold = min_jitter_threshold

    def adapt(self, evidence_list: List[Evidence]) -> List[Claim]:
        """Convert a list of Evidence samples into polarized claims."""
        claims = []
        claim_counter = 0

        for ev in evidence_list:
            claim_counter += 1
            cid = f"claim_{claim_counter}_{ev.signal}"

            # 1. rPPG Pulse SNR Signal
            if ev.signal == "pulse_snr_db":
                if ev.value >= self.rppg_snr_threshold:
                    claims.append(Claim(
                        claim_id=cid,
                        text=f"Physiological blood volume pulse detected with strong SNR ({ev.value:.2f} dB >= {self.rppg_snr_threshold} dB).",
                        polarity="authentic",
                        weight=1.8,
                        confidence=ev.confidence,
                        source_evidence=ev,
                        discountable_by=[]
                    ))
                else:
                    claims.append(Claim(
                        claim_id=cid,
                        text=f"Absence of authentic hemodynamic capillary pulse micro-flush (SNR {ev.value:.2f} dB < {self.rppg_snr_threshold} dB).",
                        polarity="synthetic",
                        weight=2.0,
                        confidence=ev.confidence,
                        source_evidence=ev,
                        discountable_by=["low_light", "multi_face"]
                    ))

            # 2. Audio Spectral Flatness Signal (Wiener Entropy)
            elif ev.signal == "spectral_flatness":
                if ev.value > self.audio_flatness_threshold:
                    claims.append(Claim(
                        claim_id=cid,
                        text=f"Anomalous white-noise/vocoder spectral flatness observed in speech stream ({ev.value:.3f} > {self.audio_flatness_threshold}).",
                        polarity="synthetic",
                        weight=1.5,
                        confidence=ev.confidence,
                        source_evidence=ev,
                        discountable_by=["bitrate_drop", "stutter"]
                    ))
                else:
                    claims.append(Claim(
                        claim_id=cid,
                        text=f"Natural vocal tract harmonic resonance and formant spectral peaks confirmed (flatness {ev.value:.3f} <= {self.audio_flatness_threshold}).",
                        polarity="authentic",
                        weight=1.2,
                        confidence=ev.confidence,
                        source_evidence=ev,
                        discountable_by=[]
                    ))

            # 3. Audio High-Frequency Cutoff Ratio
            elif ev.signal == "high_freq_ratio":
                if ev.value < self.high_freq_cutoff_ratio:
                    claims.append(Claim(
                        claim_id=cid,
                        text=f"Severe high-frequency cutoff detected typical of synthetic neural vocoders ({ev.value:.3f} < {self.high_freq_cutoff_ratio}).",
                        polarity="synthetic",
                        weight=1.7,
                        confidence=ev.confidence,
                        source_evidence=ev,
                        discountable_by=["bitrate_drop"]
                    ))
                else:
                    claims.append(Claim(
                        claim_id=cid,
                        text=f"Natural high-frequency vocal acoustic spectrum preserved ({ev.value:.3f} >= {self.high_freq_cutoff_ratio}).",
                        polarity="authentic",
                        weight=1.0,
                        confidence=ev.confidence,
                        source_evidence=ev,
                        discountable_by=[]
                    ))

            # 4. Vocal Pitch Micro-Jitter
            elif ev.signal == "pitch_jitter":
                if ev.value < self.min_jitter_threshold:
                    claims.append(Claim(
                        claim_id=cid,
                        text=f"Robotic lack of natural physiological fundamental frequency pitch micro-jitter ({ev.value:.5f} < {self.min_jitter_threshold}).",
                        polarity="synthetic",
                        weight=1.6,
                        confidence=ev.confidence,
                        source_evidence=ev,
                        discountable_by=["stutter", "bitrate_drop"]
                    ))
                else:
                    claims.append(Claim(
                        claim_id=cid,
                        text=f"Natural human vocal cord micro-perturbation and pitch jitter confirmed ({ev.value:.5f} >= {self.min_jitter_threshold}).",
                        polarity="authentic",
                        weight=1.1,
                        confidence=ev.confidence,
                        source_evidence=ev,
                        discountable_by=[]
                    ))

            # 5. Generic or Custom Signals
            else:
                pol = "synthetic" if ev.value >= 0.5 else "authentic"
                claims.append(Claim(
                    claim_id=cid,
                    text=f"Signal {ev.signal} evaluated to {ev.value:.3f}.",
                    polarity=pol,
                    weight=1.0,
                    confidence=ev.confidence,
                    source_evidence=ev,
                    discountable_by=["low_light", "bitrate_drop"]
                ))

        return claims
