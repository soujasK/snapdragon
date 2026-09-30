"""
CallGuard Core Module.

Why this file is needed:
    Exposes all core forensic and debate abstractions.
"""

from callguard.core.evidence import Evidence
from callguard.core.claim_adapter import Claim, ClaimAdapter
from callguard.core.raw_tap import RawTapPipeline
from callguard.core.enhance_path import EnhancePathPipeline
from callguard.core.captions import CaptionsPipeline
from callguard.core.debate import CallGuardDebateEngine, DebateResult, DebateArgument
from callguard.core.hysteresis import HysteresisFilter, HysteresisState
from callguard.core.audit import CallGuardAuditDB

__all__ = [
    "Evidence",
    "Claim",
    "ClaimAdapter",
    "RawTapPipeline",
    "EnhancePathPipeline",
    "CaptionsPipeline",
    "CallGuardDebateEngine",
    "DebateResult",
    "DebateArgument",
    "HysteresisFilter",
    "HysteresisState",
    "CallGuardAuditDB",
]
