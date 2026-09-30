"""
CallGuard Evaluation Subsystem.

Why this file is needed:
    Exposes dataset evaluation harness and metric calculation hooks.
"""

from eval.eval_hooks import FaceForensicsEvaluator, ASVspoofEvaluator, compute_metrics

__all__ = ["FaceForensicsEvaluator", "ASVspoofEvaluator", "compute_metrics"]
