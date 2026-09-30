"""
CallGuard Capture Subsystem.

Why this file is needed:
    Exposes unified ring buffer and source capture abstractions.

Repo capability missing:
    Provides a standardized entry point for timestamped multi-modal ingest.
"""

from callguard.capture.ring_buffer import TimestampedRingBuffer, TimestampedSample

__all__ = ["TimestampedRingBuffer", "TimestampedSample"]
