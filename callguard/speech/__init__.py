"""
CallGuard Speech Subsystem.
Provides audio capture, VAD gating, and live speech caption engines.
"""

from callguard.speech.capture import rms, is_silent, open_stream
from callguard.speech.demo_engine import DemoEngine, Transcript

__all__ = ["rms", "is_silent", "open_stream", "DemoEngine", "Transcript"]
