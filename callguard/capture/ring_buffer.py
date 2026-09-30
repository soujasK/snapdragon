"""
Timestamped Ring Buffer with Drop-Oldest Policy.

Architectural Design:
    Provides thread-safe, bounded, timestamped circular buffers for incoming video frames
    and audio PCM samples governed by a single system-wide monotonic clock. Synchronizes video
    and audio streams against a unified monotonic timeline with an explicit drop-oldest
    policy and telemetry for dropped samples under high system load.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar, Optional, List, Tuple
import numpy as np

T = TypeVar("T")


@dataclass
class TimestampedSample(Generic[T]):
    """Container holding a data payload coupled with monotonic acquisition time."""
    data: T
    timestamp: float        # Monotonic acquisition timestamp in seconds
    sequence_id: int        # Strictly monotonically increasing frame/packet counter
    metadata: dict = None   # Optional capture-level attributes (dimensions, sample rate)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TimestampedRingBuffer(Generic[T]):
    """
    Thread-safe circular ring buffer with a single monotonic clock and drop-oldest behavior.
    """

    def __init__(self, capacity: int, name: str = "RingBuffer"):
        if capacity <= 0:
            raise ValueError("Ring buffer capacity must be strictly positive.")
        self.capacity = capacity
        self.name = name
        self._buffer: deque[TimestampedSample[T]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        
        # Telemetry counters (strictly measured, never fabricated)
        self._total_pushed: int = 0
        self._total_dropped: int = 0
        self._sequence_counter: int = 0

    def push(self, data: T, timestamp: Optional[float] = None, metadata: Optional[dict] = None) -> int:
        """
        Push a new sample into the buffer.
        If the buffer is full, the oldest sample is dropped and total_dropped is incremented.
        Uses time.monotonic() if timestamp is not provided.
        Returns the assigned sequence ID.
        """
        if timestamp is None:
            timestamp = time.monotonic()

        with self._lock:
            # Check if pushing will drop the oldest element
            if len(self._buffer) == self.capacity:
                self._total_dropped += 1

            self._sequence_counter += 1
            seq = self._sequence_counter
            sample = TimestampedSample(
                data=data,
                timestamp=timestamp,
                sequence_id=seq,
                metadata=metadata or {}
            )
            self._buffer.append(sample)
            self._total_pushed += 1
            self._not_empty.notify_all()
            return seq

    def pop_oldest(self, timeout: Optional[float] = None) -> Optional[TimestampedSample[T]]:
        """Pop the oldest available sample, waiting up to timeout seconds if empty."""
        with self._not_empty:
            if not self._buffer:
                if timeout is None or timeout <= 0:
                    return None
                if not self._not_empty.wait(timeout):
                    return None
            if self._buffer:
                return self._buffer.popleft()
            return None

    def pop_latest(self) -> Optional[TimestampedSample[T]]:
        """Pop the newest available sample, discarding older queued samples."""
        with self._lock:
            if not self._buffer:
                return None
            sample = self._buffer.pop()
            dropped = len(self._buffer)
            self._total_dropped += dropped
            self._buffer.clear()
            return sample

    def peek_latest(self) -> Optional[TimestampedSample[T]]:
        """Inspect the newest sample without removing it."""
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1]

    def peek_window(self, max_samples: int) -> List[TimestampedSample[T]]:
        """Return the most recent max_samples as a list without modifying the buffer."""
        with self._lock:
            if not self._buffer:
                return []
            items = list(self._buffer)
            return items[-max_samples:]

    def get_stats(self) -> dict:
        """Return exact real-time buffer telemetry."""
        with self._lock:
            return {
                "name": self.name,
                "current_size": len(self._buffer),
                "capacity": self.capacity,
                "total_pushed": self._total_pushed,
                "total_dropped": self._total_dropped,
                "drop_rate": (self._total_dropped / self._total_pushed) if self._total_pushed > 0 else 0.0
            }

    def clear(self):
        """Clear all contents and reset buffer."""
        with self._lock:
            self._buffer.clear()
