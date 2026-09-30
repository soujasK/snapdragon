"""
Unit Tests: Timestamped Drop-Oldest Ring Buffer.

Design:
    Validates temporal ordering, monotonic clock timestamps, drop-oldest overflow behavior,
    and thread safety for multi-modal capture buffers.
"""

import time
import threading
import numpy as np
import pytest
from callguard.capture.ring_buffer import TimestampedRingBuffer


def test_ring_buffer_basic_push_pop():
    buf = TimestampedRingBuffer[int](capacity=5, name="TestBuf")
    seq1 = buf.push(100)
    seq2 = buf.push(200)

    assert seq1 == 1
    assert seq2 == 2
    assert buf.get_stats()["current_size"] == 2

    s1 = buf.pop_oldest()
    assert s1 is not None
    assert s1.data == 100
    assert s1.sequence_id == 1

    s2 = buf.pop_oldest()
    assert s2 is not None
    assert s2.data == 200

    assert buf.pop_oldest() is None


def test_ring_buffer_drop_oldest_overflow():
    buf = TimestampedRingBuffer[int](capacity=3, name="OverflowBuf")
    for i in range(1, 6):
        buf.push(i)

    stats = buf.get_stats()
    assert stats["total_pushed"] == 5
    assert stats["total_dropped"] == 2
    assert stats["current_size"] == 3

    # Retained elements should be 3, 4, 5
    remaining = [buf.pop_oldest().data for _ in range(3)]
    assert remaining == [3, 4, 5]


def test_ring_buffer_monotonic_timestamps():
    buf = TimestampedRingBuffer[str](capacity=10)
    t_prev = 0.0
    for i in range(5):
        buf.push(f"item_{i}")
        time.sleep(0.005)

    seq_prev = 0
    for i in range(5):
        sample = buf.pop_oldest()
        assert sample.timestamp >= t_prev
        assert sample.sequence_id > seq_prev
        t_prev = sample.timestamp
        seq_prev = sample.sequence_id


def test_ring_buffer_concurrency():
    buf = TimestampedRingBuffer[int](capacity=50)
    num_items = 500

    def producer():
        for i in range(num_items):
            buf.push(i)
            time.sleep(0.0001)

    consumed = []
    def consumer():
        while len(consumed) < num_items and (t.is_alive() or buf.get_stats()["current_size"] > 0):
            item = buf.pop_oldest(timeout=0.01)
            if item is not None:
                consumed.append(item.data)

    t = threading.Thread(target=producer)
    c = threading.Thread(target=consumer)
    t.start()
    c.start()
    t.join()
    c.join()

    stats = buf.get_stats()
    assert stats["total_pushed"] == num_items
    assert len(consumed) + stats["total_dropped"] == num_items
