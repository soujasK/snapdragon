"""
Unit Tests: Enforcing Zero Non-Loopback Network Egress at Runtime.

Why this file is needed:
    Enforces the strict security requirement: CallGuard NPU must run 100% locally with
    zero non-loopback network egress.

Repo capability missing:
    No repo implemented an automated egress guard test asserting zero telemetry phoning-home
    or external network queries.
"""

import socket
import pytest
from callguard.pipeline import CallGuardPipeline
from callguard.capture.sources import SyntheticCaptureSource


def test_zero_non_loopback_network_egress(monkeypatch):
    """
    Hook socket connection attempts. Allow only localhost/loopback (127.0.0.1, ::1).
    Fail immediately if any non-loopback outbound connection is attempted.
    """
    original_connect = socket.socket.connect

    def guarded_connect(self, address):
        host = address[0]
        # Allow loopback addresses
        if host in ("127.0.0.1", "localhost", "::1"):
            return original_connect(self, address)
        raise PermissionError(f"[CallGuard Security Breach] Non-loopback network egress attempted to {address}!")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

    # Run CallGuard pipeline through multiple complete iterations
    source = SyntheticCaptureSource(fps=30, simulate_deepfake=False)
    pipeline = CallGuardPipeline(source=source)
    pipeline.start()

    try:
        for _ in range(5):
            res = pipeline.process_step()
            assert res is not None
            assert res["badge"] in ("CONSISTENT", "UNCERTAIN", "LIKELY SYNTHETIC")
    finally:
        pipeline.stop()
