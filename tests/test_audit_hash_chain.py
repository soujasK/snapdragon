"""
Unit Tests: Cryptographic Audit Hash Chain.

Design:
    Verifies that the SQLite audit log establishes tamper-evident cryptographic SHA-256
    hash linkage across successive rows and detects tampering if historical rows are modified.
"""

import os
import sqlite3
import pytest
from callguard.core.audit import CallGuardAuditDB


def test_audit_hash_chain_creation_and_linkage(tmp_path):
    db_file = str(tmp_path / "test_audit.db")
    db = CallGuardAuditDB(db_path=db_file)

    # Record 3 successive windows
    r1, h1 = db.record_window(1, [{"claim": "pulse_ok"}], 0.10, 0.05, "CONSISTENT", 0.90)
    r2, h2 = db.record_window(2, [{"claim": "voice_ok"}], 0.10, 0.04, "CONSISTENT", 0.92)
    r3, h3 = db.record_window(3, [{"claim": "flat_noise"}], 0.10, 0.85, "LIKELY SYNTHETIC", 0.95)

    assert r1 == 1
    assert r2 == 2
    assert r3 == 3

    # All hashes must be distinct 64-char hex strings
    assert len(h1) == 64
    assert len(h2) == 64
    assert len(h3) == 64
    assert h1 != h2 != h3

    # Chain verification should succeed
    is_valid, msg = db.verify_chain()
    assert is_valid is True
    assert "Verified 3 audit rows" in msg


def test_audit_tamper_detection(tmp_path):
    db_file = str(tmp_path / "tamper_test.db")
    db = CallGuardAuditDB(db_path=db_file)

    db.record_window(1, [{"claim": "test1"}], 0.10, 0.05, "CONSISTENT", 0.90)
    db.record_window(2, [{"claim": "test2"}], 0.10, 0.80, "LIKELY SYNTHETIC", 0.95)
    db.record_window(3, [{"claim": "test3"}], 0.10, 0.82, "LIKELY SYNTHETIC", 0.96)

    # Tamper with row 2 posterior_prob directly via SQLite backdoor
    with sqlite3.connect(db_file) as conn:
        conn.execute("UPDATE audit_chain SET posterior_prob = 0.10 WHERE row_id = 2;")
        conn.commit()

    # Chain verification MUST fail
    is_valid, msg = db.verify_chain()
    assert is_valid is False
    assert "Tampered data at row 2" in msg
