"""
Cryptographic Audit Trail Database with SHA-256 Hash Chaining.

Architectural Design:
    Provides a persistent, tamper-evident SQLite audit log for every window evaluation,
    linking each audit entry cryptographically to its predecessor via SHA-256 hash chains.
    Enforces local immutability and complete auditability for forensic evidence records.
"""

import sqlite3
import hashlib
import json
import os
import time
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone


class CallGuardAuditDB:
    """
    SQLite audit database with tamper-evident cryptographic hash chaining.
    """

    GENESIS_HASH = "0" * 64

    def __init__(self, db_path: str = "callguard_audit.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        """Create audit table if it does not already exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_chain (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    window_id INTEGER NOT NULL,
                    claims_json TEXT NOT NULL,
                    prior_prob REAL NOT NULL,
                    posterior_prob REAL NOT NULL,
                    verdict TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    prev_hash TEXT NOT NULL,
                    current_hash TEXT NOT NULL
                );
            """)
            conn.commit()

    def get_last_hash(self) -> str:
        """Fetch the current_hash of the most recently inserted row, or genesis hash if empty."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT current_hash FROM audit_chain ORDER BY row_id DESC LIMIT 1;")
            row = cursor.fetchone()
            if row:
                return row[0]
            return self.GENESIS_HASH

    @staticmethod
    def compute_row_hash(
        row_id: int,
        timestamp: str,
        window_id: int,
        claims_json: str,
        prior_prob: float,
        posterior_prob: float,
        verdict: str,
        confidence: float,
        prev_hash: str,
    ) -> str:
        """Compute deterministic SHA-256 hash across canonical row fields."""
        payload = (
            f"{row_id}|{timestamp}|{window_id}|{claims_json}|"
            f"{prior_prob:.4f}|{posterior_prob:.4f}|{verdict}|{confidence:.4f}|{prev_hash}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def record_window(
        self,
        window_id: int,
        claims_summary: List[Dict[str, Any]],
        prior_prob: float,
        posterior_prob: float,
        verdict: str,
        confidence: float,
    ) -> Tuple[int, str]:
        """
        Appends an evaluation window record to the hash chain.
        Returns (row_id, current_hash).
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        claims_json = json.dumps(claims_summary, sort_keys=True)
        prev_hash = self.get_last_hash()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Insert with dummy hash to obtain autoincrement row_id
            cursor.execute("""
                INSERT INTO audit_chain (
                    timestamp, window_id, claims_json, prior_prob, posterior_prob,
                    verdict, confidence, prev_hash, current_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                timestamp, window_id, claims_json, float(prior_prob), float(posterior_prob),
                str(verdict), float(confidence), prev_hash, "PENDING"
            ))
            row_id = cursor.lastrowid

            # Calculate SHA-256 hash using assigned row_id
            curr_hash = self.compute_row_hash(
                row_id=row_id,
                timestamp=timestamp,
                window_id=window_id,
                claims_json=claims_json,
                prior_prob=prior_prob,
                posterior_prob=posterior_prob,
                verdict=verdict,
                confidence=confidence,
                prev_hash=prev_hash,
            )

            # Update row with cryptographic hash
            cursor.execute("UPDATE audit_chain SET current_hash = ? WHERE row_id = ?;", (curr_hash, row_id))
            conn.commit()

            return row_id, curr_hash

    def verify_chain(self) -> Tuple[bool, Optional[str]]:
        """
        Validates the entire audit log from row 1 to latest.
        Returns:
            (True, None) if completely valid.
            (False, error_explanation) if tampering or discrepancy is detected.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT row_id, timestamp, window_id, claims_json, prior_prob,
                       posterior_prob, verdict, confidence, prev_hash, current_hash
                FROM audit_chain ORDER BY row_id ASC;
            """)
            rows = cursor.fetchall()

            if not rows:
                return True, "Audit log is empty (valid genesis state)."

            expected_prev_hash = self.GENESIS_HASH

            for r in rows:
                (row_id, timestamp, window_id, claims_json, prior_prob,
                 posterior_prob, verdict, confidence, prev_hash, current_hash) = r

                # 1. Verify linkage to previous row
                if prev_hash != expected_prev_hash:
                    return False, f"Broken link at row {row_id}: prev_hash {prev_hash} != expected {expected_prev_hash}"

                # 2. Recalculate hash of current row
                recalc_hash = self.compute_row_hash(
                    row_id=row_id,
                    timestamp=timestamp,
                    window_id=window_id,
                    claims_json=claims_json,
                    prior_prob=prior_prob,
                    posterior_prob=posterior_prob,
                    verdict=verdict,
                    confidence=confidence,
                    prev_hash=prev_hash,
                )

                if recalc_hash != current_hash:
                    return False, f"Tampered data at row {row_id}: stored {current_hash} != computed {recalc_hash}"

                expected_prev_hash = current_hash

            return True, f"Verified {len(rows)} audit rows successfully."

    def get_recent_entries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return the most recent audit entries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT row_id, timestamp, window_id, claims_json, prior_prob,
                       posterior_prob, verdict, confidence, prev_hash, current_hash
                FROM audit_chain ORDER BY row_id DESC LIMIT ?;
            """, (limit,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                results.append({
                    "row_id": r[0],
                    "timestamp": r[1],
                    "window_id": r[2],
                    "claims": json.loads(r[3]),
                    "prior_prob": r[4],
                    "posterior_prob": r[5],
                    "verdict": r[6],
                    "confidence": r[7],
                    "prev_hash": r[8],
                    "current_hash": r[9],
                })
            return results
