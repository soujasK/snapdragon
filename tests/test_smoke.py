"""
Smoke Test: Full CallGuard Pipeline on Sample Video + WAV (CPU Fallback).

Why this file is needed:
    Verifies that the entire CallGuard NPU pipeline executes from end to end on actual
    video and audio media files using CPU fallback execution, validating capture ingest,
    evidence extraction, debate verification, hash chaining, and HUD generation.

Repo capability missing:
    No repo provided an end-to-end integration smoke test combining all four capabilities
    on local media files.
"""

import os
import wave
import math
import struct
import numpy as np
import pytest
import cv2

from callguard.pipeline import CallGuardPipeline
from callguard.capture.sources import FileCaptureSource
from callguard.config import load_config


@pytest.fixture(scope="session")
def sample_media(tmp_path_factory):
    """Generates a temporary 2-second sample video and WAV file for smoke testing."""
    media_dir = tmp_path_factory.mktemp("media")
    video_path = str(media_dir / "sample_call.mp4")
    audio_path = str(media_dir / "sample_call.wav")

    fps = 30
    duration_sec = 2.0
    total_frames = int(fps * duration_sec)
    width, height = 320, 240

    # 1. Create sample video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for i in range(total_frames):
        frame = np.full((height, width, 3), 100, dtype=np.uint8)
        # Face box in center with simulated pulse
        pulse = int(5.0 * math.sin(2 * math.pi * 1.2 * (i / fps)))
        frame[50:180, 80:240, 0] = 120 # B
        frame[50:180, 80:240, 1] = int(np.clip(140 + pulse, 0, 255)) # G
        frame[50:180, 80:240, 2] = 160 # R
        out.write(frame)
    out.release()

    # 2. Create sample 16kHz mono WAV audio
    sample_rate = 16000
    total_audio_samples = int(sample_rate * duration_sec)
    with wave.open(audio_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(sample_rate)
        # Synthetic speech-like harmonics
        audio_bytes = bytearray()
        for i in range(total_audio_samples):
            t = i / sample_rate
            val = 0.3 * math.sin(2 * math.pi * 220.0 * t) + 0.1 * math.sin(2 * math.pi * 440.0 * t)
            ival = int(val * 32767.0)
            audio_bytes.extend(struct.pack('<h', ival))
        wf.writeframes(audio_bytes)

    return video_path, audio_path


def test_full_pipeline_smoke_cpu(sample_media, tmp_path):
    video_path, audio_path = sample_media
    audit_file = str(tmp_path / "smoke_audit.db")

    cfg = load_config()
    cfg.audit.db_path = audit_file

    source = FileCaptureSource(
        video_path=video_path,
        audio_path=audio_path,
        fps=30,
        sample_rate=16000
    )

    pipeline = CallGuardPipeline(config=cfg, source=source)
    pipeline.start()

    processed_windows = 0
    try:
        # Process 10 evaluation steps
        for _ in range(10):
            res = pipeline.process_step()
            assert res is not None
            assert "badge" in res
            assert res["badge"] in ("CONSISTENT", "UNCERTAIN", "LIKELY SYNTHETIC")
            assert "terminal_hud" in res
            assert "NOTICE: Verdict text in UI is advisory." in res["terminal_hud"]
            assert res["audit_row_id"] > 0
            processed_windows += 1
    finally:
        pipeline.stop()

    assert processed_windows == 10

    # Verify audit hash chain
    assert pipeline.audit_db is not None
    is_valid, msg = pipeline.audit_db.verify_chain()
    assert is_valid is True
    assert "Verified 10 audit rows" in msg
