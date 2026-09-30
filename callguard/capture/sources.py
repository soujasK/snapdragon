"""
Capture Sources: Live Device, WASAPI Loopback, Media File, and Synthetic Generator.

Architectural Design:
    Provides standardized multi-modal capture drivers that push synchronized, timestamped
    frames and audio chunks into CallGuard's timestamped ring buffers.
    Supports simultaneously capturing synchronized video and audio from live devices,
    media file pairs, or synthetic streams for deterministic headless benchmarking.
"""

import os
import sys
import time
import math
import wave
import threading
from typing import Optional, Tuple
import numpy as np

# Native CallGuard Vision I/O
try:
    from callguard.vision.video_io import open_source
except ImportError:
    open_source = None

from callguard.capture.ring_buffer import TimestampedRingBuffer


class CaptureSource:
    """Base interface for CallGuard multi-modal sources."""

    def start(self, video_buffer: TimestampedRingBuffer[np.ndarray], audio_buffer: TimestampedRingBuffer[np.ndarray]):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def is_running(self) -> bool:
        raise NotImplementedError


class FileCaptureSource(CaptureSource):
    """
    Feeds synchronized video and audio frames from local media files into ring buffers.
    Uses CallGuard's native video decoding stream.
    """

    def __init__(self, video_path: str, audio_path: Optional[str] = None, fps: int = 30, sample_rate: int = 16000):
        self.video_path = video_path
        self.audio_path = audio_path
        self.fps = fps
        self.sample_rate = sample_rate
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self, video_buffer: TimestampedRingBuffer[np.ndarray], audio_buffer: TimestampedRingBuffer[np.ndarray]):
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(video_buffer, audio_buffer),
            daemon=True,
            name="FileCaptureSourceThread"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def is_running(self) -> bool:
        return self._running

    def _run_loop(self, video_buffer: TimestampedRingBuffer[np.ndarray], audio_buffer: TimestampedRingBuffer[np.ndarray]):
        # Setup video reader
        video_gen = None
        if os.path.exists(self.video_path) and open_source is not None:
            try:
                video_gen = open_source(self.video_path)
            except Exception as e:
                print(f"[FileCaptureSource] Error opening video {self.video_path}: {e}")

        # Setup audio reader if provided (.wav file)
        audio_data = None
        audio_pos = 0
        if self.audio_path and os.path.exists(self.audio_path):
            try:
                with wave.open(self.audio_path, 'rb') as wf:
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    frate = wf.getframerate()
                    n_frames = wf.getnframes()
                    raw_bytes = wf.readframes(n_frames)
                    
                    if sampwidth == 2:
                        raw_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    else:
                        raw_np = np.frombuffer(raw_bytes, dtype=np.float32)
                    
                    if n_channels > 1:
                        raw_np = raw_np.reshape(-1, n_channels).mean(axis=1)
                    
                    # Resample to 16kHz if needed
                    if frate != self.sample_rate and len(raw_np) > 0:
                        new_len = int(len(raw_np) * self.sample_rate / frate)
                        raw_np = np.interp(
                            np.linspace(0, len(raw_np), new_len, endpoint=False),
                            np.arange(len(raw_np)),
                            raw_np
                        ).astype(np.float32)
                    audio_data = raw_np
            except Exception as e:
                print(f"[FileCaptureSource] Error reading audio {self.audio_path}: {e}")

        frame_interval = 1.0 / self.fps
        audio_chunk_samples = int(self.sample_rate * frame_interval)

        while self._running:
            loop_start = time.monotonic()
            
            # Read next video frame
            frame = None
            if video_gen is not None:
                try:
                    frame = next(video_gen)
                except StopIteration:
                    break
                except Exception:
                    break
            else:
                # Fallback blank frame if video not found
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            t_now = time.monotonic()
            video_buffer.push(frame, timestamp=t_now, metadata={"fps": self.fps})

            # Read next audio chunk
            if audio_data is not None and len(audio_data) > 0:
                chunk = audio_data[audio_pos: audio_pos + audio_chunk_samples]
                audio_pos += audio_chunk_samples
                if audio_pos >= len(audio_data):
                    audio_pos = 0 # Loop audio
                if len(chunk) < audio_chunk_samples:
                    chunk = np.pad(chunk, (0, audio_chunk_samples - len(chunk)))
            else:
                # Fallback silent audio chunk
                chunk = np.zeros(audio_chunk_samples, dtype=np.float32)

            audio_buffer.push(chunk, timestamp=t_now, metadata={"sample_rate": self.sample_rate})

            elapsed = time.monotonic() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


class SyntheticCaptureSource(CaptureSource):
    """
    Generates synthetic real-time video frames and audio chunks for testing and ablation.
    Simulates authentic physiological pulses or synthetic deepfake anomalies.
    """

    def __init__(self, fps: int = 30, sample_rate: int = 16000, simulate_deepfake: bool = False, low_light: bool = False):
        self.fps = fps
        self.sample_rate = sample_rate
        self.simulate_deepfake = simulate_deepfake
        self.low_light = low_light
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self, video_buffer: TimestampedRingBuffer[np.ndarray], audio_buffer: TimestampedRingBuffer[np.ndarray]):
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(video_buffer, audio_buffer),
            daemon=True,
            name="SyntheticCaptureThread"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def is_running(self) -> bool:
        return self._running

    def _run_loop(self, video_buffer: TimestampedRingBuffer[np.ndarray], audio_buffer: TimestampedRingBuffer[np.ndarray]):
        frame_interval = 1.0 / self.fps
        audio_chunk_len = int(self.sample_rate * frame_interval)
        step = 0

        while self._running:
            t_now = time.monotonic()
            step += 1

            # Synthetic Video Frame (RGB: 360x480)
            base_val = 30 if self.low_light else 120
            frame = np.full((360, 480, 3), base_val, dtype=np.uint8)

            # Draw a synthetic face rectangle in center
            # Forehead/cheek region simulated with cardiac pulse in green channel if authentic
            pulse = 0.0
            if not self.simulate_deepfake:
                # 72 BPM authentic pulse = 1.2 Hz
                pulse = 5.0 * math.sin(2 * math.pi * 1.2 * (step / self.fps))

            face_val = int(np.clip(base_val + 40, 0, 255))
            frame[80:280, 140:340, 0] = int(np.clip(face_val - 10, 0, 255)) # B
            frame[80:280, 140:340, 1] = int(np.clip(face_val + pulse, 0, 255)) # G (rPPG signal)
            frame[80:280, 140:340, 2] = int(np.clip(face_val + 20, 0, 255)) # R

            video_buffer.push(frame, timestamp=t_now, metadata={"synthetic": True})

            # Synthetic Audio (16kHz mono)
            t_arr = np.linspace(0, frame_interval, audio_chunk_len, endpoint=False)
            if self.simulate_deepfake:
                # Synthetic deepfake audio: vocoder artifact, abrupt cutoff above 3.5 kHz, flat spectrum
                freq = 440.0
                tone = 0.3 * np.sin(2 * np.pi * freq * t_arr).astype(np.float32)
                # Zero out high frequencies
                chunk = tone
            else:
                # Authentic human voice: natural fundamental + formants + natural micro-jitter
                f0 = 130.0 + 3.0 * np.sin(2 * np.pi * 0.5 * (step / self.fps))
                jitter = 0.01 * np.random.randn(audio_chunk_len).astype(np.float32)
                chunk = (
                    0.25 * np.sin(2 * np.pi * f0 * t_arr) +
                    0.15 * np.sin(2 * np.pi * (2 * f0) * t_arr) +
                    0.08 * np.sin(2 * np.pi * (3 * f0) * t_arr) +
                    jitter
                ).astype(np.float32)

            audio_buffer.push(chunk, timestamp=t_now, metadata={"sample_rate": self.sample_rate})

            elapsed = time.monotonic() - t_now
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


class LiveCaptureSource(CaptureSource):
    """
    Live hardware capture adapter:
    - Video: Native CallGuard open_source('0') or Windows camera index.
    - Audio: WASAPI loopback / sounddevice capture.
    """

    def __init__(self, camera_index: int = 0, audio_device: Optional[int] = None, fps: int = 30, sample_rate: int = 16000):
        self.camera_index = camera_index
        self.audio_device = audio_device
        self.fps = fps
        self.sample_rate = sample_rate
        self._running = False
        self._video_thread: Optional[threading.Thread] = None
        self._audio_thread: Optional[threading.Thread] = None

    def start(self, video_buffer: TimestampedRingBuffer[np.ndarray], audio_buffer: TimestampedRingBuffer[np.ndarray]):
        self._running = True
        self._video_thread = threading.Thread(
            target=self._video_loop,
            args=(video_buffer,),
            daemon=True,
            name="LiveVideoThread"
        )
        self._audio_thread = threading.Thread(
            target=self._audio_loop,
            args=(audio_buffer,),
            daemon=True,
            name="LiveAudioThread"
        )
        self._video_thread.start()
        self._audio_thread.start()

    def stop(self):
        self._running = False
        if self._video_thread and self._video_thread.is_alive():
            self._video_thread.join(timeout=1.0)
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=1.0)

    def is_running(self) -> bool:
        return self._running

    def _video_loop(self, video_buffer: TimestampedRingBuffer[np.ndarray]):
        frame_interval = 1.0 / self.fps
        gen = None
        if open_source is not None:
            try:
                gen = open_source(str(self.camera_index))
            except Exception as e:
                print(f"[LiveCaptureSource] Camera {self.camera_index} unavailable: {e}. Using fallback generator.")

        while self._running:
            t_now = time.monotonic()
            if gen is not None:
                try:
                    frame = next(gen)
                except Exception:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            video_buffer.push(frame, timestamp=t_now)
            elapsed = time.monotonic() - t_now
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _audio_loop(self, audio_buffer: TimestampedRingBuffer[np.ndarray]):
        chunk_seconds = 0.05 # 50ms blocks
        chunk_len = int(self.sample_rate * chunk_seconds)

        # Attempt sounddevice import
        sd_stream = None
        try:
            from callguard.speech.capture import open_stream
            sd_stream = open_stream(device=self.audio_device)
            sd_stream.start()
        except Exception as e:
            print(f"[LiveCaptureSource] Live audio capture unavailable: {e}. Using silent fallback.")

        while self._running:
            t_now = time.monotonic()
            if sd_stream is not None:
                try:
                    data, overflowed = sd_stream.read(chunk_len)
                    data_np = np.frombuffer(data, dtype=np.float32)
                except Exception:
                    data_np = np.zeros(chunk_len, dtype=np.float32)
            else:
                time.sleep(chunk_seconds)
                data_np = np.zeros(chunk_len, dtype=np.float32)

            audio_buffer.push(data_np, timestamp=t_now, metadata={"sample_rate": self.sample_rate})

        if sd_stream is not None:
            try:
                sd_stream.stop()
                sd_stream.close()
            except Exception:
                pass
