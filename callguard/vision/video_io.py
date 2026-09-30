"""
CallGuard Video Frame I/O: Camera and Video File Readers.
"""

from __future__ import annotations
import pathlib
import sys
import numpy as np
from callguard.vision import imgops

HAS_CV2 = imgops.HAS_CV2
if HAS_CV2:
    import cv2


def open_source(spec: str, size: tuple[int, int] | None = None):
    """spec: integer webcam index or video file path. Yields BGR uint8 frames."""
    p = pathlib.Path(spec)
    is_image = p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if is_image:
        img = imgops.imread(str(p))
        if size:
            img = imgops.resize(img, size)
        while True:
            yield img.copy()
        return

    if HAS_CV2:
        cap = cv2.VideoCapture(int(spec) if spec.isdigit() else str(spec))
        if size and spec.isdigit():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            yield (imgops.resize(fr, size) if size and not spec.isdigit() else fr)
        cap.release()
        return

    # PyAV fallback
    try:
        import av
        container = av.open(spec)
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="bgr24")
            yield (imgops.resize(arr, size) if size else arr)
    except Exception as e:
        print(f"[VideoIO] Notice: video file reader error: {e}")
