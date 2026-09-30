"""
CallGuard Image Operations Backend.
Supports both OpenCV and a cv2-free backend (Pillow, SciPy, NumPy) for Windows ARM64.
"""

from __future__ import annotations
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

if not HAS_CV2:
    from PIL import Image, ImageDraw
    from scipy.ndimage import gaussian_filter
    try:
        from skimage import color as _skcolor, exposure as _skexposure
        HAS_SKIMAGE = True
    except ImportError:
        HAS_SKIMAGE = False


def resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """size = (w, h). Accepts HxWx3 uint8 or single-channel array."""
    if HAS_CV2:
        return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    w, h = size
    if img.ndim == 2:
        from scipy.ndimage import zoom
        zy, zx = h / img.shape[0], w / img.shape[1]
        return zoom(img, (zy, zx), order=1)
    pil = Image.fromarray(img)
    return np.array(pil.resize((w, h), Image.BILINEAR))


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if HAS_CV2:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img[..., ::-1]


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return bgr_to_rgb(img)


def box_blur(img: np.ndarray, radius: int = 5) -> np.ndarray:
    """Fast box blur for background softening."""
    if HAS_CV2:
        k = max(3, (radius * 2) | 1)
        return cv2.boxFilter(img, -1, (k, k))
    from scipy.ndimage import uniform_filter
    return uniform_filter(img, size=(radius * 2 + 1, radius * 2 + 1, 1)).astype(img.dtype)


def gaussian_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    k = ksize | 1
    if HAS_CV2:
        return cv2.GaussianBlur(img, (k, k), 0)
    from scipy.ndimage import gaussian_filter
    sigma = k / 6.0
    return gaussian_filter(img, sigma=(sigma, sigma, 0)).astype(img.dtype)


class Clahe:
    """Contrast-limited adaptive histogram equalization on the luminance channel."""

    def __init__(self, clip_limit: float = 2.0, tile: tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile = tile
        if HAS_CV2:
            self._c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        return self.apply_bgr(img_bgr)

    def apply_bgr(self, img_bgr: np.ndarray) -> np.ndarray:
        if HAS_CV2:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self._c.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        rgb = img_bgr[..., ::-1].astype(np.float64) / 255.0
        if not HAS_CV2 and 'HAS_SKIMAGE' in globals() and HAS_SKIMAGE:
            lab = _skcolor.rgb2lab(rgb)
            l = lab[..., 0] / 100.0
            l_eq = _skexposure.equalize_adapthist(l, clip_limit=0.02)
            lab[..., 0] = l_eq * 100.0
            out = np.clip(_skcolor.lab2rgb(lab), 0, 1)
        else:
            lo, hi = np.percentile(rgb, (1, 99))
            out = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
        return (out[..., ::-1] * 255).astype(np.uint8)


def imread(path: str) -> np.ndarray:
    if HAS_CV2:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        return img
    from PIL import Image
    return np.array(Image.open(path).convert("RGB"))[..., ::-1].copy()


def imwrite(path: str, img_bgr: np.ndarray) -> None:
    if HAS_CV2:
        cv2.imwrite(path, img_bgr)
        return
    from PIL import Image
    Image.fromarray(img_bgr[..., ::-1]).save(path)
