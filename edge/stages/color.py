# color.py - color space conversions and adjustments

import cv2
import numpy as np

def bgr_to_hsv(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def hsv_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

def apply_histogram_eq(image: np.ndarray) -> np.ndarray:
    """
    Histogram equalization on V channel only (preserves color).
    More aggressive than CLAHE — use one or the other, not both.
    """
    hsv = bgr_to_hsv(image)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    return hsv_to_bgr(hsv_eq)

def normalize_saturation(image: np.ndarray,
                         scale: float = 1.2) -> np.ndarray:
    """Boost or reduce color saturation. scale>1 = more vivid."""
    hsv = bgr_to_hsv(image)
    h, s, v = cv2.split(hsv)
    s = np.clip(s.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return hsv_to_bgr(cv2.merge([h, s, v]))