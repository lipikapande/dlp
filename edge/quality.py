import cv2
import numpy as np
from dataclasses import dataclass
from config import CONFIG

@dataclass
class QualityReport:
    blur_score: float
    brightness: float
    snr_db: float
    passed: bool
    rejection_reason: str | None

def compute_blur(gray: np.ndarray) -> float:
    """Laplacian variance — lower = blurrier."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def compute_brightness(gray: np.ndarray) -> float:
    return float(gray.mean())

def compute_snr(gray: np.ndarray) -> float:
    """Signal-to-noise ratio in dB. Uses mean/std estimate."""
    mean = gray.mean()
    std = gray.std()
    if std < 1e-6:
        return 0.0
    return float(20 * np.log10(mean / std))

def assess_quality(image: np.ndarray) -> QualityReport:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = compute_blur(gray)
    brightness = compute_brightness(gray)
    snr = compute_snr(gray)

    rejection = None
    if blur < CONFIG.BLUR_THRESHOLD:
        rejection = f"Too blurry (score={blur:.1f}, need>{CONFIG.BLUR_THRESHOLD})"
    elif brightness < CONFIG.BRIGHTNESS_MIN:
        rejection = f"Too dark (brightness={brightness:.1f})"
    elif brightness > CONFIG.BRIGHTNESS_MAX:
        rejection = f"Overexposed (brightness={brightness:.1f})"
    elif snr < CONFIG.SNR_THRESHOLD:
        rejection = f"High noise (SNR={snr:.1f}dB)"

    return QualityReport(
        blur_score=blur,
        brightness=brightness,
        snr_db=snr,
        passed=(rejection is None),
        rejection_reason=rejection
    )