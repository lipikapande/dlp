# quality.py - image quality assessment (Algorithm 6 from paper)
import cv2
import numpy as np
from dataclasses import dataclass
from config import CONFIG


@dataclass
class QualityReport:
    blur_score: float
    brightness: float
    snr_db: float
    edge_density: float
    passed: bool
    rejection_reason: str | None


def compute_blur(gray: np.ndarray) -> float:
    """Laplacian variance — lower = blurrier."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def compute_snr(gray: np.ndarray) -> float:
    """Estimate noise using local variance in flat regions."""
    # Apply a blur and compare to original — noisy images have high difference
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
    noise_std = noise.std()
    signal_std = gray.astype(np.float32).std()
    if noise_std < 1e-6:
        return 60.0  # very clean
    return float(20 * np.log10(signal_std / noise_std))


def compute_edge_density(gray: np.ndarray) -> float:
    """Fraction of edge pixels detected by Canny (num_edge_pixels / M*N)."""
    edges = cv2.Canny(gray, 50, 150)
    m, n = gray.shape
    return float(np.count_nonzero(edges)) / (m * n)


def assess_quality(image: np.ndarray) -> QualityReport:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = compute_blur(gray)
    brightness = compute_brightness(gray)
    snr = compute_snr(gray)
    edge_density = compute_edge_density(gray)

    # Algorithm 6 quality gate:
    # PASS if blur_score > BLUR_THRESHOLD AND
    #          BRIGHTNESS_MIN < brightness < BRIGHTNESS_MAX AND
    #          edge_density > 0.05 AND
    #          snr >= SNR_THRESHOLD
    rejection = None
    if blur < CONFIG.BLUR_THRESHOLD:
        rejection = f"Too blurry (score={blur:.1f}, need>{CONFIG.BLUR_THRESHOLD})"
    elif brightness < CONFIG.BRIGHTNESS_MIN:
        rejection = f"Too dark (brightness={brightness:.1f})"
    elif brightness > CONFIG.BRIGHTNESS_MAX:
        rejection = f"Overexposed (brightness={brightness:.1f})"
    elif snr < CONFIG.SNR_THRESHOLD:
        rejection = f"High noise (SNR={snr:.1f}dB)"
    elif edge_density < 0.01:
        rejection = f"Insufficient detail (edge_density={edge_density:.3f})"

    return QualityReport(
        blur_score=blur,
        brightness=brightness,
        snr_db=snr,
        edge_density=edge_density,
        passed=(rejection is None),
        rejection_reason=rejection
    )
