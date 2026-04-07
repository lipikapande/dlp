# restoration.py - image restoration filters (deblurring, denoising)
import cv2
import numpy as np
from scipy.signal import wiener

def apply_wiener_filter(image: np.ndarray,
                        kernel_size: int = 5,
                        noise_power: float | None = None) -> np.ndarray:
    """
    Wiener filter for deblurring. Applied per-channel.
    noise_power=None means scipy estimates it automatically.
    """
    result = np.zeros_like(image, dtype=np.float64)
    for c in range(image.shape[2]):
        channel = image[:, :, c].astype(np.float64)
        filtered = wiener(channel, (kernel_size, kernel_size), noise_power)
        result[:, :, c] = filtered
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_nlmeans_denoising(image: np.ndarray,
                            h: float = 10,
                            template_window: int = 7,
                            search_window: int = 21) -> np.ndarray:
    """
    Non-local means — better quality than Gaussian but slower.
    h controls filter strength (higher = more smoothing, less detail).
    """
    return cv2.fastNlMeansDenoisingColored(
        image, None, h, h, template_window, search_window
    )