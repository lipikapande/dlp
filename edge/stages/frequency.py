# frequency.py - frequency domain filters (e.g. Butterworth low-pass)
import cv2
import numpy as np

def apply_butterworth(image: np.ndarray,
                      cutoff: float = 0.3,
                      order: int = 2) -> np.ndarray:
    """
    Low-pass Butterworth filter in frequency domain.
    cutoff: 0.0-1.0 (fraction of max frequency to keep)
    order: higher = sharper cutoff
    NOTE: slow on laptop too — keep ENABLE_FREQUENCY_FILTER=False for now
    """
    result = np.zeros_like(image)
    rows, cols = image.shape[:2]

    # Build the filter mask once
    crow, ccol = rows // 2, cols // 2
    y = np.arange(rows).reshape(-1, 1) - crow
    x = np.arange(cols).reshape(1, -1) - ccol
    d = np.sqrt(x**2 + y**2)
    d_max = np.sqrt(crow**2 + ccol**2)
    d_norm = d / (d_max * cutoff + 1e-6)
    mask = 1 / (1 + d_norm ** (2 * order))

    for c in range(image.shape[2]):
        channel = image[:, :, c].astype(np.float32)
        dft = np.fft.fft2(channel)
        dft_shift = np.fft.fftshift(dft)
        filtered = dft_shift * mask
        idft = np.fft.ifftshift(filtered)
        back = np.abs(np.fft.ifft2(idft))
        result[:, :, c] = np.clip(back, 0, 255).astype(np.uint8)

    return result