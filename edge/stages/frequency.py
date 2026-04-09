# frequency.py - frequency domain filters (Butterworth low-pass and high-pass)
import cv2
import numpy as np


def compute_energy_ratio(image: np.ndarray) -> float:
    """
    Compute high-frequency to low-frequency energy ratio via 2D DFT.
    Low ratio (<0.3) → noisy/smooth → apply low-pass to clean.
    High ratio (>0.7) → already sharp → skip filtering.
    Mid range → apply high-pass for sharpening.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    # Low-frequency region: center 20% of spectrum
    r = int(min(crow, ccol) * 0.2)
    low_mask = np.zeros((rows, cols), dtype=bool)
    low_mask[crow - r:crow + r, ccol - r:ccol + r] = True

    low_energy = float(magnitude[low_mask].sum())
    high_energy = float(magnitude[~low_mask].sum())
    total = low_energy + high_energy + 1e-6

    return high_energy / total


def _build_butterworth_mask(rows: int, cols: int,
                             cutoff: float, order: int,
                             high_pass: bool = False) -> np.ndarray:
    crow, ccol = rows // 2, cols // 2
    y = np.arange(rows).reshape(-1, 1) - crow
    x = np.arange(cols).reshape(1, -1) - ccol
    d = np.sqrt(x ** 2 + y ** 2)
    d_max = np.sqrt(crow ** 2 + ccol ** 2)
    d_norm = d / (d_max * cutoff + 1e-6)
    low_pass = 1 / (1 + d_norm ** (2 * order))
    return (1 - low_pass) if high_pass else low_pass


def apply_butterworth(image: np.ndarray,
                      cutoff: float = 0.3,
                      order: int = 2,
                      high_pass: bool = False) -> np.ndarray:
    """
    Butterworth filter in frequency domain.
    high_pass=False → low-pass (noise smoothing, energy_ratio < 0.3)
    high_pass=True  → high-pass (sharpening, 0.3 <= energy_ratio <= 0.7)
    """
    result = np.zeros_like(image)
    rows, cols = image.shape[:2]
    mask = _build_butterworth_mask(rows, cols, cutoff, order, high_pass)

    for c in range(image.shape[2]):
        channel = image[:, :, c].astype(np.float32)
        dft = np.fft.fft2(channel)
        dft_shift = np.fft.fftshift(dft)
        filtered = dft_shift * mask
        idft = np.fft.ifftshift(filtered)
        back = np.abs(np.fft.ifft2(idft))
        result[:, :, c] = np.clip(back, 0, 255).astype(np.uint8)

    return result
