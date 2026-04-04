import cv2
import numpy as np

def apply_gaussian(image: np.ndarray, kernel_size: int = 3,
                   sigma: float = 0.0) -> np.ndarray:
    """
    Gentle denoising. kernel_size=3 on Pi is fast; 5 is 4x slower.
    sigma=0 lets OpenCV auto-compute from kernel size.
    """
    ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)

def apply_bilateral(image: np.ndarray, d: int = 9,
                    sigma_color: float = 75,
                    sigma_space: float = 75) -> np.ndarray:
    """
    Edge-preserving smoothing. Slower than Gaussian (~8x on Pi).
    Use only when texture detail matters.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)