# intensity.py - intensity adjustments (CLAHE, gamma correction)
import cv2
import numpy as np

def apply_clahe(image: np.ndarray,       
                clip_limit: float = 2.0,
                tile_size: tuple = (8, 8)) -> np.ndarray:
    """
    CLAHE on L channel only (avoids hue distortion).
    Works on BGR input, returns BGR.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_enhanced = clahe.apply(l)

    enhanced_lab = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def apply_gamma(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """
    Gamma correction. gamma > 1 brightens shadows, < 1 darkens.
    Uses a LUT for O(1) pixel-wise mapping.
    """
    inv_gamma = 1.0 / gamma
    lut = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(image, lut)

# def auto_gamma(image: np.ndarray) -> np.ndarray:
#     """Compute gamma based on mean brightness, then apply."""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     mean_brightness = gray.mean() / 255.0
#     # Darker images get stronger brightening
#     gamma = np.log(0.5) / np.log(mean_brightness + 1e-6)
#     gamma = float(np.clip(gamma, 0.5, 2.5))
#     return apply_gamma(image, gamma)

def auto_gamma(image: np.ndarray) -> np.ndarray:
    """Adaptive gamma: brightens dark images, darkens overexposed ones.

    apply_gamma uses LUT = (i/255)^(1/gamma) * 255, so:
      gamma > 1 → inv_gamma < 1 → output > input → BRIGHTENS
      gamma < 1 → inv_gamma > 1 → output < input → DARKENS

    Formula: gamma = log(mean) / log(0.5)
      dark image  (mean=0.09): gamma = log(0.09)/log(0.5) = 3.5  → brightens
      bright image (mean=0.91): gamma = log(0.91)/log(0.5) = 0.13 → darkens
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean() / 255.0
    gamma = float(np.clip(
        np.log(mean_brightness + 1e-6) / np.log(0.5),
        0.3, 4.0
    ))
    return apply_gamma(image, gamma)