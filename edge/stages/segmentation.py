# segmentation.py - edge detection and contour extraction
import cv2
import numpy as np

def apply_canny_edges(image: np.ndarray,
                      low_threshold: int = 50,
                      high_threshold: int = 150) -> np.ndarray:
    """Returns a binary edge map (single channel, uint8)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blurred, low_threshold, high_threshold)

def apply_morphology(edges: np.ndarray,
                     operation: str = "close",
                     kernel_size: int = 3) -> np.ndarray:
    """
    Morphological ops to clean up edge map.
    'close'  = dilate then erode  (fills gaps in edges)
    'open'   = erode then dilate  (removes noise)
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )
    ops = {
        "close": cv2.MORPH_CLOSE,
        "open": cv2.MORPH_OPEN,
        "dilate": cv2.MORPH_DILATE,
        "erode": cv2.MORPH_ERODE,
    }
    return cv2.morphologyEx(edges, ops.get(operation, cv2.MORPH_CLOSE), kernel)

def extract_contour_features(edges: np.ndarray) -> dict:
    """Returns basic shape features for diagnostics / logging."""
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50]
    return {
        "num_contours": len(areas),
        "max_area": max(areas) if areas else 0,
        "mean_area": float(np.mean(areas)) if areas else 0,
    }