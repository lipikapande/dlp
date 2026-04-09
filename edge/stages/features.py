# features.py - Module 6: Feature Extraction
# GLCM texture features, shape features, Hu moments
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def compute_glcm_features(image: np.ndarray) -> dict:
    """
    GLCM texture features at 4 orientations (0, 45, 90, 135 degrees).
    Returns contrast, correlation, energy, homogeneity averaged across orientations.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(gray, distances, angles, levels=256, symmetric=True, normed=True)
    return {
        "contrast":    round(float(graycoprops(glcm, 'contrast').mean()), 4),
        "correlation": round(float(graycoprops(glcm, 'correlation').mean()), 4),
        "energy":      round(float(graycoprops(glcm, 'energy').mean()), 4),
        "homogeneity": round(float(graycoprops(glcm, 'homogeneity').mean()), 4),
    }


def compute_shape_features(image: np.ndarray) -> dict:
    """
    Shape features from largest contour: aspect ratio, extent, solidity, Hu moments.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"aspect_ratio": 0, "extent": 0, "solidity": 0, "hu_moments": [0] * 4}

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    moments = cv2.moments(c)
    hu = cv2.HuMoments(moments).flatten().tolist()

    return {
        "aspect_ratio": round(w / h, 3) if h else 0,
        "extent":       round(area / (w * h), 3) if w * h else 0,
        "solidity":     round(area / hull_area, 3) if hull_area else 0,
        "hu_moments":   [round(v, 6) for v in hu[:4]],
    }


def compute_histogram_features(image: np.ndarray) -> dict:
    """Mean, variance, skewness, kurtosis of intensity histogram."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    pixels = gray.flatten().astype(np.float64)
    mean = float(np.mean(pixels))
    std = float(np.std(pixels))
    skewness = float(np.mean(((pixels - mean) / (std + 1e-6)) ** 3))
    kurtosis = float(np.mean(((pixels - mean) / (std + 1e-6)) ** 4)) - 3.0
    return {
        "mean":     round(mean, 2),
        "variance": round(std ** 2, 2),
        "skewness": round(skewness, 4),
        "kurtosis": round(kurtosis, 4),
    }


def extract_features(image: np.ndarray) -> dict:
    """Full Module 6 feature extraction — called on final pipeline output."""
    return {
        "glcm":      compute_glcm_features(image),
        "shape":     compute_shape_features(image),
        "histogram": compute_histogram_features(image),
    }
