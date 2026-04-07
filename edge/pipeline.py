import numpy as np
from config import CONFIG
from edge.stages.intensity import apply_clahe, auto_gamma
from edge.stages.spatial import apply_gaussian
from edge.stages.restoration import apply_wiener_filter
from edge.stages.frequency import apply_butterworth
import cv2

stage_images: dict[str, np.ndarray] = {}

def run_pipeline(image: np.ndarray) -> tuple[np.ndarray, dict]:
    stages = {}

    stages["01_raw"] = image.copy()

    # DENOISE
    image = apply_gaussian(image, kernel_size=5)
    stages["02_denoise"] = image.copy()

    # CLAHE
    image = apply_clahe(image, clip_limit=2.0)
    stages["03_clahe"] = image.copy()

    # GAMMA
    image = auto_gamma(image)
    stages["04_gamma"] = image.copy()

    # BUTTERWORTH
    if CONFIG.ENABLE_FREQUENCY_FILTER:
        image = apply_butterworth(image)
        stages["05_butterworth"] = image.copy()

    # EDGES
    edges = cv2.Canny(image, 100, 200)
    stages["06_edges"] = edges.copy()

    # THRESHOLD
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    stages["07_threshold"] = thresh.copy()

    # WIENER (optional)
    if CONFIG.ENABLE_WIENER:
        restored = apply_wiener_filter(image, kernel_size=5)
        if restored.mean() > 5.0:
            image = restored
        stages["08_wiener"] = image.copy()

    # SHARPEN
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    image = cv2.filter2D(image, -1, kernel)
    stages["09_sharpen"] = image.copy()

    stages["10_final"] = image.copy()

    return image, stages