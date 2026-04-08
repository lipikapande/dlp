import numpy as np
from config import CONFIG
from edge.stages.intensity import apply_clahe, auto_gamma
from edge.stages.spatial import apply_gaussian
from edge.stages.restoration import apply_wiener_filter
from edge.stages.frequency import apply_butterworth
import cv2
from edge.quality import assess_quality

stage_images: dict[str, np.ndarray] = {}

def run_pipeline(image: np.ndarray) -> tuple[np.ndarray, dict]:
    stages = {}

    stages["01_raw"] = image.copy()

    report = assess_quality(image)

    # ✅ DENOISE (only if noisy)
    if report.snr_db < 5 and report.blur_score > 120:
        image = cv2.bilateralFilter(image, 9, 75, 75)
        stages["02_denoise"] = image.copy()

    # ✅ CLAHE (only if dark)
    if report.brightness < 130:
        image = apply_clahe(image, clip_limit=2.0)
        stages["03_clahe"] = image.copy()

    # ✅ GAMMA (fine adjustment)
    if report.brightness < 140:
        image = auto_gamma(image)
        stages["04_gamma"] = image.copy()

    # ✅ BUTTERWORTH (only if blur)
    if CONFIG.ENABLE_FREQUENCY_FILTER and report.blur_score < 100:
        image = apply_butterworth(image)
        stages["05_butterworth"] = image.copy()

    # ✅ EDGES (always for demo)
    edges = cv2.Canny(image, 100, 200)
    stages["06_edges"] = edges.copy()

    # ✅ THRESHOLD (always for demo)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    stages["07_threshold"] = thresh.copy()

    # ✅ WIENER (only if very noisy)
    if CONFIG.ENABLE_WIENER and report.snr_db < 3:
        restored = apply_wiener_filter(image, kernel_size=5)
        if restored.mean() > 5.0:
            image = restored
        stages["08_wiener"] = image.copy()

    # ✅ SHARPEN (only if still blurry)
    if report.blur_score < 200:
        blur = cv2.GaussianBlur(image, (0,0), 3)
        image = cv2.addWeighted(image, 1.8, blur, -0.8, 0)
        stages["09_sharpen"] = image.copy()

    stages["10_final"] = image.copy()

    return image, stages