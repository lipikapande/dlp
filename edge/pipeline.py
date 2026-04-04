import numpy as np
from config import CONFIG
from edge.stages.intensity import apply_clahe, auto_gamma
from edge.stages.spatial import apply_gaussian
from edge.stages.restoration import apply_wiener_filter
from edge.stages.frequency import apply_butterworth   # implement similarly
from edge.stages.color import apply_histogram_eq      # implement similarly

def run_pipeline(image: np.ndarray) -> np.ndarray:
    """
    Run all enabled DIP stages in order.
    Each stage is a pure function: np.ndarray -> np.ndarray.
    Easy to disable or reorder.
    """
    # Stage 1: Contrast enhancement
    image = apply_clahe(image, clip_limit=2.0)
    image = auto_gamma(image)

    # Stage 2: Spatial filtering
    image = apply_gaussian(image, kernel_size=3)

    # Stage 3: Frequency domain (disabled by default — expensive on Pi)
    if CONFIG.ENABLE_FREQUENCY_FILTER:
        image = apply_butterworth(image, cutoff=0.3, order=2)

    # Stage 4: Restoration
    if CONFIG.ENABLE_WIENER:
        image = apply_wiener_filter(image, kernel_size=5)

    return image