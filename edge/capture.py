import cv2
import numpy as np
from config import CONFIG

def capture_snapshot() -> np.ndarray | None:
    """Capture a single frame. Returns BGR numpy array or None on failure."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.CAMERA_RESOLUTION[1])

    # Burn warmup frames so auto-exposure settles
    for _ in range(CONFIG.CAMERA_WARMUP_FRAMES):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    # Resize to max dimension (preserves aspect ratio)
    h, w = frame.shape[:2]
    scale = CONFIG.MAX_DIMENSION / max(h, w)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    return frame