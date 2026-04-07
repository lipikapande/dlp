# compression.py - image compression utilities (JPEG encoding)
import cv2
import numpy as np
import io

def compress_to_jpeg_bytes(image: np.ndarray,
                           quality: int = 75) -> bytes:
    """Returns JPEG-encoded bytes ready for upload."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, buffer = cv2.imencode(".jpg", image, encode_params)
    if not success:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()