# watermark.py - Module 7: Digital Watermarking
# Embeds a 32-bit invisible watermark in DCT mid-frequency coefficients
# Payload: 16-bit unix timestamp (low bits) + 16-bit device ID
# Robust: survives JPEG compression at quality >= 70

import cv2
import numpy as np
import time

DEVICE_ID = 0xA847  # derived from student ID 23BCE0847


def embed_watermark(image: np.ndarray, strength: float = 25.0) -> np.ndarray:
    """
    Embed 32-bit watermark in the Y channel using block DCT.

    Each 8x8 block encodes one bit at mid-frequency position (3,4).
    Quantization-index modulation (QIM): bit=1 → round to (n+0.5)*strength,
    bit=0 → round to n*strength. Invisible at strength<=30.
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)

    timestamp = int(time.time()) & 0xFFFF
    payload = (timestamp << 16) | DEVICE_ID
    bits = [(payload >> i) & 1 for i in range(32)]

    h, w = y.shape
    bit_idx = 0
    for row in range(0, h - 7, 8):
        for col in range(0, w - 7, 8):
            if bit_idx >= 32:
                break
            block = y[row:row + 8, col:col + 8].copy()
            dct = cv2.dct(block)
            coeff = dct[3, 4]
            if bits[bit_idx]:
                dct[3, 4] = (np.floor(coeff / strength) + 0.5) * strength
            else:
                dct[3, 4] = np.floor(coeff / strength) * strength
            y[row:row + 8, col:col + 8] = cv2.idct(dct)
            bit_idx += 1

    ycrcb[:, :, 0] = np.clip(y, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def extract_watermark(image: np.ndarray, strength: float = 25.0) -> int:
    """
    Extract the 32-bit watermark from an image.
    Returns the payload integer (timestamp<<16 | device_id).
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)

    h, w = y.shape
    bits = []
    bit_idx = 0
    for row in range(0, h - 7, 8):
        for col in range(0, w - 7, 8):
            if bit_idx >= 32:
                break
            block = y[row:row + 8, col:col + 8].copy()
            dct = cv2.dct(block)
            coeff = dct[3, 4]
            remainder = coeff % strength
            bits.append(1 if remainder > strength / 2 else 0)
            bit_idx += 1

    payload = 0
    for i, b in enumerate(bits[:32]):
        payload |= (b << i)
    return payload
