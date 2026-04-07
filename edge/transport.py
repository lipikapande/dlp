import httpx
import asyncio
import json
import base64
import numpy as np
import cv2
from config import CONFIG

def _encode_stages(stages: dict) -> dict:  # take it as a parameter
    result = {}
    for k, img in stages.items():
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        result[k] = base64.b64encode(buf).decode()
    return result

async def upload_image(image_bytes: bytes, stages: dict, quality_report=None) -> dict | None:
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "image": base64.b64encode(image_bytes).decode(),
        "stages": _encode_stages(stages),
        "quality": {
            "blur_score": round(quality_report.blur_score, 2),
            "brightness": round(quality_report.brightness, 2),
            "snr_db": round(quality_report.snr_db, 2),
            "passed": quality_report.passed,
        } if quality_report else {}
    }

    try:
        async with httpx.AsyncClient(timeout=CONFIG.UPLOAD_TIMEOUT) as client:
            response = await client.post(
                CONFIG.CLOUD_URL,
                content=json.dumps(payload),
                headers=headers,
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"[transport] Error: {e}")
        return None

def upload_sync(image_bytes: bytes, stages: dict, quality_report=None) -> dict | None:
    return asyncio.run(upload_image(image_bytes, stages, quality_report))