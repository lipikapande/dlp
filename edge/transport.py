import httpx
import asyncio
from config import CONFIG

async def upload_image(image_bytes: bytes) -> dict | None:
    """
    Async upload to cloud inference endpoint.
    Returns parsed JSON response or None on failure.
    """
    try:
        async with httpx.AsyncClient(timeout=CONFIG.UPLOAD_TIMEOUT) as client:
            response = await client.post(
                CONFIG.CLOUD_URL,
                content=image_bytes,
                headers={"Content-Type": "image/jpeg"},
            )
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        print("[transport] Upload timed out")
    except httpx.HTTPStatusError as e:
        print(f"[transport] HTTP {e.response.status_code}")
    except Exception as e:
        print(f"[transport] Unexpected error: {e}")
    return None

def upload_sync(image_bytes: bytes) -> dict | None:
    """Synchronous wrapper for non-async callers."""
    return asyncio.run(upload_image(image_bytes))