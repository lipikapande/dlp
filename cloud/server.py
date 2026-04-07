from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
import numpy as np
import cv2
import asyncio
import json
import base64
from cloud.detector import detect_objects
from cloud.classifier import classify_scene
from cloud.nlg import generate_description

app = FastAPI()

_latest: dict = {}
_subscribers: list[asyncio.Queue] = []

def broadcast(data: dict):
    for q in _subscribers:
        q.put_nowait(data)

@app.post("/infer")
async def infer(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")

    payload = json.loads(body)

    # Decode the final processed image
    image_bytes = base64.b64decode(payload["image"])
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    # Stages already processed on edge — just use them
    stages_b64 = payload.get("stages", {})
    quality_data = payload.get("quality", {})

    # Detect/classify on the already-processed image
    detections = detect_objects(image)
    scene = classify_scene(image)
    description = generate_description(detections, scene)

    result = {
        "description": description,
        "detections": detections,
        "scene": scene,
        "stages": stages_b64,
        "quality": quality_data,
    }

    global _latest
    _latest = result
    broadcast(result)

    return JSONResponse({
        "description": description,
        "detections": detections,
        "scene": scene,
    })

@app.get("/events")
async def sse(request: Request):
    queue: asyncio.Queue = asyncio.Queue()
    _subscribers.append(queue)

    async def event_stream():
        if _latest:
            yield f"data: {json.dumps(_latest)}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            _subscribers.remove(queue)

    return StreamingResponse(event_stream(),
                             media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})

@app.get("/", response_class=HTMLResponse)
async def ui():
    with open("cloud/ui.html", "r", encoding="utf-8") as f:
        return f.read()