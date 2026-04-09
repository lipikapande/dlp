from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
import numpy as np
import cv2
import asyncio
import threading
import json
import base64
from cloud.detector import detect_objects
from cloud.classifier import classify_scene
from cloud.nlg import generate_description
from fastapi import UploadFile, File
import cv2
import numpy as np
from edge.pipeline import run_pipeline
from edge.quality import assess_quality
from edge.capture import capture_snapshot
from tts import speak

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

    # Detect/classify on the image sent from edge.
    # NOTE: edge/transport.py sends the processed image here.
    # For best results, edge should send the raw image for detection
    # and processed stages for display only (future improvement).
    detections = detect_objects(image)
    scene = classify_scene(image)
    description = generate_description(detections, scene)
    threading.Thread(target=speak, args=(description,), daemon=True).start()

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
    
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    
    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # Run pipeline for visual stages + Module 6 features
    processed, stages, features = run_pipeline(image)

    # Quality assessed on raw image
    report = assess_quality(image)

    # Encode stages
    stages_b64 = {}
    for k, img in stages.items():
        _, b = cv2.imencode(".jpg", img)
        stages_b64[k] = base64.b64encode(b).decode()

    # Always detect on the RAW image — YOLO was trained on natural images,
    # color/contrast filters shift the distribution and reduce confidence
    detections = detect_objects(image)
    scene = classify_scene(image)
    description = generate_description(detections, scene)
    threading.Thread(target=speak, args=(description,), daemon=True).start()

    result = {
        "description": description,
        "detections": detections,
        "scene": scene,
        "stages": stages_b64,
        "features": features,
        "quality": {
            "blur_score": report.blur_score,
            "brightness": report.brightness,
            "snr_db": report.snr_db,
            "edge_density": report.edge_density,
            "passed": report.passed,
        },
    }

    global _latest
    _latest = result
    broadcast(result)

    return JSONResponse(result)

@app.get("/trigger")
async def trigger():
    image = capture_snapshot()

    if image is None:
        raise HTTPException(status_code=500, detail="Camera error")

    # Run pipeline for visual stages + Module 6 features
    processed, stages, features = run_pipeline(image)

    # Quality assessed on raw image
    report = assess_quality(image)

    stages_b64 = {}
    for k, img in stages.items():
        _, b = cv2.imencode(".jpg", img)
        stages_b64[k] = base64.b64encode(b).decode()

    # Always detect on the RAW image
    detections = detect_objects(image)
    scene = classify_scene(image)
    description = generate_description(detections, scene)

    result = {
        "description": description,
        "detections": detections,
        "scene": scene,
        "stages": stages_b64,
        "features": features,
        "quality": {
            "blur_score": report.blur_score,
            "brightness": report.brightness,
            "snr_db": report.snr_db,
            "edge_density": report.edge_density,
            "passed": report.passed,
        },
    }

    global _latest
    _latest = result
    broadcast(result)
    threading.Thread(target=speak, args=(description,), daemon=True).start()

    return JSONResponse(result)