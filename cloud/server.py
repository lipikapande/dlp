from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
import numpy as np
import cv2
import asyncio
import threading
import json
import base64
from cloud.detector import detect_objects, detect_with_clip
from cloud.classifier import classify_scene
from cloud.nlg import generate_description
from cloud.assistive_detector import detect_assistive
from fastapi import UploadFile, File
from edge.pipeline import run_pipeline
from edge.quality import assess_quality
from edge.capture import capture_snapshot
from tts import speak

app = FastAPI()

_latest: dict = {}
_subscribers: list[asyncio.Queue] = []

BEFORE_EDGE_STAGES = [
    "07_wiener", "06_sharpen", "05_butterworth_hp",
    "05_butterworth_lp", "04_denoise", "03_gamma", "02_clahe"
]

def broadcast(data: dict):
    for q in _subscribers:
        q.put_nowait(data)

def get_best_fallback_image(stages_b64: dict):
    for key in BEFORE_EDGE_STAGES:
        if key in stages_b64:
            print(f"[fallback] Using stage: {key}")
            proc_bytes = base64.b64decode(stages_b64[key])
            proc_arr = np.frombuffer(proc_bytes, np.uint8)
            img = cv2.imdecode(proc_arr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    return None

@app.post("/infer")
async def infer(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")

    payload = json.loads(body)

    image_bytes = base64.b64decode(payload["image"])
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    stages_b64 = payload.get("stages", {})
    quality_data = payload.get("quality", {})

    detections = detect_objects(image)
    scene = classify_scene(image)

    assistive = []
    if not detections:
        fallback_img = get_best_fallback_image(stages_b64)
        if fallback_img is not None:
            assistive = detect_with_clip(fallback_img)
        else:
            assistive = detect_with_clip(image)

    description = generate_description(detections, scene, assistive)
    threading.Thread(target=speak, args=(description,), daemon=True).start()

    result = {
        "description": description,
        "detections": detections,
        "assistive": assistive,
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
        "assistive": assistive,
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

    processed, stages, features = run_pipeline(image)
    report = assess_quality(image)

    stages_b64 = {}
    for k, img in stages.items():
        _, b = cv2.imencode(".jpg", img)
        stages_b64[k] = base64.b64encode(b).decode()

    detections = detect_objects(image)
    scene = classify_scene(image)

    assistive = []
    if not detections:
        fallback_img = get_best_fallback_image(stages_b64)
        if fallback_img is not None:
            assistive = detect_with_clip(fallback_img)
        else:
            assistive = detect_with_clip(image)

    description = generate_description(detections, scene, assistive)
    threading.Thread(target=speak, args=(description,), daemon=True).start()

    result = {
        "description": description,
        "detections": detections,
        "assistive": assistive,
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

    processed, stages, features = run_pipeline(image)
    report = assess_quality(image)

    stages_b64 = {}
    for k, img in stages.items():
        _, b = cv2.imencode(".jpg", img)
        stages_b64[k] = base64.b64encode(b).decode()

    detections = detect_objects(image)
    scene = classify_scene(image)

    assistive = []
    if not detections:
        fallback_img = get_best_fallback_image(stages_b64)
        if fallback_img is not None:
            assistive = detect_with_clip(fallback_img)
        else:
            assistive = detect_with_clip(image)

    description = generate_description(detections, scene, assistive)
    threading.Thread(target=speak, args=(description,), daemon=True).start()

    result = {
        "description": description,
        "detections": detections,
        "assistive": assistive,
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