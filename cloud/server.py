from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from cloud.detector import detect_objects
from cloud.classifier import classify_scene
from cloud.nlg import generate_description

app = FastAPI()

@app.post("/infer")
async def infer(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty body")

    # Decode JPEG bytes to numpy array
    nparr = np.frombuffer(body, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    detections = detect_objects(image)
    scene = classify_scene(image)
    description = generate_description(detections, scene)

    return JSONResponse({
        "description": description,
        "detections": detections,
        "scene": scene,
    })