from ultralytics import YOLO
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch
import cv2

# Loaded once when the server starts — not on every request
_model = None

def get_model():
    global _model
    if _model is None:
        # Downloads yolov8n.pt automatically on first run (~6MB)
        _model = YOLO("yolov8n.pt")
    return _model

def detect_objects(image: np.ndarray,
                   confidence: float = 0.4) -> list[dict]:
    """
    Returns list of dicts: {label, confidence, bbox}
    bbox = [x1, y1, x2, y2] in pixels
    """
    model = get_model()
    results = model(image, conf=confidence, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [round(v) for v in bbox]
            })

    return detections

_clip_model = None
_clip_processor = None

CANDIDATE_LABELS = ["car", "person", "truck", "bus", "motorcycle", "bicycle", "background"]

def detect_with_clip(image: np.ndarray) -> list[dict]:
    global _clip_model, _clip_processor
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = _clip_processor(text=CANDIDATE_LABELS, images=pil_img, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = _clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

    results = []
    for label, prob in zip(CANDIDATE_LABELS, probs):
        if label != "background" and prob.item() > 0.1:
            results.append({
                "label": label,
                "confidence": round(prob.item(), 3),
                "proximity": "nearby",   # fixes "undefined"
                "bbox": [0, 0, image.shape[1], image.shape[0]]
            })

    # Return only the single highest confidence result
    if not results:
        return []
    best = max(results, key=lambda x: x["confidence"])
    conf = best["confidence"]
    if conf > 0.6:
        proximity = "very close"
    elif conf > 0.35:
        proximity = "nearby"
    else:
        proximity = "ahead"

    best["proximity"] = proximity
    
    return [best]