from ultralytics import YOLO
import numpy as np

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