import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2

_model = None
_transform = None

# ImageNet labels for the 15 most scene-relevant classes
SCENE_LABELS = {
    "street, street scene": "street",
    "traffic light": "street",
    "crosswalk": "street",
    "dining table": "indoor",
    "sofa": "indoor",
    "bookcase": "indoor",
    "mountain": "outdoor",
    "lake": "outdoor",
    "forest": "outdoor",
    "building": "outdoor",
}

def get_model():
    global _model, _transform
    if _model is None:
        # Downloads ~14MB on first run
        _model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        _model.eval()
        _transform = models.MobileNet_V3_Small_Weights.DEFAULT.transforms()
    return _model, _transform

def classify_scene(image: np.ndarray) -> str:
    """Returns coarse scene label: 'indoor', 'outdoor', 'street', 'unknown'"""
    model, transform = get_model()

    # OpenCV is BGR, torchvision expects RGB PIL-like tensor
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(rgb)

    tensor = transform(pil_img).unsqueeze(0)  # Add batch dim

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output[0], dim=0)

    top5_indices = probs.topk(5).indices.tolist()
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    categories = weights.meta["categories"]

    for idx in top5_indices:
        label = categories[idx].lower()
        for key, scene in SCENE_LABELS.items():
            if key in label:
                return scene

    return "unknown"