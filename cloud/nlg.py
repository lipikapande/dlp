from typing import Any

SCENE_TEMPLATES = {
    "indoor": "You are indoors.",
    "outdoor": "You are outdoors.",
    "street": "You are on a street.",
    "default": "",
}

def generate_description(detections: list[dict], scene: str) -> str:
    """
    Template-first NLG. Falls back to LLM only if needed.
    Keeps latency low for common cases.
    """
    scene_phrase = SCENE_TEMPLATES.get(scene, SCENE_TEMPLATES["default"])

    if not detections:
        return f"{scene_phrase} No objects detected with high confidence."

    # Sort by confidence descending, deduplicate by label, take top 5
    seen = set()
    unique = []
    for d in sorted(detections, key=lambda x: x["confidence"], reverse=True):
        if d["label"] not in seen:
            seen.add(d["label"])
            unique.append(d)
    top = unique[:5]
    labels = [f"{d['label']} ({d['confidence']*100:.0f}%)" for d in top]

    count = len(labels)
    if count == 1:
        objects_phrase = f"I can see a {labels[0]}"
    elif count == 2:
        objects_phrase = f"I can see a {labels[0]} and a {labels[1]}"
    else:
        objects_phrase = (
            f"I can see {', '.join(labels[:-1])}, and {labels[-1]}"
        )

    return f"{scene_phrase} {objects_phrase}."