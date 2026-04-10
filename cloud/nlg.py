from typing import Any

SCENE_TEMPLATES = {
    "indoor": "You are indoors.",
    "outdoor": "You are outdoors.",
    "street": "You are on a street.",
    "default": "",
}


def generate_description(detections: list[dict], scene: str,
                          assistive: list[dict] | None = None) -> str:
    """
    Template-first NLG. Prepends HIGH-priority assistive warnings before
    the YOLO scene description so TTS says the dangerous thing first.
    """
    scene_phrase = SCENE_TEMPLATES.get(scene, SCENE_TEMPLATES["default"])

    # ── Assistive warnings (from processed-image contour detector) ────────────
    warning_text = ""
    if assistive:
        high = [a for a in assistive if a["priority"] == "HIGH"]
        if high:
            parts = []
            for a in high[:2]:
                lbl = a["label"].lower()
                prox = a["proximity"].lower()
                parts.append(f"{lbl} {prox} ahead")
            warning_text = "Warning: " + ", ".join(parts) + ". "

    # ── YOLO detections ───────────────────────────────────────────────────────
    if not detections:
        return f"{warning_text}{scene_phrase} No objects detected with high confidence."

    # Sort by confidence, deduplicate by label, take top 5
    seen: set[str] = set()
    unique: list[dict] = []
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
        objects_phrase = f"I can see {', '.join(labels[:-1])}, and {labels[-1]}"

    return f"{warning_text}{scene_phrase} {objects_phrase}."
