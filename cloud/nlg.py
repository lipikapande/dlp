from typing import Any

SCENE_TEMPLATES = {
    "indoor": "You are indoors.",
    "outdoor": "You are outdoors.",
    "street": "You are on a street.",
    "default": "",
}


def generate_description(detections: list[dict], scene: str,
                          assistive: list[dict] | None = None) -> str:
    scene_phrase = SCENE_TEMPLATES.get(scene, SCENE_TEMPLATES["default"])

    warning_text = ""
    if assistive:
        high = [a for a in assistive if a.get("priority") == "HIGH"]
        if high:
            parts = []
            for a in high[:2]:
                lbl = a["label"].lower()
                prox = a.get("proximity", "nearby").lower()
                parts.append(f"{lbl} {prox} ahead")
            warning_text = "Warning: " + ", ".join(parts) + ". "
        elif not high:
            parts = []
            for a in assistive[:1]:
                lbl = a["label"].lower()
                conf = int(a["confidence"] * 100)
                prox = a.get("proximity", "ahead")
                if prox == "very close":
                    parts.append(f"Caution! {lbl} very close, {conf}% confidence.")
                elif prox == "nearby":
                    parts.append(f"Heads up, {lbl} nearby.")
                else:
                    parts.append(f"{lbl} detected ahead.")
            warning_text = " ".join(parts)

    if not detections:
        if warning_text:
            return f"{scene_phrase} {warning_text}".strip()
        return f"{scene_phrase} No objects detected with high confidence."

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