# assistive_detector.py — domain-specific outdoor obstacle detector
# Runs on the DIP-PROCESSED image (not raw) so preprocessing feeds into this.
# YOLO runs on raw for general object labels.
# This layer adds outdoor assistive priority: OBSTACLE, PEDESTRIAN, VEHICLE, STEPS, FOOTPATH.
import cv2
import numpy as np


def detect_assistive(image: np.ndarray) -> list[dict]:
    """
    Contour + heuristic based outdoor assistive detector.

    Each detection:
        label      : OBSTACLE | PEDESTRIAN | VEHICLE | STEPS | FOOTPATH
        priority   : HIGH | MEDIUM | LOW
        proximity  : CLOSE | MEDIUM | FAR
        confidence : 0.0 – 1.0
        bbox       : [x, y, w, h] normalized 0-1 (top-left origin)
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results: list[dict] = []

    # ── FOOTPATH ─────────────────────────────────────────────────────────────
    # Bottom 30 % of frame: low texture variance + plausible brightness → path
    bottom = gray[int(h * 0.7):, :]
    tex_var = float(cv2.Laplacian(bottom, cv2.CV_64F).var())
    mean_b = float(bottom.mean())
    if tex_var < 300 and 30 < mean_b < 225:
        conf = round(min(1.0, (300 - tex_var) / 300), 2)
        results.append({
            "label": "FOOTPATH",
            "priority": "LOW",
            "proximity": "CLOSE",
            "confidence": conf,
            "bbox": [0.0, 0.7, 1.0, 0.3],
        })

    # ── STEPS / CURB ─────────────────────────────────────────────────────────
    # Significant horizontal-edge density in lower half → steps or curb
    lower = gray[h // 2:, :]
    sobel_y = cv2.Sobel(lower, cv2.CV_64F, 0, 1, ksize=3)
    h_edge_ratio = float((np.abs(sobel_y) > 30).sum()) / lower.size
    if 0.04 < h_edge_ratio < 0.35:
        results.append({
            "label": "STEPS",
            "priority": "HIGH",
            "proximity": "CLOSE",
            "confidence": round(min(1.0, h_edge_ratio * 4), 2),
            "bbox": [0.0, 0.5, 1.0, 0.5],
        })

    # ── CONTOUR ANALYSIS ─────────────────────────────────────────────────────
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = h * w
    min_area = img_area * 0.005   # ignore noise (< 0.5 %)
    max_area = img_area * 0.80    # ignore full-frame blobs

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = ch / (cw + 1e-6)          # height / width
        rel_area = area / img_area
        cy = (y + ch / 2) / h              # normalised centre-y (0=top)

        xn, yn, wn, hn = x/w, y/h, cw/w, ch/h

        if cy > 0.70 or rel_area > 0.10:
            proximity = "CLOSE"
        elif cy > 0.45 or rel_area > 0.03:
            proximity = "MEDIUM"
        else:
            proximity = "FAR"

        # PEDESTRIAN — tall narrow blob anywhere in frame
        if 1.8 < aspect < 6.0 and 0.015 < rel_area < 0.45 and yn < 0.75:
            conf = round(min(1.0, rel_area * 4 + 0.2), 2)
            results.append({
                "label": "PEDESTRIAN",
                "priority": "HIGH" if proximity == "CLOSE" else "MEDIUM",
                "proximity": proximity,
                "confidence": conf,
                "bbox": [round(xn,3), round(yn,3), round(wn,3), round(hn,3)],
            })

        # VEHICLE — wide, large, upper-to-mid frame
        elif aspect < 0.65 and rel_area > 0.04 and cy < 0.72:
            conf = round(min(1.0, rel_area * 3), 2)
            results.append({
                "label": "VEHICLE",
                "priority": "HIGH" if proximity == "CLOSE" else "MEDIUM",
                "proximity": proximity,
                "confidence": conf,
                "bbox": [round(xn,3), round(yn,3), round(wn,3), round(hn,3)],
            })

        # OBSTACLE (knee/hip height) — most dangerous for visually impaired
        elif 0.35 < cy < 0.78 and 0.008 < rel_area < 0.18 and 0.25 < aspect < 3.5:
            conf = round(min(1.0, rel_area * 7 + 0.1), 2)
            results.append({
                "label": "OBSTACLE",
                "priority": "HIGH",
                "proximity": proximity,
                "confidence": conf,
                "bbox": [round(xn,3), round(yn,3), round(wn,3), round(hn,3)],
            })

    # Deduplicate by label — keep highest confidence per class
    best: dict[str, dict] = {}
    for r in results:
        lbl = r["label"]
        if lbl not in best or r["confidence"] > best[lbl]["confidence"]:
            best[lbl] = r

    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    return sorted(best.values(),
                  key=lambda x: (priority_order[x["priority"]], -x["confidence"]))
