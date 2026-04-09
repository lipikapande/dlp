import numpy as np
import cv2
from config import CONFIG
from edge.stages.intensity import apply_clahe, auto_gamma
from edge.stages.spatial import apply_gaussian
from edge.stages.restoration import apply_wiener_filter
from edge.stages.frequency import apply_butterworth, compute_energy_ratio
from edge.stages.segmentation import apply_canny_edges, apply_morphology
from edge.stages.features import extract_features
from edge.stages.watermark import embed_watermark
from edge.quality import assess_quality

stage_images: dict[str, np.ndarray] = {}


def run_pipeline(image: np.ndarray) -> tuple[np.ndarray, dict]:
    stages: dict[str, np.ndarray] = {}
    stages["01_raw"] = image.copy()

    # ── Quality assessment on RAW image ──────────────────────────────────
    raw_report = assess_quality(image)
    print(f"[pipeline] Raw quality: blur={raw_report.blur_score:.1f}, "
          f"brightness={raw_report.brightness:.1f}, snr={raw_report.snr_db:.1f}dB, "
          f"edge_density={raw_report.edge_density:.3f}, passed={raw_report.passed}")

    if raw_report.passed:
        print("[pipeline] Raw image passed — skipping enhancement filters.")
        working = image.copy()

    else:
        print(f"[pipeline] Raw image failed: {raw_report.rejection_reason}")

        is_dark   = raw_report.brightness < CONFIG.BRIGHTNESS_MIN
        is_bright = raw_report.brightness > CONFIG.BRIGHTNESS_MAX
        is_blurry = raw_report.blur_score  < CONFIG.BLUR_THRESHOLD
        is_noisy  = raw_report.snr_db      < CONFIG.SNR_THRESHOLD

        working = image.copy()

        # ── Module 2: Intensity Transformations ───────────────────────────
        if is_dark or is_bright:
            working = apply_clahe(working, clip_limit=2.0)
            stages["02_clahe"] = working.copy()
            result = auto_gamma(working)
            if result is not working:   # auto_gamma returns same obj if no change needed
                working = result
                stages["03_gamma"] = working.copy()

        # ── Module 4: Noise removal ───────────────────────────────────────
        if is_noisy:
            if len(working.shape) == 2 or working.shape[2] == 1:
                working = cv2.fastNlMeansDenoising(working, None, h=10,
                                                   templateWindowSize=7,
                                                   searchWindowSize=21)
            else:
                working = cv2.fastNlMeansDenoisingColored(working, None, h=10, hColor=10,
                                                          templateWindowSize=7,
                                                          searchWindowSize=21)
            stages["04_denoise"] = working.copy()

        # ── Module 3: Frequency Domain Filtering ─────────────────────────
        if CONFIG.ENABLE_FREQUENCY_FILTER and (is_blurry or is_noisy):
            energy_ratio = compute_energy_ratio(working)
            print(f"[pipeline] Energy ratio: {energy_ratio:.3f}")
            if energy_ratio < 0.3:
                # Low energy ratio → noisy/smooth → low-pass to clean
                working = apply_butterworth(working, cutoff=0.3, order=2, high_pass=False)
                stages["05_butterworth_lp"] = working.copy()
            elif energy_ratio <= 0.7:
                # Mid range → high-pass for sharpening
                working = apply_butterworth(working, cutoff=0.3, order=2, high_pass=True)
                stages["05_butterworth_hp"] = working.copy()
            # energy_ratio > 0.7 → already sharp, skip

        # ── Module 4: Restoration (Wiener) ───────────────────────────────
        if is_noisy and CONFIG.ENABLE_WIENER:
            restored = apply_wiener_filter(working, kernel_size=5)
            if restored.mean() > 5.0:
                working = restored
            stages["07_wiener"] = working.copy()

        # ── Unsharp mask for blur ─────────────────────────────────────────
        if is_blurry:
            blur_pass = cv2.GaussianBlur(working, (0, 0), 3)
            working = cv2.addWeighted(working, 1.8, blur_pass, -0.8, 0)
            stages["06_sharpen"] = working.copy()

        post_report = assess_quality(working)
        print(f"[pipeline] Post-filter: blur={post_report.blur_score:.1f}, "
              f"brightness={post_report.brightness:.1f}, passed={post_report.passed}")
        if not post_report.passed:
            print(f"[pipeline] Still below gate: {post_report.rejection_reason}")

    # ── Module 5: Segmentation (always, for demo) ─────────────────────────
    edges = apply_canny_edges(working)
    stages["08_edges"] = edges.copy()

    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    morph = apply_morphology(thresh, operation="close")
    stages["09_threshold"] = morph.copy()

    # ── Module 7: Watermarking (before final output) ──────────────────────
    working = embed_watermark(working)

    stages["10_final"] = working.copy()

    # ── Module 6: Feature Extraction ─────────────────────────────────────
    features = extract_features(working)
    print(f"[pipeline] Features: GLCM contrast={features['glcm']['contrast']:.2f}, "
          f"energy={features['glcm']['energy']:.4f}, "
          f"solidity={features['shape'].get('solidity', 'N/A')}")

    return working, stages, features


# Backwards-compatible wrapper for callers that expect (image, stages)
def run_pipeline_compat(image: np.ndarray) -> tuple[np.ndarray, dict]:
    result = run_pipeline(image)
    return result[0], result[1]
