#walkthrough.md

# DLP — Vision Assistant: Full Walkthrough

> Branch: `lipika` · Repo: `lipikapande/dlp`

---

## What This Project Does

This is a **real-time vision assistant** that:

1. Captures a frame from your webcam (or accepts an uploaded image)
2. Runs a Digital Image Processing (DIP) pipeline on the edge (your laptop)
3. Sends the processed image + stage data to a local cloud server
4. Detects objects (YOLOv8) and classifies the scene (MobileNetV3)
5. If YOLOv8 finds nothing, falls back to CLIP-based semantic detection on the best available processed stage
6. Generates a natural-language description with proximity warnings and **speaks it aloud**
7. Shows all pipeline stages visually in a dark-mode web UI

The system is split into two logical halves: **edge** (capture + process) and **cloud** (detect + describe + serve UI).

---

## Folder Structure

```
dlp-lipika/
├── main.py                  # Entry point — keyboard-triggered capture loop
├── config.py                # All tunable parameters in one place
├── tts.py                   # Text-to-speech (pyttsx3, cross-platform)
├── requirements.txt         # All Python dependencies
├── yolov8n.pt               # YOLOv8 nano weights (bundled)
│
├── edge/                    # Runs on the device (laptop / Raspberry Pi)
│   ├── capture.py           # Webcam snapshot with warmup frames
│   ├── pipeline.py          # Conditional DIP stage orchestrator
│   ├── quality.py           # Blur / brightness / SNR quality gate
│   ├── compression.py       # JPEG encode for upload
│   ├── transport.py         # HTTP POST to cloud server (async via httpx)
│   └── stages/
│       ├── intensity.py     # CLAHE + gamma correction
│       ├── spatial.py       # Gaussian + bilateral filters
│       ├── frequency.py     # Butterworth low-pass (FFT-based)
│       ├── restoration.py   # Wiener filter + NL-means denoising
│       └── segmentation.py  # Canny edges + morphology + contour features
│
└── cloud/                   # FastAPI server — runs locally or on a server
    ├── server.py            # Routes: /infer, /upload, /trigger, /events, /
    ├── detector.py          # YOLOv8 + CLIP fallback detection (loaded once at startup)
    ├── classifier.py        # MobileNetV3 scene classification
    ├── assistive_detector.py# Contour-based obstacle detection on processed image
    ├── nlg.py               # Template-based NLG with proximity-aware warnings
    ├── tts.py               # (Placeholder) server-side gTTS audio bytes
    └── ui.html              # Dark-mode debug dashboard (SSE live updates)
```

---

## How It All Connects

```
[Webcam]
   │
   ▼
edge/capture.py          → captures 1280×720, burns 20 warmup frames, resizes to 640px
   │
   ▼
edge/pipeline.py         → conditional DIP stages based on quality metrics:
   │  02 CLAHE            (contrast)    — only if dark or bright
   │  03 Gamma            (brighten)    — only if dark
   │  04 Denoise          (NL-means)    — only if noisy
   │  05 Butterworth      (low/high pass FFT) — only if noisy
   │  07 Wiener restore   — only if noisy
   │  06 Sharpen          — only if blurry
   │  08 Canny edges      (always, for demo)
   │  09 Threshold        (always, for demo)
   │  10 Final output     (watermarked)
   │
   ▼
edge/quality.py          → QualityReport: blur (Laplacian), brightness (mean), SNR (local variance method)
   │
   ▼
edge/compression.py      → JPEG encode at quality=75
   │
   ▼
edge/transport.py        → POST JSON to http://localhost:8000/infer
                           payload: { image: base64 (RAW), stages: {key: base64}, quality: {...} }
   │
   ▼
cloud/server.py /infer   → decodes image, runs detection pipeline:
   │
   ├─ cloud/detector.py  → YOLOv8n on RAW image (conf=0.4)
   │                        If no detections:
   │                          → pick best pre-edge stage (07_wiener > 06_sharpen > ... > 02_clahe)
   │                          → CLIP on that stage (conf > 0.1)
   │                          → if no stages available (quality passed): CLIP on raw image
   ├─ cloud/classifier.py→ MobileNetV3 — returns "indoor" | "outdoor" | "street" | "unknown"
   └─ cloud/nlg.py       → proximity-aware description:
                            "Caution! car very close (93%)." / "Heads up, person nearby."
   │
   ▼
SSE broadcast → browser (ui.html) updates live
   │
   ▼
tts.py (edge)            → pyttsx3 speaks the description aloud
```

---

## Detection Pipeline Logic

The system uses a **two-stage detection strategy**:

1. **Primary: YOLOv8n on RAW image** — fast, accurate on well-lit natural images
2. **Fallback: CLIP on processed image** — semantic detection that handles dark, noisy, or degraded images where YOLO fails

Fallback stage selection priority (best enhanced image before edge detection):
`07_wiener → 06_sharpen → 05_butterworth_hp → 05_butterworth_lp → 04_denoise → 03_gamma → 02_clahe`

If quality gate passed (no enhancement stages exist), CLIP runs on the raw image directly.

CLIP proximity is estimated from confidence:

- `> 0.6` → "very close" → "Caution! {label} very close."
- `> 0.35` → "nearby" → "Heads up, {label} nearby."
- `≤ 0.35` → "ahead" → "{label} detected ahead."

---

## Setup

### 1. Install dependencies

```bash
cd /Volumes/Share/Projects/DLP/dlp-lipika
pip install -r requirements.txt
```

> On macOS, if `pyttsx3` has issues: `pip install pyobjc` may be needed.
> On a Raspberry Pi, replace `opencv-python` with `opencv-python-headless`.

### 2. Start the cloud server

Open **Terminal 1**:

```bash
cd /Volumes/Share/Projects/DLP/dlp-lipika
uvicorn cloud.server:app --host 0.0.0.0 --port 8000 --reload
```

> Note: CLIP model (~605MB) downloads automatically on first run from HuggingFace. Do not edit files while it is downloading — uvicorn's `--reload` will interrupt the download.

The server starts at `http://localhost:8000`.  
Open the debug UI in your browser: **http://localhost:8000**

### 3. Run the edge client

Open **Terminal 2**:

```bash
cd /Volumes/Share/Projects/DLP/dlp-lipika
python main.py
```

Press **ENTER** each time you want to capture. The pipeline runs, image uploads, and the description is spoken aloud + shown in the browser.

### 4. (Optional) Upload an image from the browser

In the web UI, click **Upload** and pick any image file. The server will run the full pipeline on it and push the result to the UI via SSE.

### 5. (Optional) Trigger capture from the browser

Click **Capture** in the UI — this calls `GET /trigger` which opens the webcam server-side and processes the image without needing Terminal 2.

---

## Key Configuration (`config.py`)

| Parameter                 | Default                | What it does                                              |
| ------------------------- | ---------------------- | --------------------------------------------------------- |
| `BLUR_THRESHOLD`          | 30.0                   | Laplacian variance below this = blurry warning            |
| `BRIGHTNESS_MIN/MAX`      | 40.0 / 240.0           | Mean pixel brightness range                               |
| `SNR_THRESHOLD`           | 8.0 dB                 | Local variance SNR — below this = noisy, triggers filters |
| `ENABLE_FREQUENCY_FILTER` | True                   | Butterworth FFT filter (slow — disable on Pi)             |
| `ENABLE_WIENER`           | True                   | Wiener deblur (only fires if noisy)                       |
| `JPEG_QUALITY`            | 75                     | Upload compression quality                                |
| `MAX_DIMENSION`           | 640                    | Resize longest side to this before upload                 |
| `CLOUD_URL`               | `localhost:8000/infer` | Change for remote server                                  |
| `CAMERA_RESOLUTION`       | 1280×720               | Webcam capture resolution                                 |
| `CAMERA_WARMUP_FRAMES`    | 20                     | Frames burned for auto-exposure to settle                 |

---

## API Endpoints

| Method | Route      | What it does                                                       |
| ------ | ---------- | ------------------------------------------------------------------ |
| `POST` | `/infer`   | Main endpoint — receives JSON `{image, stages, quality}` from edge |
| `POST` | `/upload`  | Accepts a raw image file, runs full pipeline server-side           |
| `GET`  | `/trigger` | Opens camera on the server, runs full pipeline                     |
| `GET`  | `/events`  | SSE stream — browser subscribes for live updates                   |
| `GET`  | `/`        | Serves `cloud/ui.html` debug dashboard                             |

---

## DIP Stages Explained

| Stage          | Technique                                                     | When it fires          |
| -------------- | ------------------------------------------------------------- | ---------------------- |
| 02 CLAHE       | Contrast Limited Adaptive Histogram Equalization on L channel | Dark or overexposed    |
| 03 Gamma       | Fixed γ=1.5 brightening via LUT                               | Dark                   |
| 04 Denoise     | NL-means denoising (color-aware)                              | SNR < threshold        |
| 05 Butterworth | Low-pass or high-pass FFT filter depending on energy ratio    | Noisy                  |
| 07 Wiener      | Frequency-domain deblurring                                   | Noisy + Wiener enabled |
| 06 Sharpen     | Unsharp mask via weighted Gaussian                            | Blurry                 |
| 08 Canny edges | Edge detection                                                | Always (for demo)      |
| 09 Threshold   | Binary threshold at 127 + morphological close                 | Always (for demo)      |
| 10 Final       | Watermarked output                                            | Always                 |

---

## What Could Be Improved

### Code Quality

- **`auto_gamma` is hardcoded** (`intensity.py:42`): The adaptive version using `mean_brightness` is commented out and replaced with a fixed `gamma=1.5`. The dynamic version is strictly better — restore it.
- **`segmentation.py` is never called**: `apply_canny_edges`, `apply_morphology`, and `extract_contour_features` exist but the pipeline uses raw `cv2.Canny` inline instead. Either use the module or delete it.
- **`edge/stages/spatial.py` is imported but unused**: `apply_gaussian` is imported in `pipeline.py` but never called.
- **`cloud/tts.py` is a dead placeholder**: It imports `gtts` which isn't in `requirements.txt`. Either wire it up or delete the file.
- **Duplicate `/trigger` and `/upload` logic** in `server.py`: Both routes repeat the same detection/describe/broadcast block. Extract a helper function.
- **Global mutable `_latest` and `_subscribers`** in `server.py`: Works fine for single-user local use but will break under concurrency. Use a proper state container if scaling.

### Robustness

- **No `__init__.py` files** in `edge/` or `cloud/`: Works due to Python path tricks but will break if packaged or if imports are run from a different working directory.
- **`asyncio.run()` inside `upload_sync`** (`transport.py`): Will crash if called from inside an already-running event loop. Use `httpx` sync client instead.
- **Camera opened/released on every capture** (`capture.py`): Opening `VideoCapture(0)` on every press adds ~0.5s latency. Keep it open and release on `KeyboardInterrupt`.
- **TTS engine re-initialized if an exception occurs** (`tts.py`): If `runAndWait()` throws, `_engine` stays set to the broken instance. Reset to `None` in the `except` block.

### Performance

- **Butterworth FFT runs per-channel in Python loops** (`frequency.py`): Can be vectorized with `np.fft.fftn` across channels — ~3× faster.
- **CLIP model is 605MB and slow on CPU**: Consider `clip-vit-base-patch16` or a quantized version for faster inference.
- **`CAMERA_WARMUP_FRAMES=20`** causes ~0.5–1s of wasted reads on every capture. 5–10 frames is sufficient for most USB webcams.

### Features

- **No bounding box overlay on the final image**: YOLO detections have `bbox` coordinates but they're never drawn on the image shown in the UI. Drawing boxes would make the debug view much more useful.
- **CLIP proximity is estimated from confidence, not geometry**: Since CLIP returns no bounding box, proximity is approximated from confidence score. A more accurate approach would use depth estimation or bbox area.
- **NLG scene classification returns "unknown" frequently**: MobileNetV3 is limited to 4 scene types. A more capable scene classifier would improve description quality.
- **No `.env` or secrets handling**: `CLOUD_URL` is hardcoded in `config.py`. Use `python-dotenv` or environment variables for deployment flexibility.
- **No logging framework**: All output is `print()`. Replace with Python `logging` for level control and file output.
