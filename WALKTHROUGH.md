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
5. Generates a natural-language description and **speaks it aloud**
6. Shows all pipeline stages visually in a dark-mode web UI

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
    ├── detector.py          # YOLOv8 object detection (loaded once at startup)
    ├── classifier.py        # MobileNetV3 scene classification
    ├── nlg.py               # Template-based natural language generation
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
   │  02 Denoise          (bilateral)   — only if noisy AND not blurry
   │  03 CLAHE            (contrast)    — only if dark (brightness < 130)
   │  04 Gamma            (brighten)    — only if dark (brightness < 140)
   │  05 Butterworth      (low-pass)    — only if blurry (blur_score < 100)
   │  06 Canny edges      (always)
   │  07 Threshold        (always)
   │  08 Wiener restore   — only if very noisy (SNR < 3 dB)
   │  09 Sharpen          — only if blur_score < 200
   │  10 Final output
   │
   ▼
edge/quality.py          → QualityReport: blur (Laplacian), brightness (mean), SNR (dB)
   │
   ▼
edge/compression.py      → JPEG encode at quality=75
   │
   ▼
edge/transport.py        → POST JSON to http://localhost:8000/infer
                           payload: { image: base64, stages: {key: base64}, quality: {...} }
   │
   ▼
cloud/server.py /infer   → decodes image, calls detector + classifier + NLG
   │
   ├─ cloud/detector.py  → YOLOv8n — returns [{label, confidence, bbox}, ...]
   ├─ cloud/classifier.py→ MobileNetV3 — returns "indoor" | "outdoor" | "street" | "unknown"
   └─ cloud/nlg.py       → builds sentence: "You are indoors. I can see a person (92%)..."
   │
   ▼
SSE broadcast → browser (ui.html) updates live
   │
   ▼
tts.py (edge)            → pyttsx3 speaks the description aloud
```

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

| Parameter                 | Default                | What it does                                   |
| ------------------------- | ---------------------- | ---------------------------------------------- |
| `BLUR_THRESHOLD`          | 70.0                   | Laplacian variance below this = blurry warning |
| `BRIGHTNESS_MIN/MAX`      | 30 / 225               | Mean pixel brightness range                    |
| `SNR_THRESHOLD`           | 0.0 dB                 | Noise gate                                     |
| `ENABLE_FREQUENCY_FILTER` | True                   | Butterworth FFT filter (slow — disable on Pi)  |
| `ENABLE_WIENER`           | True                   | Wiener deblur (only fires if SNR < 3 dB)       |
| `JPEG_QUALITY`            | 75                     | Upload compression quality                     |
| `MAX_DIMENSION`           | 640                    | Resize longest side to this before upload      |
| `CLOUD_URL`               | `localhost:8000/infer` | Change for remote server                       |
| `CAMERA_RESOLUTION`       | 1280×720               | Webcam capture resolution                      |
| `CAMERA_WARMUP_FRAMES`    | 20                     | Frames burned for auto-exposure to settle      |

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

| Stage          | Technique                                                     | When it fires                 |
| -------------- | ------------------------------------------------------------- | ----------------------------- |
| 02 Denoise     | Bilateral filter (edge-preserving)                            | SNR < 5 dB **and** blur > 120 |
| 03 CLAHE       | Contrast Limited Adaptive Histogram Equalization on L channel | Brightness < 130              |
| 04 Gamma       | Fixed γ=1.5 brightening via LUT                               | Brightness < 140              |
| 05 Butterworth | Low-pass FFT filter (removes high-freq noise)                 | blur_score < 100              |
| 06 Canny edges | Edge detection (always, for demo)                             | Always                        |
| 07 Threshold   | Binary threshold at 127 (always, for demo)                    | Always                        |
| 08 Wiener      | Frequency-domain deblurring                                   | SNR < 3 dB                    |
| 09 Sharpen     | Unsharp mask via weighted Gaussian                            | blur_score < 200              |

---

## What Could Be Improved

### Code Quality

- **`auto_gamma` is hardcoded** (`intensity.py:42`): The adaptive version using `mean_brightness` is commented out and replaced with a fixed `gamma=1.5`. The dynamic version is strictly better — restore it.
- **`segmentation.py` is never called**: `apply_canny_edges`, `apply_morphology`, and `extract_contour_features` exist but the pipeline uses raw `cv2.Canny` inline instead. Either use the module or delete it.
- **`edge/stages/spatial.py` is imported but unused**: `apply_gaussian` is imported in `pipeline.py` but never called.
- **`cloud/tts.py` is a dead placeholder**: It imports `gtts` which isn't in `requirements.txt`. Either wire it up or delete the file.
- **Duplicate `/trigger` and `/upload` logic** in `server.py`: Both routes repeat the same 30-line encode/detect/describe/broadcast block that's already in `/infer`. Extract a helper.
- **Global mutable `_latest` and `_subscribers`** in `server.py`: Works fine for single-user local use but will break under concurrency. Use a proper state container if scaling.

### Robustness

- **No `__init__.py` files** in `edge/` or `cloud/`: Works due to Python path tricks but will break if packaged or if imports are run from a different working directory.
- **`asyncio.run()` inside `upload_sync`** (`transport.py:44`): Will crash if called from inside an already-running event loop (e.g., a Jupyter notebook or if ever called from inside FastAPI). Use `httpx` sync client instead.
- **Camera opened/released on every capture** (`capture.py`): Opening `VideoCapture(0)` on every press adds ~0.5s latency. Keep it open and release on `KeyboardInterrupt`.
- **TTS engine re-initialized if an exception occurs** (`tts.py`): If `runAndWait()` throws, `_engine` stays set to the broken instance. Reset to `None` in the `except` block.

### Performance

- **Butterworth FFT runs per-channel in Python loops** (`frequency.py:26`): Can be vectorized with `np.fft.fftn` across channels — ~3× faster.
- **YOLOv8 and MobileNet both run on every request**: For a demo, consider running YOLO only and skipping MobileNet scene classification (which returns "unknown" most of the time anyway).
- **`CAMERA_WARMUP_FRAMES=20`** causes ~0.5–1s of wasted reads on every capture. 5–10 frames is sufficient for most USB webcams.

### Features

- **No bounding box overlay on the final image**: Detections have `bbox` coordinates but they're never drawn on the image shown in the UI. Drawing boxes would make the debug view much more useful.
- **NLG descriptions are very simple** (`nlg.py`): Only 4 scene types and a label list. Plugging in a small LLM (e.g., `ollama` locally) for the description step would dramatically improve output quality.
- **No `.env` or secrets handling**: `CLOUD_URL` is hardcoded in `config.py`. Use `python-dotenv` or environment variables for deployment flexibility.
- **No logging framework**: All output is `print()`. Replace with Python `logging` for level control and file output.
