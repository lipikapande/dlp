import time
import os
import cv2
from edge.capture import capture_snapshot
from edge.pipeline import run_pipeline
from edge.quality import assess_quality
from edge.compression import compress_to_jpeg_bytes
from edge.transport import upload_sync
from tts import speak

DEBUG_DIR = os.path.join(os.path.dirname(__file__), "..", "debug_captures")
os.makedirs(DEBUG_DIR, exist_ok=True)

def on_trigger():
    print("\n[main] --- Trigger received ---")

    # 1. Capture from webcam
    image = capture_snapshot()
    if image is None:
        speak("Camera error. Please check your webcam.")
        return

    print(f"[main] Captured: {image.shape}")

    # Save raw frame for inspection
    ts = time.strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(DEBUG_DIR, f"{ts}_raw.jpg")
    cv2.imwrite(raw_path, image)

    # 2. DIP pipeline
    processed = run_pipeline(image)

    # Save processed frame for inspection
    proc_path = os.path.join(DEBUG_DIR, f"{ts}_processed.jpg")
    cv2.imwrite(proc_path, processed)
    print(f"[main] Saved captures → {raw_path}")

    # 3. Quality gate
    report = assess_quality(processed)
    print(f"[main] Quality: blur={report.blur_score:.1f}, "
          f"brightness={report.brightness:.1f}, "
          f"snr={report.snr_db:.1f}dB, passed={report.passed}")

    if not report.passed:
        speak(f"Image rejected. {report.rejection_reason}. Please try again.")
        return

    # 4. Compress
    jpeg_bytes = compress_to_jpeg_bytes(processed, quality=75)
    print(f"[main] Compressed to {len(jpeg_bytes)/1024:.1f} KB")

    # 5. Send to local FastAPI server
    print("[main] Uploading to local server...")
    result = upload_sync(jpeg_bytes)

    if result is None:
        speak("Server not responding. Is Terminal 1 running?")
        return

    # 6. Speak the result
    description = result.get("description", "No description returned.")
    speak(description)
    print(f"[main] Done: {description}")

if __name__ == "__main__":
    print("=== Vision Assistant (Laptop Mode) ===")
    print("Press ENTER to capture. Ctrl+C to quit.\n")
    while True:
        try:
            input(">> Press ENTER to capture...")
            on_trigger()
        except KeyboardInterrupt:
            print("\n[main] Exiting.")
            break