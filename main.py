import os
from edge.capture import capture_snapshot
from edge.pipeline import run_pipeline, stage_images
from edge.quality import assess_quality
from edge.compression import compress_to_jpeg_bytes
from edge.transport import upload_sync
from tts import speak


DEBUG_DIR = os.path.join(os.path.dirname(__file__), "debug_captures")
os.makedirs(DEBUG_DIR, exist_ok=True)


def on_trigger():
    print("\n[main] --- Trigger received ---")

    image = capture_snapshot()
    if image is None:
        speak("Camera error.")
        return

    processed, stages = run_pipeline(image)
    print(f"[main] Pipeline complete. Stages: {list(stages.keys())}")

    report = assess_quality(processed)
    print(f"[main] Quality: blur={report.blur_score:.1f}, "
          f"brightness={report.brightness:.1f}, passed={report.passed}")

    if not report.passed:
        print(f"[main] Quality warning: {report.rejection_reason} — sending anyway")
        speak(f"Image quality low: {report.rejection_reason}. Describing anyway.")

    jpeg_bytes = compress_to_jpeg_bytes(processed, quality=75)
    result = upload_sync(jpeg_bytes, stages, report)  # always reaches here now

    if result is None:
        speak("Server not responding.")
        return

    speak(result.get("description", "Could not describe scene."))


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