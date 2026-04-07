from dataclasses import dataclass

@dataclass
class Config:
    # Quality gate thresholds
    BLUR_THRESHOLD: float = 70.0       # Laplacian variance — below = too blurry
    BRIGHTNESS_MIN: float = 30.0       # Mean pixel value (0-255)
    BRIGHTNESS_MAX: float = 225.0
    SNR_THRESHOLD: float = 0.0         # dB

    # DIP stages to enable (toggle for power profiling)
    ENABLE_FREQUENCY_FILTER: bool = True  # Expensive on Pi
    ENABLE_WIENER: bool = True  # was False
    ENABLE_SEGMENTATION: bool = True  # Only needed for feature export

    # Compression
    JPEG_QUALITY: int = 75             # 75 is good quality/size balance
    MAX_DIMENSION: int = 640           # Resize before upload

    # Cloud
    CLOUD_URL: str = "http://localhost:8000/infer"
    UPLOAD_TIMEOUT: float = 10.0

    # Camera
    CAMERA_RESOLUTION: tuple = (1280, 720)
    CAMERA_WARMUP_FRAMES: int = 20

CONFIG = Config()