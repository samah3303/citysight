"""CitySight Configuration — loaded from environment with sensible defaults."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Application configuration with env-var overrides."""

    # ── YOLO ────────────────────────────────────────────────────────────
    model_name: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.4"))
    # COCO classes we care about: person, bicycle, car, motorcycle, bus, truck
    target_classes: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 7])
    class_names: dict[int, str] = field(default_factory=lambda: {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
    })

    # ── Tracking ────────────────────────────────────────────────────────
    tracker_max_age: int = int(os.getenv("TRACKER_MAX_AGE", "30"))
    tracker_min_hits: int = int(os.getenv("TRACKER_MIN_HITS", "3"))
    tracker_iou_threshold: float = float(os.getenv("TRACKER_IOU_THRESHOLD", "0.3"))

    # ── Video input ─────────────────────────────────────────────────────
    video_source: str = os.getenv("VIDEO_SOURCE", "0")  # "0" = webcam, path to file, or YouTube URL
    frame_width: int = int(os.getenv("FRAME_WIDTH", "1280"))
    frame_height: int = int(os.getenv("FRAME_HEIGHT", "720"))
    target_fps: int = int(os.getenv("TARGET_FPS", "15"))
    websocket_interval_ms: int = int(os.getenv("WS_INTERVAL_MS", "100"))

    # ── Analytics ───────────────────────────────────────────────────────
    pixels_per_meter: float = float(os.getenv("PIXELS_PER_METER", "120.0"))
    heatmap_history: int = int(os.getenv("HEATMAP_HISTORY_FRAMES", "300"))
    density_low: int = int(os.getenv("DENSITY_LOW_THRESHOLD", "5"))
    density_medium: int = int(os.getenv("DENSITY_MEDIUM_THRESHOLD", "15"))
    # Above medium = high

    # ── Database (Neon PG) ──────────────────────────────────────────────
    database_url: str = os.getenv("DATABASE_URL", "")
    event_log_interval_seconds: int = int(os.getenv("EVENT_LOG_INTERVAL", "60"))

    # ── DeepSeek alerts ─────────────────────────────────────────────────
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    alert_check_frames: int = int(os.getenv("ALERT_CHECK_FRAMES", "150"))  # ~every 10s at 15fps
    alert_deviation_threshold: float = float(os.getenv("ALERT_DEVIATION_THRESHOLD", "0.3"))

    # ── Server ──────────────────────────────────────────────────────────
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))


# Singleton
config = Config()
