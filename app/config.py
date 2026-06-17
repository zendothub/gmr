"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "RetailAIPlatform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30


    # Database
    DATABASE_URL: str = "postgresql+asyncpg://retail_user:retail_pass@localhost:5433/retail_ai_db"
    DATABASE_SYNC_URL: str = "postgresql://retail_user:retail_pass@localhost:5433/retail_ai_db"

    # Storage — MinIO (S3-compatible object storage)
    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    MINIO_BUCKET_PREFIX: str = "retail"
    # Object-name prefixes for categorising blobs inside the bucket.
    SNAPSHOT_DIR: str = "snapshots"
    CROP_DIR: str = "crops"
    CLIP_DIR: str = "clips"
    REPORT_DIR: str = "reports"

    # AI Models
    YOLO_MODEL_PATH: str = "models/yolo11n.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.45
    YOLO_ALLOWED_CLASSES: str = "0"  # comma-separated class IDs
    OSNET_MODEL_PATH: str = "models/osnet_x1_0.pth"
    REID_EMBEDDING_DIM: int = 512
    REID_MATCH_THRESHOLD: float = 0.60
    REID_CROP_QUALITY_THRESHOLD: float = 0.70
    REID_ACCUMULATION_FRAMES: int = 5
    REID_CONFIDENCE_LIMIT: float = 0.75
    REID_MAX_REFINEMENT_FRAMES: int = 20

    # InsightFace
    INSIGHTFACE_MODEL: str = "buffalo_l"
    INSIGHTFACE_DET_SIZE: str = "640,640"
    INSIGHTFACE_MAX_ATTEMPTS: int = 5
    FACE_MATCH_THRESHOLD: float = 0.50

    # Runtime
    DEFAULT_FPS_TARGET: int = 10
    MAX_WORKERS: int = 8
    FRAME_BUFFER_SIZE: int = 2
    RUNTIME_SHOW_GUI: bool = False

    # Worker robustness
    WORKER_WATCHDOG_TIMEOUT: int = 30
    WORKER_TRACKER_RESET_HOURS: int = 6
    WORKER_MAX_CRASH_RETRIES: int = 3

    # ------------------------------------------------------------------
    # Streaming (ffmpeg -> MediaMTX -> WebRTC/WHEP/HLS)
    # ------------------------------------------------------------------
    # Path to the ffmpeg binary used to republish RTSP into MediaMTX.
    FFMPEG_BINARY: str = "ffmpeg"
    # MediaMTX host as reachable from this backend (container/network name).
    MEDIAMTX_HOST: str = "localhost"
    # MediaMTX ports.
    MEDIAMTX_RTSP_PORT: int = 8554
    MEDIAMTX_WEBRTC_PORT: int = 8889   # WHEP (WebRTC-HTTP Egress Protocol)
    MEDIAMTX_HLS_PORT: int = 8888
    MEDIAMTX_API_PORT: int = 9997
    # Public base URL the browser uses to reach MediaMTX WebRTC/HLS.
    # When empty it is derived from MEDIAMTX_HOST + the relevant port.
    MEDIAMTX_PUBLIC_URL: str = ""
    # Transcode preset for the republish step. "copy" = no re-encode (cheapest,
    # requires H.264). Use "lowlatency" to re-encode for browser-friendly output.
    STREAM_PUBLISH_MODE: str = "lowlatency"  # copy | lowlatency
    # Auto-stop a published stream after this many seconds with no viewers.
    STREAM_IDLE_TIMEOUT_SECONDS: int = 120
    # Snapshot (single JPEG frame) settings for the zone-drawing canvas.
    SNAPSHOT_TIMEOUT_SECONDS: int = 10
    SNAPSHOT_JPEG_QUALITY: int = 85
    # Whether to write FFmpeg stream pipeline output to logs/stream_pipeline.log (otherwise devnull)
    STREAM_PIPELINE_LOG: bool = False
    # FPS for the burn-in annotated stream (bounding boxes + person count).
    # This is the output FPS fed to FFmpeg stdin; the AI pipeline runs independently.
    STREAM_BURNIN_FPS: int = 15


    # Logging
    LOG_LEVEL: str = "INFO"

    # SMTP / Email (for feature-request notifications and other alerts)
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASS: str = ""
    SMTP_USE_TLS: bool = True
    SMTP_FROM: str = ""
    NOTIFICATION_EMAIL: str = ""

    # CORS (comma-separated dashboard origins)
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    @property
    def yolo_allowed_classes_list(self) -> List[int]:
        return [int(c.strip()) for c in self.YOLO_ALLOWED_CLASSES.split(",") if c.strip()]

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    class Config:
        # .env overrides .env.example when present (local development)
        env_file = (".env.example", ".env")
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
