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

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://retail_user:retail_pass@localhost:5432/retail_ai_db"
    DATABASE_SYNC_URL: str = "postgresql://retail_user:retail_pass@localhost:5432/retail_ai_db"

    # Storage
    STORAGE_ROOT: str = "/app/storage"
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

    # Logging
    LOG_LEVEL: str = "INFO"

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
