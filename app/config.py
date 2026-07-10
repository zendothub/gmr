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
    MINIO_PUBLIC_URL: str = ""
    MINIO_PUBLIC_ENDPOINT: str = ""
    MINIO_PUBLIC_SECURE: bool = True
    MINIO_REGION: str = "us-east-1"
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
    REID_MATCH_THRESHOLD: float = 0.50   # Body ReID match threshold. With MSMT17-trained OSNet weights, same-person
                                           # cross-camera median=0.680 (p10=0.393), diff-person median=0.386 (p90=0.534).
                                           # Best F1=0.49. 0.50 + 2-of-3 consensus gate gives clean separation.
                                           # Previously 0.85 (calibrated against broken ImageNet-backbone weights — see
                                           # CONTEXT.md issue #16, corrected 2026-07-09).
    REID_CROP_QUALITY_THRESHOLD: float = 0.30
    REID_ACCUMULATION_FRAMES: int = 5
    REID_CONFIDENCE_LIMIT: float = 0.75
    REID_MIN_QUALITY_FOR_SWITCH: float = 0.80
    REQUIRE_FACE_FOR_IDENTITY: bool = True

    # InsightFace
    INSIGHTFACE_MODEL: str = "buffalo_l"
    INSIGHTFACE_DET_SIZE: str = "640,640"
    FACE_MATCH_THRESHOLD: float = 0.40        # Positive face match threshold — same as dedup. Same-person cross-angle
                                                # 0.40-0.70, different-person 0.10-0.30. 0.40 catches all same-person
                                                # matches. False merges are corrected by dedup job (same threshold).
    FACE_CONTRADICTION_THRESHOLD: float = 0.25 # Disassociation threshold — only trigger if face is DEFINITELY different
                                                # (same person cross-angle ~0.40-0.47, so 0.25 avoids false disassociation)
    FACE_BODY_EXCLUSION_THRESHOLD: float = 0.30 # Body candidate exclusion gate — more permissive than match threshold
                                                 # allows body ReID to proceed when face is same-person but cross-angle
    FACE_MIN_DET_SCORE: float = 0.50          # Minimum InsightFace detection score for a face to be considered valid
    FACE_MIN_SIZE_PX: int = 30                # Minimum face width in pixels
    FACE_MIN_EYE_SPREAD: float = 0.25         # Minimum normalised eye-spread (eye_distance / face_width) for frontality
                                               # Frontal: ~0.35+, profile: ~0.0-0.15, 3/4: ~0.20-0.30
    FACE_FRONTALITY_WEIGHT: float = 0.35      # How much frontality score contributes to face_quality (0–1)
    FACE_CONTAMINATION_THRESHOLD: float = 0.35 # Running-consensus contamination gate: if a new face embedding has
                                                # cosine similarity < 0.35 to ALL previously accumulated faces for this
                                                # track, it is rejected as contamination (face from adjacent person).
                                                # Same person cross-angle range: 0.40–0.70+.  Different person: 0.10–0.30.
    BODY_CONTAMINATION_THRESHOLD: float = 0.50 # When storing a body embedding, if median cosine similarity to existing
                                                # body embeddings (>=3) is below this, reject as contamination. Also used
                                                # by periodic dedup cleanup to remove cluster outliers.
                                                # With MSMT17 OSNet weights: same-person median=0.680 (p25=0.537),
                                                # diff-person median=0.386 (p75=0.444). 0.50 cleanly separates with
                                                # good margin on both sides. Previously 0.60 (too close to same-person
                                                # p25=0.537, rejected valid cross-angle embeddings).

    FACE_IDENTITY_MIN_SCORE: float = 0.60     # Minimum face quality score required to create a new PersonIdentity
    FACE_IDENTITY_MIN_DETECTIONS: int = 2     # Minimum good face detections across track lifetime required for identity creation
    MAX_FACE_EMBEDDINGS_PER_PERSON: int = 5   # Maximum face embeddings stored per person identity (multi-angle)
    BODY_ONLY_CONFIDENCE_LIMIT: float = 0.95  # Body-only (no face) matches require higher confidence for re-identification
    # FACE_SEARCH_THRESHOLD intentionally removed — skip_body_reid logic has been eliminated (caused duplicate registrations)

    # ── Recent-window matching (plan 2026-07-09) ─────────────────────────────
    # Within a short window of a person's first_seen_at, body ReID is reliable
    # (same visit → same clothing) and the candidate pool is small. Allow a
    # relaxed face threshold + a single-candidate body override in that window
    # to catch same-person cross-angle/cross-camera handoffs that the strict
    # thresholds miss (which otherwise create duplicate identities that the
    # dedup job then has to merge).
    #
    # Outside the window: unchanged strict behaviour (clothing changes make body
    # ReID unreliable; larger candidate pool raises false-positive risk).
    ENABLE_RECENT_WINDOW_MATCHING: bool = True
    RECENT_WINDOW_MINUTES: int = 5
    FACE_MATCH_THRESHOLD_RECENT: float = 0.35   # relaxed face match within the window (strict 0.40 outside)
    RECENT_BODY_SINGLE_MATCH_THRESHOLD: float = 0.55  # single-candidate body override (median sim, ≥2 bodies each side, non-overlapping tracks on same camera, faces don't contradict at 0.25)
    FACE_MATCH_MEDIAN_THRESHOLD: float = 0.30  # When recent face best-pair is in grey zone [0.35, 0.40),
                                                 # require median of ALL cross-pairs >= this. Same-person
                                                 # min median=0.401, diff-person p50=0.200. At 0.30:
                                                 # 0% same-person rejected, 97.5% diff-person rejected.
                                                 # Only checked when >= 3 total cross-pairs.

    # ------------------------------------------------------------------
    # Staff detection — auto-classifies frequent visitors so purchase
    # analytics exclude employees (who generate hundreds of billing events
    # per shift).  Runs inside the periodic dedup job (every 10 min).
    # ------------------------------------------------------------------
    STAFF_DURATION_THRESHOLD_SECONDS: int = 1800   # total visible time across all sessions (default 30 min)
    STAFF_DISTINCT_DAYS_THRESHOLD: int = 3          # appeared on 3+ distinct calendar days

    # ------------------------------------------------------------------
    # SigLIP2 — zero-shot gender classifier (pre-computed text embeddings)
    # ------------------------------------------------------------------
    SIGLIP2_MODEL_ID: str = "google/siglip2-base-patch16-224"  # zero-shot gender (~18ms/img, ~1.4GB GPU)

    # ------------------------------------------------------------------
    # MiVOLO — gender + age (replaces InsightFace demographics)
    # ------------------------------------------------------------------
    MIVOLO_MODEL_PATH: str = "models/mivolo/mivolo_fairface.pth.tar"  # MiVOLO-D1 IMDB (ViT-Small, 3-class, ~103 MB, best of 4)
    
    # YOLO-Pose for enhanced ReID quality assessment
    YOLO_POSE_MODEL_PATH: str = "models/yolo11n-pose.pt"
    YOLO_POSE_CONFIDENCE: float = 0.3  # Keypoint confidence threshold

    # Runtime
    DEFAULT_FPS_TARGET: int = 10
    MAX_WORKERS: int = 10
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
    SMTP_FROM: str = "abdur@zendot.in"          # source / sender address
    NOTIFICATION_EMAIL: str = "tech@zendot.in"  # destination address for feature-request alerts

    # CORS (comma-separated dashboard origins)
    CORS_ORIGINS: str = (
        "http://localhost:3000,"
        "http://localhost:5173,"
        "http://10.69.154.89,"
        "http://10.69.154.89:80,"
        "http://10.69.154.89:3000,"
        "http://10.69.154.89:5173,"
        "http://10.69.154.89:8080,"
        "http://10.8.0.2,"
        "https://10.8.0.2,"
        "http://10.8.0.2:80,"
        "http://10.8.0.2:3000,"
        "http://10.8.0.2:5173,"
        "http://10.8.0.2:8080,"
        "http://retaileye.bluecloudsoftech.com,"
        "https://retaileye.bluecloudsoftech.com"
    )

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
