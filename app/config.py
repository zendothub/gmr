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
                                           # Live path: gallery body MEDIAN ≥0.50 (n_bodies≥2) + BODY_MATCH_AMBIGUITY;
                                           # not unique-person 2-of-3 votes (structurally dead — CONTEXT #25, 2026-07-17).
                                           # Previously 0.85 (broken ImageNet-backbone weights — CONTEXT #16).
    REID_CROP_QUALITY_THRESHOLD: float = 0.30
    REID_ACCUMULATION_FRAMES: int = 5
    REID_CONFIDENCE_LIMIT: float = 0.75
    REID_MIN_QUALITY_FOR_SWITCH: float = 0.80
    REQUIRE_FACE_FOR_IDENTITY: bool = True
    # Body-only identity create (FEATURE): faceless track may create a person
    # when recent-window body is strong quality and far from all recent galleries
    # (esp. staff). Explicit bypass of REQUIRE_FACE_FOR_IDENTITY with logs.
    # Keep confidence non-confident (no face). Never body-create outside window
    # isolation (nearest/staff gates use recent gallery only).
    ENABLE_BODY_ONLY_IDENTITY_CREATE: bool = True
    BODY_ONLY_CREATE_MIN_QUALITY: float = 0.55
    BODY_ONLY_CREATE_MAX_NEAREST_SIM: float = 0.45   # nearest recent person median must be < this
    BODY_ONLY_CREATE_MAX_STAFF_SIM: float = 0.48     # best staff median (full gal if active) must be < this

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

    # ── Recent-window matching (plan 2026-07-09; body gallery 2026-07-20) ────
    # Body ReID is clothing-dependent. Customer body match uses ONLY body
    # embeddings with captured_at inside RECENT_WINDOW_MINUTES (same visit /
    # outfit). Older bodies stay stored — mismatch does NOT delete them; we
    # simply do not merge. Activity-recent = track_sessions overlap OR any
    # body emb in the window (not stale person.last_seen alone).
    # Staff reattach: activity-recent + FULL lifetime body gallery (uniform).
    ENABLE_RECENT_WINDOW_MATCHING: bool = True
    RECENT_WINDOW_MINUTES: int = 5
    # Customer body ANN + median gallery restricted to this window.
    BODY_MATCH_USE_RECENT_GALLERY_ONLY: bool = True
    # Staff body median / reattach may use all stored bodies if activity-recent.
    STAFF_BODY_USE_FULL_GALLERY: bool = True
    FACE_MATCH_THRESHOLD_RECENT: float = 0.35   # relaxed face match within the window (strict 0.40 outside)
    RECENT_BODY_SINGLE_MATCH_THRESHOLD: float = 0.55  # single-candidate body override (median sim, ≥2 recent bodies, face non-contradiction)
    # Reject body match if top-2 person medians are within this gap (ambiguous clothing / uniforms).
    BODY_MATCH_AMBIGUITY: float = 0.03
    FACE_MATCH_MEDIAN_THRESHOLD: float = 0.30  # When recent face best-pair is in grey zone [0.35, 0.40),
                                                 # require median of ALL cross-pairs >= this. Same-person
                                                 # min median=0.401, diff-person p50=0.200. At 0.30:
                                                 # 0% same-person rejected, 97.5% diff-person rejected.
                                                 # Only checked when >= 3 total cross-pairs.
    # Require median face sim to candidate's full gallery when gallery size ≥ 2
    # (blocks lucky best-pair into mixed / multi-person face galleries).
    # Aligns with FACE_CONTAMINATION_THRESHOLD (0.35).
    FACE_MATCH_CLUSTER_MEDIAN_THRESHOLD: float = 0.35
    ENABLE_FACE_MATCH_CLUSTER_MEDIAN: bool = True
    # On face contradiction, never rematch the same person_id (force new person
    # or a different identity). Fixes c7bdce30-style self-rematch loops.
    ENABLE_CONTRADICTION_SAME_ID_BLOCK: bool = True

    # ── Phase 1: Occlusion-aware face assignment + tight body crops (2026-07-10) ──
    # Side-by-side / occlusion is the remaining contamination source (CONTEXT #24).
    # Detect overlapping tracks, harden immature face geometry, Hungarian assign,
    # skip OSNet on occluded frames, and use tight YOLO boxes for body ReID.
    OCCLUSION_IOU_THRESHOLD: float = 0.10       # Pairwise body IoU ≥ this → both tracks occluded
    FACE_ASSIGN_UPPER_BODY_FRAC: float = 0.45   # Immature tracks: face centre must be in top 45% of body height
    FACE_ASSIGN_MIN_OVERLAP: float = 0.70       # Immature: fraction of face area inside body bbox
    FACE_ASSIGN_MIN_SCORE_IMMATURE: float = 0.35  # Immature track minimum composite face score
    FACE_ASSIGN_AMBIGUITY_RATIO: float = 0.85   # If 2nd-best track score / best ≥ this for same face → assign to neither
    ENABLE_HUNGARIAN_FACE_ASSIGN: bool = True   # False → greedy sort (legacy)
    SKIP_BODY_REID_WHEN_OCCLUDED: bool = True   # Do not extract OSNet body embeddings on occluded frames
    BODY_CROP_PADDING_PCT: float = 0.0          # Body crop padding for OSNet (0.0 = tight YOLO box; raise via env if needed)

    # ── Same-camera temporal overlap gate (2026-07-13) ─────────────────
    # Two track sessions on the SAME camera that overlap in time cannot be the
    # same physical person. Cross-camera overlap (entry + counter) is allowed.
    # Applied in live decide_identity (face/body/staff reattach) and periodic
    # dedup before merging. ε ignores 1-frame ByteTrack glitches.
    ENABLE_SAME_CAMERA_OVERLAP_GATE: bool = True
    SAME_CAMERA_OVERLAP_MIN_SECONDS: float = 1.0

    # ------------------------------------------------------------------
    # Staff detection — auto-classifies frequent visitors so purchase
    # analytics exclude employees (who generate hundreds of billing events
    # per shift).  Runs inside the periodic dedup job (every 10 min).
    # ------------------------------------------------------------------
    STAFF_DURATION_THRESHOLD_SECONDS: int = 1800   # total visible time across all sessions (default 30 min)
    STAFF_DISTINCT_DAYS_THRESHOLD: int = 3          # appeared on 3+ distinct calendar days

    # ── Staff reattach (2026-07-10) ─────────────────────────────────────
    # When face match fails (blur/side face) but body strongly matches an is_staff
    # identity within RECENT_WINDOW_MINUTES, reattach instead of creating a new person.
    # Face quality veto is stubbed in code (commented) until quality metric is calibrated.
    ENABLE_STAFF_REATTACH: bool = True
    STAFF_REATTACH_BODY_MEDIAN: float = 0.70      # median body vs staff (was 0.55 — uniform FPs at 0.55–0.65)
    STAFF_REATTACH_MIN_BODIES: int = 2            # staff must have ≥2 stored body embeddings
    STAFF_REATTACH_FACE_MIN: float = 0.30         # face_sim below this → reject (was 0.20 — too loose)
    STAFF_REATTACH_REQUIRE_FACE: bool = True      # faceless tracks never reattach via body alone
    STAFF_REATTACH_AMBIGUITY: float = 0.03        # reject if top-2 staff body medians within this gap
    # STAFF_REATTACH_FACE_QUALITY_HIGH: float = 0.75  # FUTURE — high-quality face + low sim = hard veto

    # ── Billing visit repair (post-dedup; 2026-07-31) ─────────────────
    # ByteTrack splits one counter stay into N track_sessions; dwell resets
    # each time so no fragment may hit dwell_threshold alone (missed BI),
    # or each can fire separately. Live _refresh/_backfill only same session.
    # After dedup: fill null BI person, body/face-stitch null billing-zone
    # sessions (same cam, gap≤60s), then group by person+cam and sum dwell.
    # Body stitch thr high (0.80); face thr 0.40; no staff attach target.
    ENABLE_BILLING_VISIT_REPAIR: bool = True
    BILLING_VISIT_LOOKBACK_HOURS: int = 48
    BILLING_VISIT_STITCH_GAP_SECONDS: float = 60.0   # max gap between fragments / null stitch (1 min)
    BILLING_VISIT_DEFAULT_DWELL_THRESHOLD: float = 25.0  # fallback if rule thr null
    # Null-session stitch onto nearby same-camera person (billing zone only).
    # Body thr high (0.80) — CONTEXT audit; face 0.40 matches live thr.
    # Reject same-camera time overlap (different people). No staff attach.
    ENABLE_BILLING_VISIT_BODY_STITCH: bool = True
    ENABLE_BILLING_VISIT_FACE_STITCH: bool = True
    BILLING_VISIT_STITCH_BODY_MEDIAN: float = 0.80
    BILLING_VISIT_STITCH_BODY_MIN_BODIES: int = 1
    BILLING_VISIT_STITCH_BODY_AMBIGUITY: float = 0.05
    BILLING_VISIT_STITCH_FACE_THRESHOLD: float = 0.40
    BILLING_VISIT_STITCH_FACE_CONTRADICTION: float = 0.25
    BILLING_VISIT_STITCH_MAX_STAFF_BODY: float = 0.70  # reject if staff body med ≥ this

    # ------------------------------------------------------------------
    # SigLIP2 — zero-shot gender classifier (pre-computed text embeddings)
    # ------------------------------------------------------------------
    SIGLIP2_MODEL_ID: str = "google/siglip2-base-patch16-224"  # zero-shot gender (~18ms/img, ~1.4GB GPU)
    # Female-biased decision boundary: predict M only if (male_best − female_best) > δ.
    # Empirically δ=0.5 → ~98% on stored faces, fixes most F→M, body path disabled.
    SIGLIP2_GENDER_MARGIN_DELTA: float = 0.5
    # Body crops bias toward male on this CCTV — keep face-only for gender.
    SIGLIP2_USE_BODY_FOR_GENDER: bool = False

    # Age: InsightFace buffalo_l genderage head (median over track face samples).
    # MiVOLO weights may remain on disk under models/mivolo/ but are not loaded.

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
    # Target video bitrate for the browser overlay stream (burn-in) and
    # lowlatency republish step.  1200k is plenty for annotated 1080p CCTV.
    # Without this, FFmpeg defaults to CRF-based quality encoding which
    # produces 5-8 Mbps — 4-6× heavier than necessary.
    STREAM_BITRATE: str = "1200k"
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

    # ── Device session tracking ────────────────────────────────────────
    # Inactive device sessions (no API activity) expire after this many seconds.
    SESSION_IDLE_TIMEOUT_SECONDS: int = 1800  # 30 minutes

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
