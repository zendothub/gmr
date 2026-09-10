# Mine Attendance Module — Architecture Document

**Author:** Technical Architecture Review  
**Date:** 2026-09-03  
**Status:** Draft for Manager Review  

---

## 1. Executive Summary

This document describes the architecture for adding a **Mine Attendance module** to the existing RetailEye backend (`gmr`). The system will monitor 14 CP Plus cameras at a coal mine site over a WireGuard VPN tunnel, perform face recognition against registered mine workers, and produce entry/exit attendance logs accessible via REST API and the existing RetailEye dashboard.

The core recognition pipeline (InsightFace, pgvector, asyncio workers) already exists in this codebase. The Mine Attendance module is an extension — not a replacement — that adds worker registration, attendance logging, and shift management on top of the proven infrastructure.

---

## 2. System Overview

### 2.1 High-Level Data Flow

```
Coal Mine Site (192.168.31.0/24)
  ├── Camera 01 (192.168.31.15) ─────┐
  ├── Camera 02 (192.168.31.16) ─────┤
  │         ...                       ├── WireGuard VPN ──► EC2 (13.234.173.13)
  ├── Camera 13 (192.168.31.27) ─────┤       │
  └── Camera 14 (192.168.31.28) ─────┘       │
                                              ▼
                                   ┌─────────────────────┐
                                   │  MineAttendance      │
                                   │  Supervisor          │
                                   │  (asyncio)           │
                                   └────────┬────────────┘
                                            │ per-camera tasks
                                   ┌────────▼────────────┐
                                   │  Frame Sampler       │
                                   │  (cv2 / FFmpeg)      │
                                   │  1-2 fps             │
                                   └────────┬────────────┘
                                            │ raw frames
                                   ┌────────▼────────────┐
                                   │  InsightFace         │
                                   │  buffalo_l           │
                                   │  detect + embed      │
                                   └────────┬────────────┘
                                            │ 512-dim vectors
                                   ┌────────▼────────────┐
                                   │  pgvector cosine     │
                                   │  similarity search   │
                                   │  vs worker gallery   │
                                   └────────┬────────────┘
                                            │ matched worker_id + confidence
                                   ┌────────▼────────────┐
                                   │  Attendance Logger   │
                                   │  (entry/exit +       │
                                   │   cooldown gate)     │
                                   └────────┬────────────┘
                                            │
                              ┌─────────────┼──────────────┐
                              ▼             ▼              ▼
                         PostgreSQL      MinIO          REST API
                      (attendance      (face crops)   (dashboard)
                         logs)
```

### 2.2 Network Topology

| Component | Address | Notes |
|-----------|---------|-------|
| EC2 Public IP | 13.234.173.13 | WireGuard endpoint |
| EC2 VPN IP | 10.0.0.1 | WireGuard interface |
| Mine Site VPN IP | 10.0.0.2 | WireGuard peer |
| Mine Camera Subnet | 192.168.31.0/24 | Routed through VPN |
| Example Camera | 192.168.31.15:554 | RTSP, H.265, 1080p@30fps |

RTSP URL pattern:
```
rtsp://admin:Cctv%40123@192.168.31.{X}:554/video/live?channel=1&subtype=0
```
The `subtype=0` selects main stream (1080p). Use `subtype=1` for sub-stream (360p) if bandwidth is a concern during face sampling.

---

## 3. How It Fits Into the Existing Codebase

### 3.1 What Already Exists (Reused)

| Existing Component | Location | How Reused |
|-------------------|----------|------------|
| `Camera` model | `app/core/db/models/camera.py` | Add mine cameras to same table with `site_type='mine'` flag |
| InsightFace pipeline | `app/modules/ai_runtime/` | Same buffalo_l model for face detection + 512-dim embedding |
| `PersonFaceEmbedding` table | `app/core/db/models/person.py` | Parallel `MineWorkerFaceEmbedding` with identical vector(512) pattern |
| pgvector cosine search | Used in existing ReID | Same `<=>` operator for worker gallery search |
| `CameraRecorder` / FFmpeg | `app/modules/recording/recorder.py` | Frame sampling uses same RTSP TCP connection pattern |
| `WorkerSupervisor` pattern | `app/modules/ai_runtime/` | `MineAttendanceSupervisor` follows identical supervisor singleton pattern |
| MinIO client | `app/modules/storage/minio_client.py` | Face crop storage |
| Auth / JWT | `app/modules/auth/` | All new endpoints use same `get_current_user` dependency |
| `AsyncSessionLocal` | `app/core/db/session.py` | All new async DB ops use same session factory |
| APScheduler worker | `app/worker.py` | Shift auto-close job added to same scheduler |
| Alembic migrations | `alembic/versions/` | New migration for mine tables |

### 3.2 New Module Structure

```
app/
└── modules/
    └── mine_attendance/
        ├── __init__.py
        ├── router.py          # FastAPI routes
        ├── schemas.py         # Pydantic request/response models
        ├── service.py         # Business logic (enrollment, attendance)
        ├── supervisor.py      # MineAttendanceSupervisor (asyncio singleton)
        ├── frame_worker.py    # Per-camera frame sampling + face matching
        └── shift_jobs.py      # APScheduler jobs (shift auto-close, reports)

app/core/db/models/
├── mine_worker.py             # MineWorker, MineWorkerFaceEmbedding
├── mine_shift.py              # MineShift
└── mine_attendance_log.py     # MineAttendanceLog
```

The router is registered in `app/main.py` exactly like the existing `recording_router`.

---

## 4. Database Models

### 4.1 `mine_workers`

```sql
CREATE TABLE mine_workers (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    employee_id     VARCHAR(50) UNIQUE NOT NULL,   -- company-issued ID
    name            VARCHAR(255) NOT NULL,
    department      VARCHAR(100),
    designation     VARCHAR(100),
    phone           VARCHAR(20),
    enrollment_status  VARCHAR(20) DEFAULT 'pending',  -- pending | enrolled | failed
    enrolled_at     TIMESTAMPTZ,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    metadata_json   JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### 4.2 `mine_worker_face_embeddings`

Mirrors `person_face_embeddings` exactly. One worker may have up to 5 embeddings (different angles, lighting).

```sql
CREATE TABLE mine_worker_face_embeddings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    worker_id       UUID NOT NULL REFERENCES mine_workers(id) ON DELETE CASCADE,
    embedding       VECTOR(512) NOT NULL,   -- InsightFace buffalo_l
    face_score      FLOAT NOT NULL DEFAULT 0.0,
    face_crop_path  VARCHAR(500),           -- MinIO object key
    captured_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX mine_worker_face_emb_worker_idx ON mine_worker_face_embeddings(worker_id);
-- pgvector HNSW index for ANN search
CREATE INDEX mine_worker_face_emb_vector_idx
    ON mine_worker_face_embeddings
    USING hnsw (embedding vector_cosine_ops);
```

### 4.3 `mine_shifts`

```sql
CREATE TABLE mine_shifts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(100) NOT NULL,   -- "Morning", "Afternoon", "Night"
    shift_date      DATE NOT NULL,
    start_time      TIMESTAMPTZ NOT NULL,
    end_time        TIMESTAMPTZ NOT NULL,
    site_name       VARCHAR(100) DEFAULT 'GMR Mine',
    is_closed       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

Shifts are auto-created daily by APScheduler (configured in `shift_jobs.py`). Three standard shifts: Morning (06:00–14:00), Afternoon (14:00–22:00), Night (22:00–06:00).

### 4.4 `mine_attendance_logs`

```sql
CREATE TABLE mine_attendance_logs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    worker_id       UUID NOT NULL REFERENCES mine_workers(id),
    camera_id       UUID NOT NULL REFERENCES cameras(id),
    shift_id        UUID REFERENCES mine_shifts(id),
    event_type      VARCHAR(10) NOT NULL,   -- 'entry' | 'exit'
    detected_at     TIMESTAMPTZ NOT NULL,
    confidence      FLOAT NOT NULL,         -- cosine similarity score
    face_crop_path  VARCHAR(500),           -- MinIO object key
    raw_bbox        JSONB,                  -- {x1,y1,x2,y2} in original frame
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX mine_att_worker_idx ON mine_attendance_logs(worker_id, detected_at DESC);
CREATE INDEX mine_att_camera_idx ON mine_attendance_logs(camera_id, detected_at DESC);
CREATE INDEX mine_att_shift_idx  ON mine_attendance_logs(shift_id);
```

### 4.5 `cameras` table extension

Add two columns to the existing `cameras` table via Alembic migration:

```sql
ALTER TABLE cameras ADD COLUMN site_type VARCHAR(20) DEFAULT 'retail';
-- 'retail' | 'mine'
ALTER TABLE cameras ADD COLUMN mine_location VARCHAR(100);
-- e.g. "Main Gate", "Shaft Entry", "Conveyor Exit"
```

Mine cameras are registered as `Camera` rows with `site_type='mine'`. This reuses all existing camera management APIs (add, list, status, activate/deactivate).

---

## 5. New API Endpoints

All endpoints are prefixed `/mine/` and require JWT authentication.

### 5.1 Worker Registration & Enrollment

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/mine/workers/` | Register new mine worker |
| `GET` | `/mine/workers/` | List workers (filter: dept, status, is_active) |
| `GET` | `/mine/workers/{worker_id}` | Get worker detail + enrollment status |
| `PUT` | `/mine/workers/{worker_id}` | Update worker info |
| `DELETE` | `/mine/workers/{worker_id}` | Deactivate worker |
| `POST` | `/mine/workers/{worker_id}/enroll` | Upload face images for enrollment (multipart, up to 5 images) |
| `DELETE` | `/mine/workers/{worker_id}/embeddings` | Clear face embeddings (re-enroll) |

**Enrollment flow (`POST /mine/workers/{id}/enroll`):**
1. Accept up to 5 JPEG/PNG uploads in a single multipart request.
2. Run InsightFace detection on each image.
3. Reject images where no face detected, face score < 0.5, or face width < 80px.
4. Store passing embeddings in `mine_worker_face_embeddings`.
5. Save face crops to MinIO under `mine-faces/{worker_id}/`.
6. Set `enrollment_status = 'enrolled'` when at least 2 embeddings stored.

### 5.2 Attendance & Shifts

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/mine/attendance/` | Attendance logs (filters: worker_id, shift_id, date, event_type, camera_id) |
| `GET` | `/mine/attendance/live` | Workers currently on-site (last seen < 4h, most recent event = entry) |
| `GET` | `/mine/attendance/report` | Daily/shift attendance summary (present/absent per worker) |
| `GET` | `/mine/attendance/report/export` | CSV export of attendance report |
| `GET` | `/mine/shifts/` | List shifts (filter: date range, is_closed) |
| `POST` | `/mine/shifts/` | Manually create shift |
| `GET` | `/mine/shifts/{shift_id}` | Shift detail + attendance summary |
| `GET` | `/mine/cameras/` | Mine cameras with live status (running/stopped) |
| `POST` | `/mine/cameras/{camera_id}/start` | Start face detection worker for a camera |
| `POST` | `/mine/cameras/{camera_id}/stop` | Stop face detection worker for a camera |

---

## 6. RTSP Ingestion Pipeline Design

### 6.1 MineAttendanceSupervisor

Singleton (same pattern as `WorkerSupervisor`). Manages one `asyncio.Task` per mine camera.

```
MineAttendanceSupervisor.get_instance()
  ├── start_camera(camera_id)   →  spawns MineFrameWorker task
  ├── stop_camera(camera_id)    →  cancels task, joins
  ├── stop_all()                →  used on app shutdown
  └── status()                  →  dict of camera_id → {running, fps, last_frame_at}
```

Registered in `app/lifecycle.py` alongside `WorkerSupervisor` and `RecordingSupervisor`.

### 6.2 MineFrameWorker (per-camera asyncio task)

```
Loop:
  1. Open RTSP via cv2.VideoCapture (rtsp_transport=tcp)
  2. Read frames
  3. Subsample: process 1 frame every N seconds (default: N=2, configurable)
  4. Run InsightFace face detection on subsampled frame
  5. For each detected face with score >= threshold:
       a. Extract 512-dim embedding
       b. Query pgvector gallery for nearest worker embedding
       c. If cosine_similarity >= MINE_FACE_MATCH_THRESHOLD (0.40):
            - Apply cooldown gate (skip if same worker logged < COOLDOWN_SECONDS ago on this camera)
            - Determine event_type (entry/exit) from camera.mine_location
            - Write MineAttendanceLog row
            - Save face crop to MinIO (async, non-blocking)
  6. On cv2 read failure: sleep RECONNECT_DELAY_SECONDS (5s), retry
```

### 6.3 Frame Sampling Rate

| Scenario | Frames/sec sampled | Rationale |
|----------|--------------------|-----------|
| Entry/exit gates | 0.5 fps (1 frame/2s) | Workers walk through; 2s window is sufficient |
| Shift change (peak) | 1 fps | Higher density, faster sampling |

At 0.5 fps × 14 cameras = 7 inference ops/sec. InsightFace buffalo_l on T4 GPU: ~15ms/face → comfortably real-time.

### 6.4 CP Plus H.265 Decoding

CP Plus cameras stream H.265 (HEVC). OpenCV on most Linux builds does not include H.265 hardware decode by default. Two options:

**Option A (recommended): FFmpeg subprocess pipe**
```python
cmd = [
    "ffmpeg", "-rtsp_transport", "tcp",
    "-i", rtsp_url,
    "-vf", f"fps=1/{frame_interval}",
    "-f", "rawvideo", "-pix_fmt", "bgr24", "-"
]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
# Read raw BGR frames from stdout pipe
```
FFmpeg handles H.265 decode natively. Frame interval is applied at the FFmpeg level (no wasted decode cycles).

**Option B: cv2 with ffmpeg backend**
```python
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
```
Works if OpenCV was built with FFmpeg support (the `opencv-python-headless` wheel included in requirements.txt includes FFmpeg).

Use Option A for reliability. The `CameraRecorder` in this codebase already follows the subprocess pattern successfully.

---

## 7. Face Detection and Recognition Approach

### 7.1 Model Choice: InsightFace buffalo_l

**Already in the codebase.** Do not add DeepFace or a second face recognition library.

| Property | Value |
|----------|-------|
| Model | InsightFace buffalo_l |
| Embedding dim | 512 |
| Detection | RetinaFace (included in buffalo_l pack) |
| Recognition | ArcFace ResNet-100 |
| Det size | 640×640 (existing `INSIGHTFACE_DET_SIZE` config) |
| Matching metric | Cosine similarity via pgvector `<=>` operator |

### 7.2 Face Quality Gates (reuse existing thresholds)

| Gate | Setting | Value |
|------|---------|-------|
| Min detection score | `FACE_MIN_DET_SCORE` | 0.50 |
| Min face width | `FACE_MIN_SIZE_PX` | 30px (raise to 60px for mine use — helmets occlude) |
| Match threshold | `MINE_FACE_MATCH_THRESHOLD` (new) | 0.40 |
| Enrollment min score | — | 0.60 |

### 7.3 Gallery Search Query

```sql
SELECT
    mwfe.worker_id,
    1 - (mwfe.embedding <=> :query_embedding) AS similarity
FROM mine_worker_face_embeddings mwfe
JOIN mine_workers mw ON mw.id = mwfe.worker_id
WHERE mw.is_active = TRUE
  AND mw.enrollment_status = 'enrolled'
ORDER BY mwfe.embedding <=> :query_embedding
LIMIT 5;
```

Then in Python: group by `worker_id`, take the max similarity per worker. If max similarity >= 0.40, it's a match.

### 7.4 Helmet / PPE Challenge

Mine workers wear helmets which partly occlude the upper face. Mitigation:
- Set enrollment to capture face angles with helmet ON (HR to conduct enrollment session in mine gear).
- Lower `FACE_MIN_EYE_SPREAD` for mine enrollment (helmet brim pushes the face lower in frame).
- Reject detections where face area < 60×60px (too far/small for reliable match under a helmet).

---

## 8. Attendance Logic

### 8.1 Entry/Exit Determination

Two strategies; recommend **Strategy A** for simplicity:

**Strategy A — Camera location tag (recommended)**  
Each mine camera has `mine_location` set in the database (e.g., "Main Gate Entry", "Shaft Entry", "Conveyor Exit"). The `event_type` is fixed per camera:
- Cameras tagged `*Entry*` → event_type = `entry`
- Cameras tagged `*Exit*` → event_type = `exit`
- Cameras tagged `*Both*` or untagged → use time-of-day heuristic (first sighting of a shift = entry, last = exit)

**Strategy B — Line crossing (future enhancement)**  
Draw a virtual line on the camera frame. Track face positions across consecutive frames; crossing direction determines entry vs exit. Higher complexity — implement after Strategy A is validated.

### 8.2 Cooldown Gate (Anti-Duplicate)

```
In-memory dict: {(worker_id, camera_id) → last_log_timestamp}

Before logging an attendance event:
  if (worker_id, camera_id) in cooldown_dict:
    if now - last_log_timestamp < MINE_ATTENDANCE_COOLDOWN_SECONDS (default: 300):
      skip — already logged within the cooldown window
  else:
    log event, update cooldown_dict
```

Cooldown is per (worker × camera) to allow logging on different cameras independently (e.g., entry gate + shaft entry).

### 8.3 Shift Assignment

When logging an attendance event, assign `shift_id` by querying:
```sql
SELECT id FROM mine_shifts
WHERE shift_date = CURRENT_DATE
  AND start_time <= NOW()
  AND end_time > NOW()
  AND is_closed = FALSE
LIMIT 1;
```

Night shift crosses midnight — handled by checking `shift_date = CURRENT_DATE - 1` when `NOW()` is between midnight and 06:00.

### 8.4 On-Site Presence Calculation (`/mine/attendance/live`)

A worker is considered **on-site** if:
- Their most recent `mine_attendance_log` within the last 8 hours has `event_type = 'entry'`, AND
- They have no subsequent `exit` event.

```sql
SELECT DISTINCT ON (worker_id) worker_id, event_type, detected_at
FROM mine_attendance_logs
WHERE detected_at > NOW() - INTERVAL '8 hours'
ORDER BY worker_id, detected_at DESC;
-- Filter in Python: keep rows where event_type = 'entry'
```

### 8.5 Absence Detection

APScheduler job runs at end of each shift:
1. Find all enrolled, active workers.
2. Find workers with at least one attendance log in the shift period.
3. Mark absent = (all_workers - attended_workers).
4. Optionally send SMS/email alert for absent workers (hook into SMTP config already in `Settings`).

---

## 9. Configuration (New Settings)

Add to `app/config.py` `Settings` class:

```python
# Mine Attendance
MINE_FACE_MATCH_THRESHOLD: float = 0.40
MINE_FACE_MIN_SIZE_PX: int = 60           # larger than retail (helmets)
MINE_FRAME_SAMPLE_INTERVAL_SECONDS: float = 2.0
MINE_ATTENDANCE_COOLDOWN_SECONDS: int = 300   # 5 minutes per (worker, camera)
MINE_RECONNECT_DELAY_SECONDS: float = 5.0
MINE_MAX_FACE_EMBEDDINGS_PER_WORKER: int = 5
MINE_BUCKET_PREFIX: str = "mine"          # MinIO sub-bucket prefix
```

---

## 10. Infrastructure Requirements

### 10.1 Current EC2 Instance

The existing RetailEye workload already runs on EC2 (13.234.173.13). The mine module adds:
- 14 RTSP connections over WireGuard (persistent TCP sockets)
- InsightFace inference on 7 frames/sec (0.5fps × 14 cameras)
- Minimal PostgreSQL write load (attendance events are infrequent)

### 10.2 Recommended EC2 Sizing

| Use Case | Instance | vCPU | RAM | GPU | Cost/mo (approx) |
|----------|----------|------|-----|-----|-----------------|
| CPU only (acceptable) | `t3.xlarge` | 4 | 16GB | — | ~$120 |
| GPU (recommended) | `g4dn.xlarge` | 4 | 16GB | T4 16GB | ~$380 |
| GPU (if also running retail AI) | `g4dn.2xlarge` | 8 | 32GB | T4 16GB | ~$560 |

**Recommendation:** `g4dn.xlarge`  
InsightFace buffalo_l on T4: ~15ms/face. At 7 faces/sec peak: 10% GPU utilization. The existing YOLO + OSNet pipeline already benefits from the T4. Combining both workloads on `g4dn.xlarge` is cost-efficient.

**CPU-only fallback:** InsightFace buffalo_l on 4 vCPU: ~200-300ms/face. At 0.5fps × 14 cameras = 7 inference calls/sec → CPU would be saturated. Acceptable only if frame sampling is dropped to 0.2fps (1 frame every 5 seconds) or if cameras are processed in sequence rather than parallel.

### 10.3 WireGuard VPN Bandwidth

| Traffic | Bandwidth |
|---------|-----------|
| 14 cameras × 1080p H.265 full stream | ~14 × 2-4 Mbps = 28-56 Mbps |
| **Recommended: sub-stream only** | 14 × 360p H.265 ≈ 14 × 0.5 Mbps = **7 Mbps** |

Use `subtype=1` in RTSP URL for the sub-stream (360p) for face sampling. Sub-stream is sufficient for face recognition at a gate camera (faces are close and large). Only the recording path (for evidence) needs the main stream.

Sub-stream RTSP: `rtsp://admin:Cctv%40123@192.168.31.15:554/video/live?channel=1&subtype=1`

### 10.4 Storage

| Data | Estimate | Storage |
|------|----------|---------|
| Face crops (enrollment) | 14 workers × 5 crops × 200KB ≈ 14MB | MinIO |
| Attendance face crops (evidence) | 500 events/day × 200KB × 365 days ≈ 36GB/year | MinIO (lifecycle policy: delete after 1 year) |
| PostgreSQL tables | Negligible (mostly UUID + timestamps) | RDS or local Postgres |
| Video recordings (full-res) | 14 cams × 3h chunks × 24h × 2GB/chunk ≈ 2TB/day | `/mnt/hdd1` (already configured) |

---

## 11. Tech Stack Choices with Reasoning

| Choice | Alternative Considered | Reasoning |
|--------|----------------------|-----------|
| **InsightFace buffalo_l** for face recognition | DeepFace, FaceNet | Already in codebase. 512-dim ArcFace embeddings match existing pgvector schema. No additional dependency. |
| **pgvector cosine search** for gallery lookup | FAISS, in-memory numpy | Already in use for body ReID. No new infra. Scales to thousands of workers. Transactional — embeddings added/removed atomically with worker records. |
| **FFmpeg subprocess** for H.265 frame extraction | OpenCV VideoCapture | CP Plus H.265 decode is unreliable with OpenCV-headless. FFmpeg natively handles H.265. Same pattern already used in `CameraRecorder`. |
| **asyncio tasks** (one per camera) | Celery, threading | Existing pattern in codebase (`WorkerSupervisor`, `RecordingSupervisor`). No new infra (no Redis, no Celery broker). I/O-bound work (RTSP read + DB write) suits asyncio. |
| **APScheduler** for shift jobs | Celery Beat, cron | Already in `app/worker.py`. Shift creation and absence detection are simple periodic tasks. |
| **Existing `cameras` table** + `site_type` flag | Separate `mine_cameras` table | Reuses all camera management APIs and UI. Mine cameras appear in the same dashboard alongside retail cameras. |
| **MinIO** for face crops | Local filesystem | Already running. Face crops are small objects — MinIO is suitable. Consistent with how retail crops are stored. |
| **Sub-stream (360p)** for face sampling | Full-stream (1080p) | 7× bandwidth reduction over WireGuard VPN. Faces at gate cameras are close enough that 360p is sufficient for buffalo_l at 640px det size. |

---

## 12. Implementation Phases

### Phase 1 — Foundation (Week 1-2)
- Alembic migration: `mine_workers`, `mine_worker_face_embeddings`, `mine_shifts`, `mine_attendance_logs`, cameras `site_type` column
- Worker CRUD API (`/mine/workers/`, enrollment endpoint)
- Face enrollment pipeline (upload images → InsightFace → store embeddings)
- Unit tests for enrollment

### Phase 2 — Detection Pipeline (Week 2-3)
- `MineFrameWorker` (FFmpeg pipe + InsightFace + pgvector match)
- `MineAttendanceSupervisor` (asyncio singleton, lifecycle hooks)
- Cooldown gate + shift assignment logic
- Camera start/stop API (`/mine/cameras/{id}/start|stop`)
- Integration test with 1 real camera over VPN

### Phase 3 — Attendance API (Week 3-4)
- Attendance log query API (filters, pagination)
- Live on-site view (`/mine/attendance/live`)
- Shift management (auto-create via APScheduler)
- Daily report endpoint + CSV export
- Absence detection job

### Phase 4 — Dashboard Integration (Week 4-5)
- Frontend: Worker registration form + face upload
- Frontend: Attendance table + report view (existing RetailEye dashboard)
- Alert: SMTP notification for absent workers at shift end
- Load test: 14 cameras simultaneously for 4+ hours

---

## 13. Open Questions for Manager Sign-Off

1. **Entry vs Exit per camera** — Which of the 14 cameras are at entry gates vs exit gates? This determines `mine_location` tags and `event_type` assignment.
2. **Enrollment process** — Who conducts face enrollment? Will workers be enrolled on-site (HR), or remotely by uploading photos? Are helmet photos available?
3. **Attendance report format** — What format does management need? (CSV, PDF, Excel, direct dashboard view?)
4. **Shift schedule** — Confirm the three standard shifts (Morning/Afternoon/Night) and exact start/end times for the mine site.
5. **Absence alerting** — Should absent workers trigger SMS or only email? (SMS requires additional Twilio/AWS SNS integration.)
6. **Data retention** — How long should attendance logs and face crops be retained? (Suggested: 2 years for logs, 1 year for crops.)
7. **GPU budget** — `g4dn.xlarge` (~$380/mo) vs CPU-only `t3.xlarge` (~$120/mo, reduced accuracy under load). Which is approved?
8. **Separate deployment or same instance?** — If mine attendance and retail AI both run on the same EC2, `g4dn.xlarge` handles both. If separate security zones are required, a dedicated instance is needed.

---

## 14. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| WireGuard VPN drops | Medium | High | Auto-reconnect in `MineFrameWorker` (5s retry loop); watchdog pings VPN endpoint |
| Helmet occlusion → low recognition accuracy | High | High | Enroll with helmet on; consider lower face in frame; accept ~85% accuracy target |
| H.265 decode failure | Low | Medium | FFmpeg subprocess with error logging; fallback to sub-stream |
| Same-person multiple enrollments | Medium | Medium | Enrollment dedup: compare new embedding against existing gallery before inserting (reject if similarity > 0.70) |
| pgvector gallery scan slowness as workers grow | Low | Low | HNSW index on embedding column (already used in existing schema) |
| Night shift crossing midnight | Low | Medium | Explicit date logic in shift assignment query (covered in §8.3) |
