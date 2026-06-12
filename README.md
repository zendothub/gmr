# Retail AI Platform

Production-ready **single modular FastAPI application** for an on-prem AI CCTV retail analytics system (pharmacy). All capabilities live as internal modules of one application with clean boundaries — no microservices — so any module can be split out later if needed.

## Stack

| Concern | Technology |
| --- | --- |
| API | FastAPI (async), Pydantic v2 |
| Database | PostgreSQL 16 + `uuid-ossp` + `pgvector` |
| ORM / Migrations | SQLAlchemy 2.0 (async) / Alembic |
| Camera I/O | OpenCV (RTSP test + runtime frame reading) |
| Detection | Ultralytics YOLO (person by default) |
| Tracking | ByteTrack adapter (stub tracker included; TODO real weights) |
| ReID | torchreid OSNet 512-dim embeddings stored in `VECTOR(512)` (pgvector cosine search) |
| Media storage | Local filesystem (snapshots / crops / clips / reports) |
| Background jobs | APScheduler (in-process) |
| Auth | JWT (python-jose) + bcrypt |

## Architecture

```
retail-ai-platform/
├── app/
│   ├── main.py               # FastAPI app, router registration
│   ├── config.py             # Pydantic settings (.env driven)
│   ├── dependencies.py       # DI: DB session, current user, superuser
│   ├── lifecycle.py          # startup/shutdown (storage dirs, scheduler, workers)
│   ├── core/db/              # Base, async/sync sessions, SQLAlchemy models
│   ├── modules/
│   │   ├── auth/             # signup / login / JWT
│   │   ├── users/            # user & role admin
│   │   ├── cameras/          # camera CRUD + RTSP test + start/stop/health
│   │   ├── camera_views/     # ROI polygons (full_frame, entry_gate_view, ...)
│   │   ├── zones/            # polygon/line zones (entry_line, billing_zone, ...)
│   │   ├── rules/            # rule CRUD (line_crossing, zone_dwell, ...)
│   │   ├── ai_runtime/       # worker_supervisor, camera_worker, frame_buffer
│   │   ├── detection/        # yolo_detector.py (DetectionResult)
│   │   ├── tracking/         # bytetrack_adapter.py, track_manager.py
│   │   ├── reid/             # crop_quality, osnet_extractor, identity_decision_engine
│   │   ├── rule_engine/      # rule_evaluator, camera_view_engine, config_loader
│   │   ├── events/           # event listing / ack / false-positive
│   │   ├── billing/          # billing_interactions
│   │   ├── analytics/        # footfall / billing / dwell / occupancy / journey
│   │   ├── storage/          # local filesystem + storage_objects registry
│   │   └── jobs/             # APScheduler: daily analytics, cleanup
│   └── utils/                # geometry, image, time, encryption, pagination
├── alembic/                  # migrations (initial schema + pgvector index)
├── database/init.sql         # extensions (uuid-ossp, vector)
├── deployment/               # systemd unit, nginx example
├── tests/
├── docker-compose.yml        # PostgreSQL (pgvector) + app
├── Dockerfile
├── requirements.txt
└── .env.example
```

## AI Pipeline (per camera worker)

```
RTSP stream ──> LatestFrameBuffer (capture thread, keeps newest frame only)
   │  sampled at camera.fps_target (never the native 25 FPS)
   ▼
YOLO detection (person class) ──> camera-view ROI filter (center point in polygon)
   ▼
ByteTrack tracking ──> TrackManager (in-memory state + track_sessions in PostgreSQL)
   ▼
Zone update (point-in-polygon, dwell seconds per zone)
   ▼
ReID (gated):  track age > 1.5s, bbox height > 120px,
               stability > 0.65, last ReID > 3s ago
   crop ─> quality (reject < 0.70) ─> OSNet 512-dim embedding
        ─> pgvector cosine search ─> identity decision:
   final_score = 0.60*visual + 0.15*crop_quality + 0.10*time
               + 0.10*camera_transition + 0.05*stability
   score >= 0.78 → match existing person, else new anonymous identity
   ▼
Rule engine (in-memory cache, cooldown_seconds dedup) ──> events + billing_interactions
```

**No per-frame DB queries** for configuration: rules/zones/views are cached in memory and refreshed only via `POST /api/runtime/reload-config`.

## Quick Start (Docker)

```bash
cd retail-ai-platform
cp .env.example .env          # adjust SECRET_KEY etc.
docker compose up --build
```

- API: http://localhost:8000 — Swagger: http://localhost:8000/docs
- PostgreSQL (pgvector/pgvector:pg16) is initialized with `uuid-ossp` and `vector` extensions; Alembic migrations run automatically on app start.

## Quick Start (local, without Docker)

```bash
# 1. PostgreSQL with pgvector
docker run -d --name retail_pg -p 5432:5432 \
  -e POSTGRES_USER=retail_user -e POSTGRES_PASSWORD=retail_pass \
  -e POSTGRES_DB=retail_ai_db pgvector/pgvector:pg16
psql postgresql://retail_user:retail_pass@localhost:5432/retail_ai_db \
  -c 'CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; CREATE EXTENSION IF NOT EXISTS vector;'

# 2. Python env
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env   # set STORAGE_ROOT to a writable local path, e.g. ./storage

# 4. Migrate + run
alembic upgrade head
uvicorn app.main:app --reload
```

### Model weights (TODO stubs included)

Place real weights in `models/` (configured via `.env`):

- `models/yolov8n.pt` — auto-downloaded by ultralytics on first run, or copy manually.
- `models/osnet_x1_0.pth` — torchreid OSNet weights. Until provided, the extractor runs a deterministic stub (marked `TODO` in `osnet_extractor.py`).
- ByteTrack: the adapter ships with a simple IoU tracker stub; plug in the real ByteTrack implementation in `bytetrack_adapter.py` (marked `TODO`).

## Typical Workflow

1. `POST /api/auth/signup` → `POST /api/auth/login` → use `Bearer` token.
2. `POST /api/cameras/test-rtsp` — verify the stream with OpenCV before saving.
3. `POST /api/cameras` — save camera (role: `entry_gate` / `billing_counter` / `queue` / `product_shelf`, plus `fps_target`, `detection_model`, `reid_enabled`, `demographic_enabled`, `resolution`).
4. `POST /api/cameras/{id}/views` — draw ROI polygon (`full_frame`, `entry_gate_view`, `billing_counter_view`, `queue_view`, `product_shelf_view`, `ignore_area`).
5. `POST /api/camera-views/{view_id}/zones` — create zones (`entry_line`, `exit_line`, `billing_zone`, `queue_zone`, `product_zone`, `ignore_zone`, `restricted_zone`) as polygon or line JSON.
6. `POST /api/rules` — configure rules (`line_crossing`, `zone_dwell`, `billing_interaction`, `queue_count`, `possible_purchase`, `restricted_zone`) with `cooldown_seconds`.
7. `POST /api/cameras/{id}/start` (or `POST /api/runtime/start` for all active cameras).
8. After editing config: `POST /api/runtime/reload-config`.
9. Monitor: `GET /api/runtime/status`, `GET /api/cameras/{id}/health`, `GET /api/events`, `GET /api/analytics/*`.

## API Surface

| Area | Endpoints |
| --- | --- |
| Auth | `POST /api/auth/signup`, `POST /api/auth/login`, `GET /api/auth/me` |
| Users | `POST/GET /api/users`, `GET/PUT/DELETE /api/users/{id}`, roles |
| Cameras | `POST /api/cameras/test-rtsp`, CRUD `/api/cameras`, `/{id}/start`, `/{id}/stop`, `/{id}/health` |
| Views | `POST/GET /api/cameras/{id}/views`, `GET/PUT/DELETE /api/camera-views/{id}`, `/{id}/set-default` |
| Zones | `POST/GET /api/camera-views/{id}/zones`, `GET/PUT/DELETE /api/zones/{id}` |
| Rules | CRUD `/api/rules`, `/{id}/enable`, `/{id}/disable` |
| Runtime | `POST /api/runtime/reload-config`, `/start`, `/stop`, `GET /status` |
| Events | `GET /api/events`, `GET /{id}`, `POST /{id}/acknowledge`, `POST /{id}/false-positive` |
| Billing | `GET /api/billing/interactions` |
| Analytics | `GET /api/analytics/footfall`, `/billing`, `/dwell`, `/zone-occupancy`, `/person-journey/{person_id}` |
| Storage | `GET /api/storage/objects`, `/{id}/download` |

## Database Tables

`users`, `roles`, `stores`, `cameras`, `camera_views`, `zones`, `rules`,
`track_sessions`, `track_observations`, `person_identities`,
`person_embeddings` (`VECTOR(512)` + IVFFlat cosine index), `events`,
`billing_interactions`, `daily_analytics_summary`, `storage_objects`.

## Background Jobs

- `daily_analytics_aggregation` — 00:15 daily, fills `daily_analytics_summary`.
- `close_stale_track_sessions` — every 5 min, closes orphaned sessions.
- `storage_cleanup` — 02:00 daily, removes media older than 30 days.

## Testing

```bash
pytest tests/ -v
```

## Production Notes

- Run a **single uvicorn worker** (`--workers 1`): camera workers and rule caches are in-process state.
- Set a strong `SECRET_KEY`, restrict CORS origins in `app/main.py`.
- Bare-metal deployment: see `deployment/retail-ai.service` (systemd) and `deployment/nginx.conf`.
- GPU: install CUDA-enabled torch and ultralytics will use it automatically.