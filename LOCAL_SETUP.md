# Local Setup — Fresh New Client Database

This guide walks you through spinning up a **brand-new, clean database** for a new client deployment. All tables (including employees and attendance) are created automatically via Alembic migrations.

---

## Prerequisites

| Tool | Minimum Version |
|------|-----------------|
| Docker + Docker Compose | v2.20+ |
| Python | 3.12+ |
| pip | 23+ |

---

## Step 1 — Clone the Repository

```bash
git clone git@github.com:zendothub/gmr.git retail-ai-platform
cd retail-ai-platform
```

---

## Step 2 — Configure Environment

```bash
cp .env.example .env
```

Open `.env` and set **at minimum** these values:

```dotenv
# Required: strong random secret key
SECRET_KEY=your-super-secret-key-change-this

# Database — points to the Dockerised Postgres on port 5433
DATABASE_URL=postgresql+asyncpg://retail_user:retail_pass@localhost:5433/retail_ai_db
DATABASE_SYNC_URL=postgresql://retail_user:retail_pass@localhost:5433/retail_ai_db

# MinIO — local Docker instance
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false
MINIO_BUCKET_PREFIX=retail
```

> All other values have sensible defaults in `.env.example`.

---

## Step 3 — Start Infrastructure (Postgres + MinIO + MediaMTX)

```bash
docker compose up -d
```

This starts:
- **PostgreSQL 16** with pgvector at `localhost:5433`
- **MinIO** object storage at `localhost:9000` (console: `http://localhost:9001`)
- **MediaMTX** media server at `localhost:8889` (WebRTC) / `localhost:8888` (HLS)

Wait for Postgres to be healthy:

```bash
docker compose ps        # STATUS should show "(healthy)" for retail_ai_postgres
```

---

## Step 4 — Install Python Dependencies

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Step 5 — Run All Migrations (Creates Every Table)

```bash
alembic upgrade head
```

This runs migrations **0001 → 0009** in sequence, creating all tables:

| Migration | Tables Created |
|-----------|----------------|
| 0001 | stores, cameras, zones, users, person_identities, tracks, … (core schema) |
| 0002 | person_debug |
| 0003 | person_debug camera nullable |
| 0004 | body_crop_path column |
| 0005 | face_embedding index |
| 0006 | device_sessions, stream_viewer_sessions |
| 0007 | identity_merge_events, fragmented_track_events |
| 0008 | audit_job_run columns |
| **0009** | **shift_slots, employees, attendance_records** (employee attendance) |

Verify:

```bash
alembic current     # Should show: 0009 (head)
```

---

## Step 6 — Seed Default Admin User

```bash
python -m app.seed
```

This creates a default `super_admin` account. Check the seed output for credentials (or see `app/seed.py`).

---

## Step 7 — Start the Application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> ⚠️ **`--workers 1` is mandatory.** Camera workers share in-process state (ByteTrack instances, frame buffers) that breaks with multiple workers.

The API is now live at:
- **REST API**: `http://localhost:8000`
- **Interactive API Docs**: `http://localhost:8000/docs`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

---

## Step 8 — Export `openapi.json` for Frontend

```bash
curl -s http://localhost:8000/openapi.json -o openapi.json
```

Copy this file to the frontend repo (`Retail-Eye-Insights/openapi.json`) and regenerate the API hooks:

```bash
cd /path/to/Retail-Eye-Insights
bun run api:gen     # Regenerates src/api/ from openapi.json
```

---

## All-in-One Quick Start (Summary)

```bash
git clone git@github.com:zendothub/gmr.git retail-ai-platform && cd retail-ai-platform
cp .env.example .env          # Edit DATABASE_URL, MINIO_*, SECRET_KEY
docker compose up -d
pip install -r requirements.txt
alembic upgrade head          # Creates ALL tables including shift_slots, employees, attendance_records
python -m app.seed            # Creates default admin user
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## Resetting the Database (Start Completely Fresh)

If you need to wipe everything and start over:

```bash
# Stop containers and remove ALL data volumes
docker compose down -v

# Restart fresh infrastructure
docker compose up -d

# Re-run migrations
alembic upgrade head

# Re-seed admin
python -m app.seed
```

---

## New Employee Attendance Tables (Migration 0009)

Three new tables are created by migration `0009`:

### `shift_slots`
Defines work shifts (e.g., Morning 6AM–2PM, Evening 2PM–10PM).

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| label | VARCHAR(50) | Unique shift name |
| start_time | TIME | Shift start |
| end_time | TIME | Shift end |
| crosses_midnight | BOOLEAN | True if end_time < start_time |
| is_active | BOOLEAN | Soft toggle |

### `employees`
Links human employees to their AI-detected `person_identity`.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| emp_id | VARCHAR(100) | Unique employee ID (e.g., "EMP001") |
| name | VARCHAR(255) | Full name |
| person_identity_id | UUID FK | Links to AI identity (SET NULL on delete) |
| shift_slot_id | UUID FK | Assigned shift |
| face_crop_path | VARCHAR | MinIO path to face crop used for registration |
| is_active | BOOLEAN | Soft delete |

### `attendance_records`
Daily attendance records upserted by the camera worker on each detection.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| employee_id | UUID FK | Links to employees |
| attendance_date | DATE | Calendar date |
| shift_slot_id | UUID FK | Shift on this date |
| first_seen_at | TIMESTAMPTZ | First camera detection (immutable after first write) |
| last_seen_at | TIMESTAMPTZ | Last camera detection (updated on each detection) |
| total_hours | FLOAT | Computed from first→last seen |
| status | ENUM | present / absent / late / half_day |
| check_in_camera_id | UUID FK | Camera where first seen |
| check_out_camera_id | UUID FK | Camera where last seen |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `alembic upgrade head` fails with "relation does not exist" | Postgres not healthy yet — wait 10s and retry |
| `pgvector extension not found` | The `pgvector/pgvector:pg16` image includes it; ensure `./database/init.sql` runs on first start |
| Port 5433 already in use | Change `"5433:5432"` to `"5434:5432"` in docker-compose.yml and update `DATABASE_URL` |
| MinIO connection refused | Run `docker compose ps` — minio container must be healthy first |
| Camera worker crashes with multiple workers | Always use `--workers 1` |
