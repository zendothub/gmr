# Retail Eye Insights — Backend

On-premises AI CCTV retail analytics for pharmacies.
FastAPI (Python 3.10) + PostgreSQL/pgvector + MinIO + RTX 4070 Ti GPU.

For architecture, pipeline, identity matching, and detailed documentation,
see **[docs/README.md](docs/README.md)**.

---

## Requirements

- Python 3.10+
- PostgreSQL 14+ with pgvector extension
- MinIO (or any S3-compatible object storage)
- NVIDIA GPU (RTX 4070 Ti or better) with CUDA 12+
- ffmpeg with NVENC support (for stream broadcasting)
- MediaMTX (for WebRTC/HLS streaming)

## Install

```bash
# Clone
git clone <YOUR_REPO_URL> retail-ai-platform
cd retail-ai-platform

# Create virtualenv
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8n weights (auto-downloads on first run, or manually)
wget -O models/yolov8n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

# OSNet body ReID weights (MSMT17 checkpoint)
# Place at models/osnet_x1_0.pth — see docs/REID_PRODUCTION_INTEGRATION.md

# Configure
cp .env.example .env
# Edit .env: DATABASE_URL, MINIO settings, SECRET_KEY, CORS_ORIGINS
```

## Database Setup

```bash
# Create database + extensions
createdb retail_ai_db
psql retail_ai_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql retail_ai_db -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"

# Run migrations
alembic upgrade head

# Build pgvector indexes (also done automatically by migration)
psql retail_ai_db -c "
  CREATE INDEX IF NOT EXISTS idx_person_embeddings_embedding
  ON person_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
  CREATE INDEX IF NOT EXISTS idx_person_face_embeddings_embedding
  ON person_face_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
"
```

## Run (Development)

```bash
# Start the API server (single worker — camera workers share in-process state)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# In a separate terminal, start the background worker
python -m app.worker

# Verify
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

## Production Deployment

### Three-Process Architecture

The backend runs as **three separate systemd services**:

```
┌─────────────────────────────┐  ┌──────────────────────────────┐  ┌──────────────────────────────┐
│ retail-ai.service           │  │ retail-ai-worker.service     │  │ reextract-faces.timer        │
│ (API Server, ~3.4 GB GPU)   │  │ (Background Jobs, ~100 MB)   │  │ (GPU Worker, oneshot)        │
│                             │  │                              │  │                              │
│ - FastAPI HTTP endpoints    │  │ - deduplicate_persons (10m)  │  │ - Re-extract face from crops │
│ - Camera workers (YOLO,     │  │ - close_stale_tracks (5m)    │  │ - Delete faceless persons    │
│   InsightFace, OSNet, etc.) │  │ - probe_cameras (2m)         │  │ - Every 20 min               │
│ - Stream broadcasters       │  │ - daily_analytics (00:15)    │  │ - Loads InsightFace (1.5GB) │
│ - In-memory track state     │  │ - storage_cleanup (02:00)    │  │ - Frees GPU on exit          │
│ - NO background jobs        │  │ - NO GPU, NO camera          │  │                              │
└─────────────────────────────┘  └──────────────────────────────┘  └──────────────────────────────┘
```

All three share PostgreSQL + MinIO. The worker handles heavy DB/MinIO so the
API server never freezes. The face re-extraction loads InsightFace as a
separate process to avoid GPU memory contention.

**Critical:** `--workers 1` is MANDATORY for the API server — camera workers
share in-process ByteTrack state that breaks with multiple workers.

### Install systemd Services

```bash
# Copy service files
sudo cp systemd/retail-ai.service /etc/systemd/system/
sudo cp systemd/retail-ai-worker.service /etc/systemd/system/
sudo cp systemd/reextract-faces.service /etc/systemd/system/
sudo cp systemd/reextract-faces.timer /etc/systemd/system/

# Reload and enable
sudo systemctl daemon-reload
sudo systemctl enable retail-ai.service retail-ai-worker.service reextract-faces.timer

# Start (order matters: API first, then worker)
sudo systemctl start retail-ai.service
sudo systemctl start retail-ai-worker.service
sudo systemctl start reextract-faces.timer
```

All three are `enabled` by default and start automatically on boot.

### Restart

```bash
sudo systemctl restart retail-ai.service          # API server
sudo systemctl restart retail-ai-worker.service    # background worker
# No need to restart reextract-faces.timer unless the script changed
```

Edit `WorkingDirectory` in the service files if your install path is not `/gmr/gmr`.

## EC2 / Cloud Deployment

### Instance Sizing

| Type | Recommendation |
|---|---|
| API + cameras on-prem | GPU box (RTX 4070 Ti+) with local Postgres + MinIO |
| API-only on cloud | `t3.large` minimum (2 vCPU / 8 GB) — no camera workers, no GPU |
| Full stack on cloud | `g4dn.xlarge`+ (GPU instance) |

### Security Group (inbound)

| Port | Source | Purpose |
|---|---|---|
| 22 | Your IP only | SSH |
| 80 | 0.0.0.0/0 | HTTP (certbot redirect) |
| 443 | 0.0.0.0/0 | HTTPS |

Do NOT open 8000 or 5432 — nginx proxies locally, Postgres stays internal.

### Deploy Steps

```bash
ssh -i your-key.pem ubuntu@<ELASTIC_IP>

# System packages
sudo apt update && sudo apt upgrade -y
sudo apt install -y git nginx certbot python3-certbot-nginx
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu && newgrp docker

# Get code
cd /opt
sudo mkdir retail-ai-platform && sudo chown ubuntu:ubuntu retail-ai-platform
git clone <YOUR_REPO_URL> retail-ai-platform
cd retail-ai-platform

# Configure
cp deployment/.env.production .env
# Edit .env: SECRET_KEY, POSTGRES_PASSWORD, CORS_ORIGINS

# Start with Docker Compose
docker compose -f docker-compose.prod.yml up -d --build
curl http://127.0.0.1:8000/health   # → {"status":"ok"}

# Nginx + HTTPS
sudo cp deployment/nginx-api.gmr.zendot.in.conf /etc/nginx/sites-available/api.gmr.zendot.in
sudo ln -s /etc/nginx/sites-available/api.gmr.zendot.in /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d api.gmr.zendot.in --redirect -m you@zendot.in --agree-tos -n

# Verify from outside
curl https://api.gmr.zendot.in/health   # → {"status":"ok"}
```

### DNS

```
A    api.gmr    ->  <ELASTIC_IP>    TTL 300
```

Allocate an Elastic IP so the IP survives instance stop/start.

### Operations

```bash
cd /opt/retail-ai-platform

docker compose -f docker-compose.prod.yml logs -f app       # logs
docker compose -f docker-compose.prod.yml restart app       # restart API
docker compose -f docker-compose.prod.yml up -d --build      # deploy new code
docker compose -f docker-compose.prod.yml down               # stop (data persists)

# DB backup (cron daily)
docker exec retail_ai_postgres pg_dump -U retail_user retail_ai_db \
  | gzip > /opt/backups/db_$(date +%F).sql.gz
```

### Production Checklist

- [ ] `SECRET_KEY` and `POSTGRES_PASSWORD` changed from defaults
- [ ] Ports 8000/5432 NOT in the security group
- [ ] SSH restricted to your IP
- [ ] HTTPS works and HTTP redirects
- [ ] Elastic IP attached
- [ ] DB backups scheduled
- [ ] RTSP cameras reachable from the instance (site-to-site VPN if on-prem)

## Logs

| Service | Log location |
|---|---|
| API server | `logs/ai_processing.log` |
| Background worker | `logs/ai_processing.log` (shared setup) + `journalctl -u retail-ai-worker.service` |
| Face re-extraction | `journalctl -u reextract-faces.service` |

## Configuration

All thresholds and settings are in `app/config.py` (Pydantic Settings).
Environment overrides via `.env` file in the project root.

## Danger Scripts

Diagnostic and fix scripts live in `danger/`. Key scripts:

| Script | Purpose |
|---|---|
| `reset_tracking_data.py` | Full data reset (preserves config) + rebuilds pgvector indexes |
| `normalize_face_embeddings.py` | L2-normalize all face embeddings + rebuild index |
| `diagnose_persons.py` | Deep-dive diagnostic for specific person IDs |
| `reextract_or_delete_faceless.py` | Re-extract faces or delete faceless persons (systemd timer) |
| `clean_contaminated_embeddings.py` | Median-based face+body contamination cleanup |
| `merge_recent_window_duplicates.py` | Historical backfill — merge same-visit duplicates |
| `measure_body_reid.py` | Measure OSNet same/diff body similarity distributions |
| `recompute_body_embeddings.py` | Recompute all body embeddings from MinIO crops |
