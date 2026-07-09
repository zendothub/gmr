# Retail Eye Insights — Backend

On-premises AI CCTV retail analytics for pharmacies.
FastAPI (Python 3.10) + PostgreSQL/pgvector + MinIO + RTX 4070 Ti GPU.

## Architecture — Three Processes

The backend runs as **three separate systemd services**, each with its own process:

### 1. `retail-ai.service` — API Server (Main)

```
Start:    systemctl start retail-ai.service
Command:  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
Memory:   ~3.4 GB (GPU models: InsightFace, OSNet, SigLIP2, MiVOLO)
```

The main FastAPI HTTP server. Handles:
- REST API endpoints (`/api/v1/...`)
- Camera workers (YOLO + ByteTrack, per-camera)
- AI pipeline (face detection, gender, age, body ReID, identity matching)
- Stream broadcasters (FFmpeg NVENC → MediaMTX → WebRTC)
- In-memory track state (ByteTrack, face accumulation, gender voting)
- GPU model inference

**Critical:** Must run with `--workers 1` — camera workers share in-process state.
**No background jobs run here** — they are offloaded to the worker process to prevent
API freezes during heavy DB/MinIO operations.

### 2. `retail-ai-worker.service` — Background Job Worker

```
Start:    systemctl start retail-ai-worker.service
Command:  python -m app.worker
Memory:   ~100 MB (no GPU models, no camera workers)
```

Standalone process running APScheduler with all periodic background jobs.
Does NOT load any GPU models or camera workers — pure DB + MinIO operations.

**Scheduled jobs:**

| Job | Interval | Purpose |
|---|---|---|
| `deduplicate_persons` | every 10 min | Merge duplicate persons, absorb face/body embeddings, re-vote gender, clean contaminated embeddings, sweep MinIO, classify staff |
| `close_stale_track_sessions` | every 5 min | Close track sessions that stopped receiving updates |
| `probe_camera_statuses` | every 2 min | Probe RTSP streams and update camera status (ACTIVE/INACTIVE) |
| `aggregate_daily_analytics` | daily 00:15 | Aggregate yesterday's metrics into daily summary |
| `cleanup_old_storage` | daily 02:00 | Delete snapshots/crops older than retention period |

**Why separate?** The dedup job performs heavy DB queries (pgvector LATERAL with
`probes=50`), numpy computations (pairwise similarity matrices for 1k+ persons),
and MinIO batch deletions (15k+ objects). Running these in the API server's event
loop caused 1-2 minute freezes on all HTTP requests.

### 3. `reextract-faces.timer` — Face Re-extraction (GPU Worker)

```
Start:    systemctl start reextract-faces.timer
Command:  python danger/reextract_or_delete_faceless.py
Memory:   ~1.5 GB (loads InsightFace buffalo_l temporarily, freed on exit)
Runs:     every 20 min via systemd timer (OnUnitActiveSec=1200)
```

Standalone oneshot process that runs periodically via systemd timer.
Loads InsightFace to re-extract face embeddings from track crops for persons
left faceless by contamination cleanup. If no face is found in any crop, deletes
the person entirely (tracks are orphaned — they'll be re-identified on next visit).

**Why separate from the worker?** This job loads InsightFace (1.5 GB GPU memory).
Running it inside the worker would contend with the API server's GPU models.
As a oneshot process, it allocates GPU memory only during execution and frees it
on exit.

## Startup Order

```bash
# 1. Start the API server (loads GPU models, starts camera workers)
sudo systemctl start retail-ai.service

# 2. Start the background worker (DB + MinIO jobs, no GPU)
sudo systemctl start retail-ai-worker.service

# 3. The face re-extraction timer auto-starts on boot (OnBootSec=120)
sudo systemctl start reextract-faces.timer
```

All three are `enabled` by default and start automatically on boot.

## Logs

| Service | Log location |
|---|---|
| API server | `/gmr/gmr/logs/ai_processing.log` + `/gmr/gmr/logs/retail-ai.log` |
| Background worker | `journalctl -u retail-ai-worker.service` |
| Face re-extraction | `journalctl -u reextract-faces.service` |

## Configuration

All thresholds and settings are in `app/config.py` (Pydantic Settings).
Environment overrides via `.env` file in the project root.

See `/gmr/CONTEXT.md` for the complete architecture, model choices, threshold
rationales, and known issues.

## Danger Scripts

Diagnostic and fix scripts live in `danger/`. See `/gmr/CONTEXT.md` for the full
list. Key scripts:

| Script | Purpose |
|---|---|
| `reset_tracking_data.py` | Full data reset (preserves config) + rebuilds pgvector indexes |
| `normalize_face_embeddings.py` | One-time migration: L2-normalize all face embeddings + rebuild index |
| `diagnose_persons.py` | Deep-dive diagnostic for specific person IDs |
| `reextract_or_delete_faceless.py` | Re-extract faces or delete faceless persons (runs via systemd timer) |
