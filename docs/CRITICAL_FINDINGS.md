# Critical Findings & Troubleshooting

> **Created:** July 6, 2026
> **Purpose:** Documents critical bugs discovered and fixed during the ReID/tracking overhaul. Read this BEFORE modifying camera worker, inference pool, or YOLO detector code.

---

## 1. ByteTrack State Corruption (Shared YOLO Model)

**Status:** FIXED (July 6, 2026)

### Problem
ByteTrack (`persist=True`) maintains internal state: Kalman filters, track ID assignments, velocity estimates. This state is **NOT thread-safe**. When multiple camera workers share a single YOLO model instance and call `model.track()` concurrently, the internal state gets corrupted.

### Symptoms
- Track IDs skyrocket: 1 → 1919 → 1928 → 3294 → 4165 (thousands of new IDs)
- Most tracks have `total_frames = 1` (detected once, then lost)
- Person standing still gets a new track ID every 5-10 seconds
- Zone events spam: `zone_enter` fires 7868+ times for the same person/zone
- "Anon" displayed instead of track ID on the stream
- No identities created (tracks die before accumulating 2+ good faces)

### Root Cause
`yolo_detector.py` used a shared detector pattern:
```python
# BAD — shared model across cameras
_shared_detectors[key] = YOLODetector(model_path=key)
# Both cameras call model.track(persist=True) on the SAME model instance
```

### Fix
Each camera gets its own YOLO model instance via `get_camera_detector(camera_id)`:
```python
# GOOD — per-camera model, isolated ByteTrack state
self.detector = get_camera_detector(camera_id=str(self.camera_id), ...)
```

**Files changed:**
- `app/modules/detection/yolo_detector.py` — added `get_camera_detector()`, kept `get_shared_detector()` as legacy
- `app/modules/ai_runtime/camera_worker.py` — uses `get_camera_detector()` instead of `get_shared_detector()`

### Verification
After fix: track IDs stay small (1, 2, 3, 6...), tracks persist for 100+ frames, zone events fire once per entry/exit.

---

## 2. Inference Pool Starvation

**Status:** FIXED (July 6, 2026)

### Problem
The shared inference thread pool had only 2 threads (`_MAX_INFERENCE_THREADS = 2`). With 2 cameras, one camera's heavy ReID pipeline (YOLO-Pose + OSNet + InsightFace = ~150ms per track) would occupy both threads, starving the other camera's YOLO detection.

### Symptoms
- Camera A actively tracking and doing ReID
- Camera B shows NO detections at all (person visible on stream but no bounding boxes)
- No logs from Camera B (YOLO call blocks indefinitely waiting for a thread)

### Fix
- `inference_pool.py`: reads `MAX_WORKERS` from config instead of hardcoded `2`
- `config.py`: default `MAX_WORKERS = 10`
- `.env`: set `MAX_WORKERS=12` for 2-camera setup (6 per camera)

### Recommended Thread Counts

| Cameras | MAX_WORKERS | Rationale |
|---------|-------------|-----------|
| 1 | 6 | 6 threads per camera |
| 2 | 12 | 6 × 2 |
| 3 | 18 | 6 × 3 |
| Default | 10 | Safe baseline |

### Why NOT increase uvicorn workers
The app runs with `--workers 1` deliberately. Camera workers maintain in-process state:
- `TrackManager.tracks` — active tracks, bbox history, zone state
- `self.track_embeddings` — accumulated face/body embeddings
- `self.temporary_person_ids` — temporary identities
- `RuleEvaluator.cache` — cached rules/zones
- `LatestFrameBuffer` — RTSP capture thread

Multiple processes would each have their own state copies, breaking track continuity and identity resolution.

---

## 3. Face Match Threshold Too Strict

**Status:** FIXED (July 6, 2026)

### Problem
`FACE_MATCH_THRESHOLD` was 0.55. InsightFace ArcFace embeddings for the same person from different angles typically score 0.45-0.70. At 0.55, many same-person pairs were rejected, creating duplicate identities.

### Evidence
- Same person, 2 frontal views: face similarity = 0.5249 (below 0.55, above 0.48)
- 55 persons had 43 duplicate pairs at 0.48 threshold
- Only 8 duplicate pairs at 0.60 threshold (all created by old code)

### Fix
Lowered `FACE_MATCH_THRESHOLD` from 0.55 → 0.48 in `config.py` and `.env`.

### Impact
- Same-person cross-angle matching now works (0.52 > 0.48 → match)
- Face contradiction gate still prevents false merges (different people score < 0.40)
- The `pg_advisory_xact_lock(1001)` in `decide_identity()` serializes identity creation across cameras, so threshold lowering doesn't cause race-condition duplicates

---

## 4. Face Embedding Duplicate Storage

**Status:** FIXED (July 6, 2026)

### Problem
Two bugs caused the same face to be stored multiple times per identity:

**Bug A — Accumulation dedup missing:**
`track.face_embedding_list` accumulated the best face from each 5-frame window. But if the person stood still, consecutive windows produced the SAME face crop. The list filled with duplicates, wasting the 5-slot cap.

**Bug B — Double-store after identity resolution:**
`decide_identity()` stores the best face via `_store_face_embedding()`. Then camera worker code stored ALL faces from `face_embedding_list` again — including the one already stored. The best face got stored 2x, filling the cap with duplicates.

### Fix
- **Bug A:** Before appending to `face_embedding_list`, compare cosine similarity against existing entries. Skip if > 0.95 (same crop/angle).
- **Bug B:** After identity resolution, skip any face with > 0.95 similarity to `track.best_face_embedding` (already stored by `decide_identity`).

---

## 5. Advisory Lock Behavior

### How `pg_advisory_xact_lock(1001)` Works

The lock in `decide_identity()` serializes identity creation across camera workers:

**Different sessions (cross-camera):**
```
Camera A: acquires lock → creates Person X → commit → releases lock
Camera B: BLOCKS until A commits → acquires lock → searches → finds Person X → matches
```
This works correctly.

**Same session (same camera, same frame, multiple tracks):**
```
Track A: acquires lock (no-op, same session) → creates Person X → db.flush()
Track B: acquires lock (no-op, same session) → searches → finds Person X (flushed, same transaction) → matches
```
This also works because `db.flush()` makes inserts visible within the same transaction.

### Key Insight
The lock was NEVER the cause of duplicate identities. The real cause was the **threshold being too high** (0.55 rejected valid same-person matches at 0.52 similarity).

---

## 6. Zone Event Spam (Symptom, Not Root Cause)

### Problem
When ByteTrack loses a track and creates a new one (due to the shared-model corruption in Issue #1), the `ZoneEventDetector` sees the new track as a fresh entry:
- New track has no `_last_zones` entry → `prev_zones = set()`
- New track is in the zone → `entered_zones = current_zones - set()` = the zone
- `zone_enter` fires AGAIN

### Fix
Fixed by resolving Issue #1 (per-camera YOLO model). With stable tracks, zone events fire once per genuine entry/exit.

### Note
The `ZoneEventDetector` itself is working correctly — it was reacting to ByteTrack's track ID churn. No code change needed in `zone_event_detector.py`.

---

## Diagnostic Scripts

### Check for duplicate identities
```bash
PYTHONPATH=. venv/bin/python danger/dedup_faces.py
```
Prints all identity pairs with face similarity above `FACE_MATCH_THRESHOLD`. Read-only, no modifications.

### Reset all tracking data
```bash
PYTHONPATH=. venv/bin/python danger/reset_tracking_data.py [--yes]
```
Wipes all tracking, person identity, body/face embeddings, events, analytics, and MinIO crops. Preserves config tables (cameras, zones, rules, stores).

---

## Configuration Quick Reference

| Setting | Value | File | Purpose |
|---------|-------|------|---------|
| `MAX_WORKERS` | `12` | `.env` | Inference pool threads (6 per camera × 2) |
| `FACE_MATCH_THRESHOLD` | `0.48` | `.env` | Face similarity threshold for matching |
| `FACE_IDENTITY_MIN_SCORE` | `0.60` | `config.py` | Min face quality for identity creation |
| `FACE_IDENTITY_MIN_DETECTIONS` | `2` | `config.py` | Min good face detections per track |
| `MAX_FACE_EMBEDDINGS_PER_PERSON` | `5` | `config.py` | Multi-angle face storage cap |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.30` | `.env` | Person detection confidence |
| Billing dwell threshold | `90s` | DB `rules` table | Time in billing zone before purchase event |
| Stale track timeout | `5.0s` | `track_manager.py` | Track removed if unseen for this long |
