# Retail AI Platform — Architecture & Pipeline Documentation

## Table of Contents

1. [Adding a Camera from Dashboard (V2 API)](#1-adding-a-camera-from-dashboard-v2-api)
2. [The AI Pipeline — Per-Frame Processing](#2-the-ai-pipeline--per-frame-processing)
3. [Database Schema & Significance](#3-database-schema--significance)
4. [Zone Creation with Polygons](#4-zone-creation-with-polygons)
5. [Hardcoded Events & Rules](#5-hardcoded-events--rules)
6. [Models Used in the Pipeline](#6-models-used-in-the-pipeline)

---

## 1. Adding a Camera from Dashboard (V2 API)

### Step-by-step Flow

**Frontend Action:** User clicks "Add Camera" → fills form: Camera Name, RTSP URL, selects **Store** from dropdown.

**API Call:** `POST /api/v2/cameras` → handled by `app/modules/cameras/v2_router.py:create_camera_v2`

```
POST /api/v2/cameras
Body: { name, rtsp_url, store_id, skip_rtsp_test }
                                |
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│ STEP 1: Validate Store exists                                         │
│   SELECT Store WHERE id = store_id                                    │
│   → 404 if not found                                                  │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│ STEP 2: CameraService.create_camera_v2(db, data)                      │
│                                                                       │
│   a. test_rtsp_stream(rtsp_url)                                       │
│      → cv2.VideoCapture + frame read (timeout: 10s)                   │
│      → Auto-detects resolution (e.g. 1920x1080)                       │
│      → Returns RTSPTestResponse { success, resolution, fps }          │
│                                                                       │
│   b. Creates Camera row in DB:                                        │
│      - name, rtsp_url, store_id                                       │
│      - status = INACTIVE                                              │
│      - is_active = True                                               │
│      - resolution = auto-detected                                     │
│                                                                       │
│   c. stream_path = /stream/{camera_id}                                │
│      (deterministic MediaMTX feed path, persisted after db.flush)     │
│                                                                       │
│   d. db.flush() → camera gets a UUID, stream_path is saved            │
│                                                                       │
│   IMPORTANT: Does NOT auto-start StreamManager here — the router      │
│   calls start_camera() immediately after, which handles stream setup. │
│   Starting both simultaneously causes dual-ffmpeg conflicts.           │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│ STEP 3: db.commit()                                                   │
│   CRITICAL: Commits BEFORE starting the worker so the new camera      │
│   row is visible to WorkerSupervisor's own DB connection.             │
│   Without this, WorkerSupervisor sees "Camera not found".             │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│ STEP 4: Auto-start the AI pipeline                                    │
│   CameraService.start_camera(db, camera.id):                          │
│                                                                       │
│   a. Sets camera.status = ACTIVE                                      │
│      db.flush() + db.refresh()                                        │
│                                                                       │
│   b. WorkerSupervisor.start_camera(camera_id)                         │
│      → load_runtime_config(db, camera_id)  (zones + rules from DB)    │
│      → Creates CameraWorker instance                                  │
│      → camera_worker.start():                                         │
│         - frame_buffer.start()  (RTSP pull thread)                    │
│         - asyncio.create_task(_run_loop())  (AI pipeline loop)         │
│      → Stores worker in self.workers dict                             │
│                                                                       │
│   c. Stream setup (depends on burnin_enabled):                        │
│      - burnin_enabled = TRUE (default):                               │
│        → CameraWorker._start_broadcaster()                            │
│        → StreamBroadcaster pushes annotated frames to MediaMTX        │
│        → Waits for broadcaster to be ready (15s timeout)              │
│                                                                       │
│      - burnin_enabled = FALSE:                                        │
│        → StreamManager.add_viewer(camera_id, rtsp_url)                │
│        → ffmpeg pulls RTSP → republishes to MediaMTX                  │
│        → No annotation overlay, raw stream only                       │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│ STEP 5: Returns CameraResponse                                        │
│                                                                       │
│   {                                                                    │
│     "id": "uuid",                                                      │
│     "name": "Entry Gate Cam",                                         │
│     "status": "active",                                               │
│     "store_name": "Store Name",                                       │
│     "store_zone_gate": "Gate B4",                                     │
│     "stream_path": "/stream/uuid",                                    │
│     "webrtc_url": "http://host:8889/stream/uuid/whep",                │
│     "hls_url": "http://host:8889/stream/uuid/hls",                    │
│     "resolution": "1920x1080",                                        │
│     "fps_target": 5,                                                  │
│     "reid_enabled": true,                                             │
│     "demographic_enabled": true,                                      │
│     "burnin_enabled": true                                            │
│   }                                                                    │
└───────────────────────────────────────────────────────────────────────┘
```

After this, the camera tile appears in the dashboard showing LIVE status. The AI pipeline is already running — no separate `/start` call required.

### Key Files in this Flow

| File | Role |
|------|------|
| `app/modules/cameras/v2_router.py` | API endpoint: `POST /api/v2/cameras` |
| `app/modules/cameras/service.py` | `create_camera_v2()`, `start_camera()`, `test_rtsp_stream()` |
| `app/modules/cameras/schemas.py` | `CameraCreateV2`, `CameraResponse` schemas |
| `app/modules/ai_runtime/worker_supervisor.py` | Manages camera workers lifecycle |
| `app/modules/ai_runtime/camera_worker.py` | Per-camera AI pipeline |
| `app/modules/streaming/mediamtx.py` | MediaMTX path generation, WebRTC/HLS URLs |
| `app/modules/streaming/manager.py` | StreamManager: ffmpeg RTSP pull → MediaMTX push |
| `app/core/db/models/camera.py` | Camera + Zone ORM models |
| `app/core/db/models/store.py` | Store ORM model |

---

## 2. The AI Pipeline — Per-Frame Processing

### File: `app/modules/ai_runtime/camera_worker.py`

The pipeline runs at `fps_target` (default 5 FPS, configurable per camera). All models are shared across cameras (singleton pattern).

### Lifecycle

```
CameraWorker.__init__(camera_config, runtime_config)
    │
    ├── detector = get_shared_detector(YOLOv8n)  ← shared across all cameras
    ├── track_manager = TrackManager(camera_id)
    ├── zone_event_detector = ZoneEventDetector(camera_id)
    ├── rule_evaluator = RuleEvaluator()
    ├── reid_extractor = get_shared_extractor(OSNet)  ← shared
    ├── identity_engine = IdentityDecisionEngine()
    ├── insightface_analyzer = get_shared_analyzer(InsightFace)  ← shared
    ├── frame_buffer = LatestFrameBuffer(rtsp_url)
    └── stream_broadcaster = StreamBroadcaster(...)  ← if burnin_enabled
    │
    ▼
CameraWorker.start()
    ├── frame_buffer.start()  ← background thread: cv2.VideoCapture → deque
    ├── is_running = True
    ├── _task = asyncio.create_task(_run_loop())
    └── if burnin_enabled: _start_broadcaster()
    │
    ▼
CameraWorker.stop()
    ├── is_running = False
    ├── _task.cancel()
    ├── stream_broadcaster.stop()
    ├── frame_buffer.stop()
    ├── _close_all_track_sessions()  ← close open DB sessions
    └── track_manager.reset()
```

### Main Loop: `_run_loop()`

```
while self.is_running:
    │
    ├── frame, frame_ts = frame_buffer.get_latest()  ← non-blocking, returns latest frame
    │
    ├── WATCHDOG: If no new frame for 30s → reset frame_buffer (reconnect RTSP)
    │
    ├── PERIODIC RESET: Every 6 hours → reset all tracker state
    │   (close all track sessions, reset ByteTrack internal state)
    │
    ├── Skip if frame is None or duplicate timestamp
    │
    └── await _process_frame(frame)
        │
        └── Maintains fps_target via sleep(interval - elapsed)
```

### Pipeline Stages: `_process_frame(frame)`

```
┌───────────────────────────────────────────────────────────────────────┐
│                                                                       │
│                        RTSP STREAM                                     │
│                            │                                           │
│                LatestFrameBuffer (background thread)                   │
│                            │                                           │
│                      latest frame                                      │
│                            │                                           │
└────────────────────────────┬──────────────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────────────┐
│ STAGE 1: YOLO Detection + ByteTrack Tracking                           │
│                                                                       │
│   detector.track(frame) → tracked_detections                          │
│                                                                       │
│   Model: YOLOv8n                                                      │
│   Function: model.track() — combines detection + tracking in one pass │
│   Returns: [TrackedDetection(track_id, bbox, confidence, class), ...] │
│                                                                       │
│   Runs on shared InferencePool (ThreadPoolExecutor) to keep models    │
│   on GPU and prevent blocking the asyncio event loop.                 │
│                                                                       │
│   Confidence threshold: configurable (YOLO_CONFIDENCE_THRESHOLD)      │
│   Allowed classes: person only (filtered from COCO classes)           │
│                                                                       │
│   Key file: app/modules/detection/yolo_detector.py                    │
│   Key file: app/modules/ai_runtime/inference_pool.py                  │
└────────────────────────────┬──────────────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Track State Update + Zone Membership                          │
│                                                                       │
│   For each tracked_detection:                                         │
│                                                                       │
│   2a. track_manager.update_track(track_id, bbox, confidence)          │
│       → Creates or updates ActiveTrack in memory                      │
│       → Maintains: prev_bbox, bbox_history (last 30),                 │
│         total_frames, confidence_sum, stability_score                 │
│       → Stability score: variance of bbox centers over time           │
│         (lower variance = higher stability, 0-1 scale)                │
│                                                                       │
│   2b. track_manager.update_zones(track, zones, width, height)         │
│       → For each zone, converts polygon from % coords (0-100)         │
│         to pixel coordinates                                           │
│       → point_in_polygon() on BOTH:                                   │
│         - bottom_center (feet position)                               │
│         - bbox_center (body position)                                 │
│       → Updates track.current_zones (set of zone UUID strings)        │
│       → Updates track.zone_enter_times (when first entered)           │
│       → Updates track.dwell_seconds (time since entry, per zone)      │
│                                                                       │
│   Classifications after processing:                                   │
│   - new_tracks:    track_session_id is None → first appearance       │
│   - reid_tracks:   should_run_reid() → bbox height > 100px,          │
│                    if confident: only every 5th frame                  │
│   - observations:  sampled every 2s per track (OBS_SAMPLE_SECONDS)   │
│                                                                       │
│   Key file: app/modules/tracking/track_manager.py                     │
│   Key file: app/utils/geometry.py                                     │
└────────────────────────────┬──────────────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Automatic Zone Event Detection                                │
│                                                                       │
│   zone_event_detector.detect(active_tracks)                           │
│                                                                       │
│   Compares current_zones vs _last_zones (per track, frame-over-frame) │
│                                                                       │
│   Three event types fired:                                            │
│                                                                       │
│   ┌───────────────────┬────────────────────────────────────────────┐ │
│   │ zone_enter        │ current_zones - prev_zones ≠ ∅             │ │
│   │                   │ Person entered a zone polygon               │ │
│   ├───────────────────┼────────────────────────────────────────────┤ │
│   │ zone_exit         │ prev_zones - current_zones ≠ ∅             │ │
│   │                   │ Person left a zone polygon                  │ │
│   │                   │ Includes dwell_seconds in metadata          │ │
│   ├───────────────────┼────────────────────────────────────────────┤ │
│   │ zone_dwell_       │ Dwell reaches 30s, 60s, or 120s            │ │
│   │ milestone         │ Fires only once per threshold per track:zone│ │
│   │                   │ Fires highest matching threshold first      │ │
│   └───────────────────┴────────────────────────────────────────────┘ │
│                                                                       │
│   Memory management: tracks removed from _last_zones when stale       │
│   Key file: app/modules/rule_engine/zone_event_detector.py            │
└────────────────────────────┬──────────────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────────────┐
│ STAGE 4: Rule Evaluation (Pure in-memory)                              │
│                                                                       │
│   rule_evaluator.evaluate(camera_id, active_tracks, w, h)             │
│                                                                       │
│   Rules are loaded from RuleCache (loaded once from DB, refreshed     │
│   via /api/runtime/reload-config or when zone/rule CRUD occurs).      │
│                                                                       │
│   For each enabled rule, evaluates against each active track:         │
│                                                                       │
│   ┌──────────────────────┬─────────────────────────────────────────┐  │
│   │ line_crossing        │ Detects bbox bottom-center crossing a   │  │
│   │                      │ 2-point line (shape=line zone).         │  │
│   │                      │ Compares prev_bbox → curr_bbox.         │  │
│   │                      │ Normalizes to [0-1] range.              │  │
│   │                      │ Event: line_crossing_{direction}        │  │
│   │                      │ (left/right)                            │  │
│   ├──────────────────────┼─────────────────────────────────────────┤  │
│   │ zone_dwell           │ Fires when dwell_seconds ≥              │  │
│   │                      │ dwell_threshold_seconds.                │  │
│   │                      │ Event: zone_dwell_exceeded              │  │
│   ├──────────────────────┼─────────────────────────────────────────┤  │
│   │ billing_interaction  │ Person in zone with zone_type =         │  │
│   │                      │ billing_zone for ≥ threshold (10s).     │  │
│   │                      │ ALSO creates BillingInteraction row.    │  │
│   │                      │ Event: billing_interaction              │  │
│   ├──────────────────────┼─────────────────────────────────────────┤  │
│   │ queue_count          │ PLACEHOLDER — not implemented per-track │  │
│   ├──────────────────────┼─────────────────────────────────────────┤  │
│   │ possible_purchase    │ Person dwells in zone with zone_type =  │  │
│   │                      │ purchase_intent ≥ 15s.                  │  │
│   │                      │ Event: purchase (severity=HIGH)         │  │
│   ├──────────────────────┼─────────────────────────────────────────┤  │
│   │ restricted_zone      │ Person enters restricted_zone polygon.  │  │
│   │                      │ Event: restricted_zone_intrusion        │  │
│   │                      │ (severity=alert)                        │  │
│   └──────────────────────┴─────────────────────────────────────────┘  │
│                                                                       │
│   Cooldown mechanism: Each rule has cooldown_seconds (default 30s).   │
│   After firing for [rule_id, track_id] pair, won't fire again until   │
│   cooldown expires. Prevents event spam.                              │
│                                                                       │
│   Cooldown tracker is periodically pruned to prevent memory leaks.    │
│                                                                       │
│   Key file: app/modules/rule_engine/rule_evaluator.py                 │
│   Key file: app/modules/rule_engine/config_loader.py                  │
└────────────────────────────┬──────────────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────────────┐
│ STAGE 5: Stale Track Cleanup                                          │
│                                                                       │
│   track_manager.cleanup_stale_tracks()                                │
│   → Removes tracks not seen for > 5 seconds                           │
│   → Returns stale tracks for DB session closure                       │
│   → Also cleans up _last_obs_time and track_embeddings buffers        │
└────────────────────────────┬──────────────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────────────┐
│ STAGE 6: _persist_batch (One DB Transaction)                           │
│                                                                       │
│   Opens 1 AsyncSessionLocal() connection.                             │
│   ALL writes in a single transaction.                                 │
│   Skips entirely if no work to do (quiet frames).                    │
│                                                                       │
│   ┌─────────────────────────────────────────────────────────────────┐ │
│   │ 6a. _create_track_session (for new_tracks)                      │ │
│   │                                                                  │ │
│   │   INSERT INTO track_sessions:                                    │ │
│   │   - camera_id, local_track_id, started_at, last_seen_at         │ │
│   │   - total_frames, is_active=True                                │ │
│   │   - Extract initial crop → save to MinIO → best_crop_path       │ │
│   │                                                                  │ │
│   │   HARDCODED EVENT: "person_entered_view"                        │ │
│   │   INSERT INTO events:                                           │ │
│   │   - camera_id, track_session_id, event_type="person_entered_view"│ │
│   │   - severity=LOW, snapshot_path, occurred_at                    │ │
│   │   - person_identity_id=None (resolved later by ReID)            │ │
│   │   - metadata_json: { local_track_id }                           │ │
│   └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│   ┌─────────────────────────────────────────────────────────────────┐ │
│   │ 6b. _run_reid (for reid_tracks)                                 │ │
│   │                                                                  │ │
│   │   Complete ReID pipeline per track:                             │ │
│   │                                                                  │ │
│   │   Step 1: extract_crop(frame, bbox) → OpenCV crop               │ │
│   │                                                                  │ │
│   │   Step 2: assess_crop_quality(crop)                              │ │
│   │     → Blur detection, size check, aspect ratio                  │ │
│   │     → Reject if quality < REID_CROP_QUALITY_THRESHOLD           │ │
│   │                                                                  │ │
│   │   Step 3: reid_extractor.extract(crop)                           │ │
│   │     → Model: OSNet (torchreid)                                  │ │
│   │     → Returns: 512-dim body embedding vector                    │ │
│   │     → Runs on InferencePool (ThreadPoolExecutor)                │ │
│   │                                                                  │ │
│   │   Step 4 (if demographic_enabled):                               │ │
│   │     insightface_analyzer.analyze(crop)                           │ │
│   │     → Model: InsightFace buffalo_l                               │ │
│   │     → Returns: gender, age, age_group, face_embedding (512-dim),│ │
│   │       face_score, face_crop (aligned)                           │ │
│   │     → Saves best face_crop to MinIO                             │ │
│   │     → Updates track.best_demographics (keeps highest face_score)│ │
│   │                                                                  │ │
│   │   Step 5: Accumulate in track_embeddings buffer                  │ │
│   │     track_embeddings[local_track_id] = list of tuples:           │ │
│   │     (embedding, quality, crop_path,                             │ │
│   │      face_embedding, face_score, face_crop_path)                │ │
│   │                                                                  │ │
│   │   Step 6: On 5th accumulated frame (REID_ACCUMULATION_FRAMES):  │ │
│   │     a. mean_embedding = np.mean(embeddings, axis=0)              │ │
│   │     b. L2 normalize mean_embedding                               │ │
│   │     c. Find best_crop (highest quality) and best_face            │ │
│   │        (highest face_score) in the window                       │ │
│   │     d. Delete unused crops from MinIO (batched cleanup)         │ │
│   │     e. identity_engine.decide_identity(                          │ │
│   │          db, mean_embedding, camera_id, crop_quality,           │ │
│   │          crop_path, current_person_id, previous_score,          │ │
│   │          is_temporary, face_embedding, face_score, face_crop_path│ │
│   │        )                                                         │ │
│   │        → Cosine similarity search on pgvector PersonEmbedding    │ │
│   │        → Also searches PersonFaceEmbedding for face matching    │ │
│   │        → If match score > threshold: returns existing person    │ │
│   │        → If no match: creates NEW PersonIdentity (temporary)    │ │
│   │        → Temporary IDs promoted on re-sighting                  │ │
│   │        → Returns: (person_id, score, is_confident, is_new,     │ │
│   │          prune_old_id)                                          │ │
│   │                                                                  │ │
│   │     f. UPDATE track_sessions SET person_identity_id = person_id │ │
│   │                                                                  │ │
│   │     g. If first time resolved: UPDATE the person_entered_view   │ │
│   │        event with person_identity_id and updated description    │ │
│   │                                                                  │ │
│   │   Accumulation buffer cleared after decision. ReID continues     │ │
│   │   running (for confident tracks: every 5th frame) to allow      │ │
│   │   correction if a higher-quality match appears later.           │ │
│   └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│   ┌─────────────────────────────────────────────────────────────────┐ │
│   │ 6c. _persist_events (rule_events + zone_events)                 │ │
│   │                                                                  │ │
│   │   Saves frame snapshot to MinIO (shared across all events in    │ │
│   │   this batch).                                                  │ │
│   │                                                                  │ │
│   │   For each RuleEvent:                                           │ │
│   │     INSERT INTO events:                                         │ │
│   │     - camera_id, rule_id, zone_id, person_identity_id           │ │
│   │     - track_session_id, event_type, severity, description       │ │
│   │     - snapshot_path, metadata_json, occurred_at                 │ │
│   │                                                                  │ │
│   │     Severity mapping:                                           │ │
│   │     - purchase → HIGH                                           │ │
│   │     - All others → LOW                                          │ │
│   │                                                                  │ │
│   │     If rule_type == "billing_interaction":                      │ │
│   │       ALSO INSERT INTO billing_interactions:                    │ │
│   │       - camera_id, person_identity_id, track_session_id         │ │
│   │       - zone_id, entered_at, dwell_seconds, interaction_type    │ │
│   │       - metadata_json                                           │ │
│   │                                                                  │ │
│   │   For each ZoneEvent:                                           │ │
│   │     INSERT INTO events:                                         │ │
│   │     - rule_id = None (auto-detected, not rule-based)            │ │
│   │     - severity = LOW                                            │ │
│   │     - event_type: zone_enter | zone_exit | zone_dwell_milestone │ │
│   └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│   ┌─────────────────────────────────────────────────────────────────┐ │
│   │ 6d. _persist_observations (sampled every 2s per track)          │ │
│   │                                                                  │ │
│   │   For each observation:                                         │ │
│   │     INSERT INTO track_observations:                             │ │
│   │     - track_session_id, timestamp, bbox, confidence             │ │
│   │     - zone_ids (list of zone UUIDs the person is in)            │ │
│   │                                                                  │ │
│   │   Used for: playback (trajectory reconstruction),               │ │
│   │   heatmap generation, analytics                                  │ │
│   └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│   ┌─────────────────────────────────────────────────────────────────┐ │
│   │ 6e. _close_track_session (for stale tracks)                     │ │
│   │                                                                  │ │
│   │   UPDATE track_sessions SET:                                    │ │
│   │   - ended_at = now()                                            │ │
│   │   - last_seen_at, is_active = False                             │ │
│   │   - total_frames, avg_confidence, stability_score               │ │
│   │   - bbox_history (last 30), best_crop_path                      │ │
│   │   - gender, age_group (if demographics captured)                │ │
│   │                                                                  │ │
│   │   HARDCODED EVENT: "person_left_view"                           │ │
│   │   INSERT INTO events:                                           │ │
│   │   - event_type = "person_left_view"                             │ │
│   │   - severity = LOW                                              │ │
│   │   - metadata_json: { total_frames, duration_seconds }           │ │
│   │                                                                  │ │
│   │   UPDATE person_identities:                                     │ │
│   │   - If track.best_demographics has higher face_score than       │ │
│   │     current person.best_face_score:                             │ │
│   │     → UPDATE gender, age_group, estimated_age, best_face_score, │ │
│   │       face_crop_path            │ │
│   └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│   ┌─────────────────────────────────────────────────────────────────┐ │
│   │ 6f. db.commit() — ALL OR NOTHING                                │ │
│   │   On failure: db.rollback() + re-raise exception                │ │
│   └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│   Key file: app/modules/reid/osnet_extractor.py                       │
│   Key file: app/modules/reid/insightface_analyzer.py                  │
│   Key file: app/modules/reid/crop_quality.py                          │
│   Key file: app/modules/reid/identity_decision_engine.py              │
│   Key file: app/utils/image_utils.py                                  │
│   Key file: app/utils/geometry.py                                     │
└────────────────────────────┬──────────────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────────────┐
│ STAGE 7: Update latest_tracks for StreamBroadcaster (burn-in)         │
│                                                                       │
│   self.latest_tracks.clear() + extend()  (IN-PLACE mutation)          │
│   → List of bounding boxes visible to the broadcaster thread         │
│   → StreamBroadcaster runs in a separate thread, reads this list     │
│     and draws bounding boxes + zone polygons on annotated frames      │
│   → Pushes to MediaMTX via ffmpeg pipe (WHEP/HLS)                    │
│                                                                       │
│   IMPORTANT: Must mutate in-place (clear+extend), not reassign,       │
│   because StreamBroadcaster holds a reference to the same list object │
│                                                                       │
│   Key file: app/modules/ai_runtime/stream_broadcaster.py              │
└────────────────────────────┬──────────────────────────────────────────┘
                             │
                             ▼
┌───────────────────────────────────────────────────────────────────────┐
│ STAGE 8: Optional GUI Display (cv2.imshow for debugging)              │
│                                                                       │
│   Only if RUNTIME_SHOW_GUI = True and a display is available.         │
│   Draws:                                                              │
│   - Zone polygons (green semi-transparent overlay)                    │
│   - Bounding boxes:                                                   │
│     - Blue: default track                                             │
│     - Yellow: ReID resolved but not confident                        │
│     - Cyan: ReID resolved & confident                                │
│   - Labels: local_track_id, person_id (if resolved), demographics     │
└───────────────────────────────────────────────────────────────────────┘
```

### Pipeline Summary

```
RTSP → LatestFrameBuffer → [YOLO Detect+Track] → [Track+Zone State]
    → [Zone Event Detection] → [Rule Evaluation]
    → [Stale Cleanup] → [Persist Batch: TrackSession + ReID + Events + Observations]
    → [StreamBroadcaster (burn-in)] → [GUI (optional)]
```

---

## 3. Database Schema & Significance

### Entity Relationship Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│                                                                       │
│  stores                    store_zones             store_categories    │
│  ──────                    ───────────             ─────────────────   │
│  id (PK)                   id (PK)                 id                  │
│  name                      name                    name                │
│  zone_gate                 store_id (FK)           icon                │
│  status                    ...                                         │
│  address                                                              │
│  ...                                                                   │
│                                                                       │
│  A store has many cameras.                                            │
│  A store has many store_zones (physical positions within the store).  │
│  store_categories and store_levels are lookup tables for store types. │
│                                                                       │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                │ store_id (FK)
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│  cameras                                                              │
│  ──────────                                                           │
│  id (PK, UUID)          name (varchar 255)                            │
│  rtsp_url (varchar 500)  stream_path (varchar 255, nullable)          │
│  status (enum: active | inactive | maintenance | error)                │
│  fps_target (int, default 5)                                          │
│  resolution (varchar 20, default "1920x1080")                         │
│  detection_model (varchar 100, default "yolov8n")                     │
│  reid_enabled (bool, default true)                                    │
│  demographic_enabled (bool, default true)                             │
│  frame_rotation (int, nullable: 90/180/270)                           │
│  location_description (varchar 500, nullable)                         │
│  is_active (bool, default true)                                       │
│  burnin_enabled (bool, default true)                                  │
│  area_id (UUID, FK → areas, deprecated in V2)                        │
│  store_id (UUID, FK → stores, ondelete SET NULL)                     │
│  zone_id (UUID, FK → store_zones, ondelete SET NULL)                 │
│  created_at, updated_at (timestamps)                                  │
│                                                                       │
│  CAMERA is the CENTRAL ENTITY. Everything references it.             │
│  One camera = one RTSP source = one AI pipeline worker.              │
│  Static mounting — no PTZ, no ROI/view concept.                      │
│  Detection runs on the full frame and is filtered by zones.          │
│                                                                       │
│  Relationships:                                                       │
│  - camera.store → Store (many-to-one)                                │
│  - camera.store_zone → StoreZone (many-to-one)                       │
│  - camera.zones → [Zone] (one-to-many, cascade delete-orphan)        │
│  - camera.track_sessions → [TrackSession] (one-to-many)              │
│  - camera.events → [Event] (one-to-many)                             │
│  - camera.billing_interactions → [BillingInteraction] (one-to-many)  │
│                                                                       │
└───┬───────────────────┬───────────────────────┬───────────────────────┘
    │                   │                       │
    ▼                   ▼                       ▼
┌──────────────┐  ┌──────────────────┐  ┌───────────────────────────────┐
│  zones       │  │  track_sessions  │  │  events                        │
│  ─────       │  │  ────────────── │  │  ──────                        │
│  id (PK)     │  │  id (PK)         │  │  id (PK)                       │
│  camera_id   │  │  camera_id (FK)  │  │  camera_id (FK, index)         │
│  name        │  │  local_track_id  │  │  rule_id (FK, nullable)        │
│  zone_type   │  │  person_ident_id │  │  zone_id (FK, nullable)        │
│  (enum)      │  │  started_at      │  │  person_identity_id (FK,index) │
│  shape       │  │  last_seen_at    │  │  track_session_id (FK)         │
│  (enum)      │  │  ended_at        │  │  event_type (varchar 100,idx)  │
│  polygon     │  │  bbox_history    │  │  severity (varchar 50)         │
│  (JSONB)     │  │  (JSONB)         │  │  description (text)            │
│  is_active   │  │  avg_confidence  │  │  metadata_json (JSONB)         │
│              │  │  total_frames    │  │  snapshot_path (varchar 500)   │
│  ForeignKey: │  │  stability_score │  │  clip_path (varchar 500)       │
│   cameras.id │  │  is_active       │  │  occurred_at (timestamp,index) │
│  ON DELETE   │  │  gender          │  │  is_acknowledged (bool)        │
│   CASCADE    │  │  age_group       │  │  is_false_positive (bool)      │
│              │  │  best_crop_path  │  │  acknowledged_by (FK→users)     │
│  ZONES are   │  │                  │  │                                │
│  polygons    │  │  ForeignKey:     │  │  EVENTS are the output         │
│  drawn on    │  │   cameras.id     │  │  artifact of the pipeline.     │
│  camera feed. │  │   ON DELETE     │  │  Generated by:                 │
│  zone_type   │  │   CASCADE        │  │  - RuleEvaluator (rules)       │
│  determines   │  │                  │  │  - ZoneEventDetector (auto)   │
│  which AI     │  │  Sessions       │  │  - CameraWorker hardcoded     │
│  analytics    │  │  represent one  │  │    (person_entered_view,       │
│  event it     │  │  person's visit │  │     person_left_view)          │
│  measures.    │  │  start→end.     │  │                                │
│              │  │                  │  │  ForeignKey:                   │
│  zone_type    │  │  track_observations                              │  │
│  enum values: │  │  ─────────────────                                │  │
│  - footfall   │  │  id (PK)                                           │  │
│  - dwell_time │  │  track_session_id (FK, index)                      │  │
│  - queue_     │  │  timestamp (timestamp)                            │  │
│    length     │  │  bbox (JSONB)                                     │  │
│  - entry_exit │  │  confidence (float)                               │  │
│  - heatmap    │  │  zone_ids (JSONB, nullable)                       │  │
│  - purchase_  │  │                                                    │  │
│    intent     │  │  ForeignKey:                                       │  │
│  - entry_line │  │   track_sessions.id                               │  │
│  - exit_line  │  │   ON DELETE CASCADE                               │  │
│  - billing_   │  │                                                    │  │
│    zone       │  │  Sampled every 2s per track.                      │  │
│  - queue_zone │  │  Used for playback, heatmap, analytics.           │  │
│  - product_   │  │                                                    │  │
│    zone       │  │                                                    │  │
│  - ignore_zone│  │                                                    │  │
│  - restricted │  │                                                    │  │
│  - medicine_  │  │                                                    │  │
│    pickup     │  │                                                    │  │
│              │  │                                                    │  │
│  shape enum: │  │                                                    │  │
│  - polygon    │  │                                                    │  │
│  - line       │  │                                                    │  │
└──────┬───────┘  └──────────────────┘  └───────────────────────────────┘
       │
       ▼
┌───────────────────────────────────────────────────────────────────────┐
│  rules                                                                 │
│  ─────                                                                 │
│  id (PK, UUID)                                                         │
│  name (varchar 255)                                                    │
│  rule_type (enum):                                                     │
│    - line_crossing                                                     │
│    - zone_dwell                                                        │
│    - billing_interaction                                               │
│    - queue_count                                                       │
│    - possible_purchase                                                 │
│    - restricted_zone                                                   │
│  zone_id (UUID, FK → zones.id, ON DELETE SET NULL, nullable)          │
│  camera_id (UUID, FK → cameras.id, ON DELETE SET NULL, nullable)      │
│  config (JSONB, nullable, default {})                                  │
│  cooldown_seconds (int, default 30)                                    │
│  severity (varchar 20, default "info")                                 │
│  dwell_threshold_seconds (int, nullable)                               │
│  count_threshold (int, nullable)                                       │
│  is_enabled (bool, default true)                                       │
│  created_at, updated_at                                                │
│                                                                        │
│  RULES link a zone to a behavior:                                      │
│  "When a person fulfills condition X in zone Y, fire event Z"         │
│                                                                        │
│  camera_id=NULL means global rule (applies to all cameras).            │
│  camera_id=UUID means camera-scoped rule.                             │
│                                                                        │
│  ForeignKey: zones.id (nullable)                                       │
│  ForeignKey: cameras.id (nullable)                                     │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│  person_identities                                                     │
│  ─────────────────                                                     │
│  id (PK, UUID)                                                         │
│  label (varchar 255, nullable) — operator-assigned name               │
│  first_seen_at (timestamp)                                             │
│  last_seen_at (timestamp)                                              │
│  visit_count (int, default 1)                                          │
│  metadata_json (JSONB, nullable)                                       │
│  is_anonymous (bool, default true)                                     │
│  gender (varchar 10, nullable)                                         │
│  age_group (varchar 20, nullable)                                      │
│  estimated_age (int, nullable)                                         │
│  best_face_score (float, nullable)                                     │
│  face_crop_path (varchar 500, nullable)                               │
│  created_at, updated_at                                                │
│                                                                        │
│  Represents a UNIQUE PERSON across all cameras.                        │
│  Cross-camera ReID: same person seen on multiple cameras →             │
│  same PersonIdentity (matched via pgvector embeddings).               │
│  is_anonymous=True until labeled by an operator.                      │
│                                                                        │
│  Relationships:                                                        │
│  - embeddings → [PersonEmbedding] (one-to-many)                       │
│  - face_embeddings → [PersonFaceEmbedding] (one-to-many)              │
│  - track_sessions → [TrackSession] (one-to-many)                      │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ person_embeddings                                                │  │
│  │ ────────────────                                                 │  │
│  │ id (PK, UUID)                                                    │  │
│  │ person_identity_id (FK, index)                                   │  │
│  │ embedding (pgvector, 512-dim) ← OSNet body embedding             │  │
│  │ camera_id (UUID, FK, nullable)                                   │  │
│  │ crop_quality (float, default 0.0)                                │  │
│  │ crop_path (varchar 500)                                          │  │
│  │ captured_at (timestamp)                                          │  │
│  │                                                                  │  │
│  │ Used for person re-identification (body appearance).             │  │
│  │ Cosine similarity search via pgvector.                           │  │
│  │ ForeignKey: person_identities.id ON DELETE CASCADE              │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ person_face_embeddings                                           │  │
│  │ ──────────────────────                                           │  │
│  │ id (PK, UUID)                                                    │  │
│  │ person_identity_id (FK, index)                                   │  │
│  │ embedding (pgvector, 512-dim) ← InsightFace face embedding       │  │
│  │ camera_id (UUID, FK, nullable)                                   │  │
│  │ face_score (float, default 0.0)                                  │  │
│  │ face_crop_path (varchar 500)                                     │  │
│  │ captured_at (timestamp)                                          │  │
│  │                                                                  │  │
│  │ Used for face-based identity matching + demographics.            │  │
│  │ ForeignKey: person_identities.id ON DELETE CASCADE              │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│  billing_interactions                                                  │
│  ───────────────────                                                   │
│  id (PK, UUID)                                                         │
│  camera_id (UUID, FK)                                                  │
│  person_identity_id (UUID, FK)                                        │
│  track_session_id (UUID, FK)                                          │
│  zone_id (UUID, FK)                                                    │
│  entered_at (timestamp)                                                │
│  dwell_seconds (float)                                                │
│  interaction_type (varchar 50)                                        │
│  metadata_json (JSONB)                                                │
│                                                                        │
│  Separate structured record for billing counter interactions.         │
│  Created when rule_type=billing_interaction fires.                    │
│  ForeignKey: cameras.id ON DELETE CASCADE                             │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────┐
│  daily_analytics_summary                                               │
│  ───────────────────────                                               │
│  id (PK, UUID)                                                         │
│  store_id (UUID, FK)                                                   │
│  date (date)                                                           │
│  event_type (varchar 100)                                             │
│  count (int)                                                           │
│                                                                        │
│  Pre-aggregated analytics for dashboard widgets.                      │
│  Generated by background scheduler jobs.                              │
│  Key file: app/modules/jobs/scheduler.py                               │
│  Key file: app/modules/jobs/tasks.py                                   │
└───────────────────────────────────────────────────────────────────────┘
```

### Database Significance Summary

| Table | Purpose | Lifecycle |
|-------|---------|-----------|
| **cameras** | Central entity. One row = one RTSP source = one AI worker. | Created via API, deleted via API. Cascade deletes zones, track_sessions, events, billing_interactions. |
| **zones** | Detection polygons drawn on camera feed. `cascade="all, delete-orphan"` from camera. | Created via polygon editor, managed via V2 zones API. |
| **track_sessions** | One person's visit (enter → exit). Links to PersonIdentity when ReID resolves. | Created when new track appears, closed when track goes stale (5s). Stores demographics from best face. |
| **track_observations** | Sampled trajectory points (every 2s per track). | Sampled during active track. Used for playback and heatmap generation. |
| **events** | The output artifact. All detected behaviors become events. | Generated from 4 sources: hardcoded (enter/leave), ZoneEventDetector (enter/exit/dwell_milestone), RuleEvaluator (6 rule types). Has acknowledgment/false_positive flags. |
| **person_identities** | Cross-camera person identity. Anonymous until labeled. | Created by IdentityDecisionEngine when a new person is seen. Matched via pgvector cosine similarity on body (OSNet) and face (InsightFace) embeddings. |
| **person_embeddings** | Body embeddings (OSNet, 512-dim) for re-identification. | Stored per person_identity. Used for cosine similarity matching. |
| **person_face_embeddings** | Face embeddings (InsightFace, 512-dim) for face matching + demographics. | Stored per person_identity. Used for face-based identity verification. |
| **rules** | Configurable behavior: "when condition X in zone Y, fire event Z". | Created via `/api/rules` CRUD API. 6 hardcoded rule_type enums. Evaluated in-memory by RuleEvaluator each frame. |
| **billing_interactions** | Structured record of billing counter interactions. | Created when billing_interaction rule fires. Separate from generic events for structured analytics. |
| **daily_analytics_summary** | Pre-aggregated analytics for dashboard widgets. | Generated by background scheduler. |

### Key Design Decisions

1. **Camera is the central entity.** Everything is scoped to a camera. Deleting a camera cascades: zones, track_sessions, events, billing_interactions.

2. **Two distinct "zone" concepts:**
   - **Store Zones** (`store_zones` table): Physical locations within a store (e.g., "Gate B4", "Entry", "Checkout"). Managed via `/api/stores/zones`.
   - **Detection Zones** (`zones` table): Polygons drawn on a camera's live feed. Managed via `/api/v2/cameras/{id}/zones`. Each zone has a `zone_type` that determines which AI analytics it measures.

3. **pgvector for embeddings.** PostgreSQL with pgvector extension for 512-dim vector storage and cosine similarity search. Avoids needing a separate vector database.

4. **JSONB for flexible data.** bbox, polygon, metadata_json, config all use JSONB for schema flexibility without migrations.

5. **Cascade deletes.** Most foreign keys use ON DELETE CASCADE or SET NULL. Camera deletion cleans up all related data.

6. **Two embedding types.** Body (OSNet) and Face (InsightFace) are stored separately. Both are used for identity matching with different confidence thresholds.

---

## 4. Zone Creation with Polygons

### Flow: User clicks Eye Icon → draws polygon on live feed

```
Dashboard Camera Row → Eye Icon (👁)
    │
    ▼
┌───────────────────────────────────────────────────────────────────────┐
│ GET /api/v2/cameras/{camera_id}/polygon-editor                        │
│                                                                       │
│ Returns:                                                              │
│ - Camera identity + store context (name, zone_gate)                   │
│ - Live stream URLs (WebRTC/HLS) so the frontend can render the feed   │
│ - All existing detection zones with their polygon points and types    │
│ - available_event_types list for the event-type dropdown              │
│                                                                       │
│ Response: CameraPolygonEditorResponse {                               │
│   id, name, store_id, store_name, store_zone_gate, status,           │
│   stream_path, webrtc_url, hls_url,                                   │
│   zones: [{id, name, zone_type, zone_type_label, shape, polygon,     │
│             is_active}],                                              │
│   available_event_types: [{value: "footfall", label: "Footfall"},    │
│                           {value: "dwell_time", label: "Dwell Time"}, │
│                           ...]                                        │
│ }                                                                     │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│ Frontend renders:                                                     │
│ - Live video feed (WebRTC or HLS)                                    │
│ - Existing polygons overlaid on the feed (if any)                    │
│ - Drawing tool for new polygons                                       │
│ - Event type dropdown for each zone                                   │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ User draws polygon on the frame
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│ POST /api/v2/cameras/{camera_id}/zones                                │
│                                                                       │
│ Body: {                                                               │
│   "name": "Billing Counter",                                          │
│   "zone_type": "footfall",           ← default, functional immediately│
│   "shape": "polygon",                                                 │
│   "polygon": {                                                        │
│     "points": [[x1,y1], [x2,y2], [x3,y3], ...]                       │
│   },                                                                  │
│   "is_active": true                                                   │
│ }                                                                     │
│                                                                       │
│ Valid zone_type values:                                               │
│   footfall | dwell_time | queue_length | entry_exit | heatmap |       │
│   purchase_intent                                                      │
│ Also accepts legacy types:                                            │
│   entry_line | exit_line | billing_zone | queue_zone |                │
│   product_zone | ignore_zone | restricted_zone |                      │
│   medicine_pickup_zone                                                │
│                                                                       │
│ → INSERT INTO zones (camera_id, name, zone_type, shape, polygon,      │
│   is_active)                                                          │
│ → Returns DetectionZoneResponse with zone_type_label                  │
└───────────────────────────────┬───────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│ CameraWorker picks up the new zone:                                   │
│                                                                       │
│ 1. After zone is created, the frontend (or operator) triggers         │
│    POST /api/runtime/reload-config                                    │
│    → WorkerSupervisor.reload_camera_config(camera_id)                 │
│    → Loads fresh zones + rules from DB                                │
│    → CameraWorker.apply_runtime_config(runtime_config)                │
│                                                                       │
│ 2. On each frame, TrackManager.update_zones() does:                  │
│    a. For each zone, convert polygon from % coords (0-100)            │
│       to pixel coordinates:                                           │
│       pixel_poly = [(p[0] * width / 100, p[1] * height / 100)]       │
│    b. Check both bottom_center (feet) AND bbox_center (body)          │
│       against the polygon using point_in_polygon()                    │
│    c. Update track.current_zones, zone_enter_times, dwell_seconds     │
│                                                                       │
│ 3. ZoneEventDetector starts tracking zone entries/exits/dwell        │
│    milestones for this zone.                                          │
│                                                                       │
│ 4. If a rule is created linking this zone to a behavior,              │
│    RuleEvaluator evaluates it each frame.                             │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ User changes event type from dropdown
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│ PUT /api/v2/cameras/{camera_id}/zones/{zone_id}                       │
│                                                                       │
│ Body: { "zone_type": "dwell_time" }                                   │
│                                                                       │
│ → Updates zone.zone_type in DB                                        │
│ → After reload-config, CameraWorker picks up the change               │
│ → The zone now measures dwell_time instead of footfall                │
│                                                                       │
│ Also supports: updating polygon points, name, is_active               │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ User deletes a zone
                                ▼
┌───────────────────────────────────────────────────────────────────────┐
│ DELETE /api/v2/cameras/{camera_id}/zones/{zone_id}                    │
│                                                                       │
│ → Deletes zone from DB                                                │
│ → Cascade: rules linked to this zone get zone_id=NULL (SET NULL)      │
│ → After reload-config, CameraWorker stops tracking this zone          │
└───────────────────────────────────────────────────────────────────────┘
```

### Polygon Coordinate System

- **Storage:** Points stored as 0-100 percentage values (resolution-independent).
- **Evaluation:** Converted to pixel coordinates using actual frame dimensions at runtime.
- **Point-in-polygon check:** Uses both `bottom_center` (feet) AND `bbox_center` (body) — so a person standing behind a counter with feet outside the zone but body visible is still detected.

### Zone Type → What It Measures

| Zone Type | What It Measures | Typical Use Case |
|-----------|-----------------|------------------|
| `footfall` | Count people entering the polygon | Entry/exit counting, traffic flow |
| `dwell_time` | How long people stay inside | Engagement analysis, wait time |
| `queue_length` | Number of people simultaneously inside | Queue monitoring |
| `entry_exit` | Directional flow (line crossing) | Entry/exit counting with direction |
| `heatmap` | Positional density heatmap | Store layout optimization |
| `purchase_intent` | Dwell > 15s in billing area | Possible purchase detection |
| `billing_zone` | Billing counter interaction | Billing service time analysis |
| `restricted_zone` | Intrusion detection | Security alerts |
| `medicine_pickup_zone` | Pharmacy pickup point | Apollo pharmacy tracking |

---

### `update_zones()` — Deep Dive with Examples

This is the most important function in `TrackManager` (`app/modules/tracking/track_manager.py`). It's called **every frame, for every active track** to determine which zones a person is currently standing in. Here's the complete line-by-line walkthrough.

#### Full Function Signature

```python
def update_zones(self, track: ActiveTrack, zones_data: List[dict],
                 frame_width: int = 1920, frame_height: int = 1080):
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `track` | `ActiveTrack` | In-memory track representing one person. Has `bbox`, `current_zones`, `zone_enter_times`, `dwell_seconds`. |
| `zones_data` | `List[dict]` | All active zones for this camera. Each zone has `id`, `polygon`, `zone_type`, `name`, `shape`. |
| `frame_width` | `int` | Actual frame width in pixels (e.g., 1920). Used to convert zone % coords → pixels. |
| `frame_height` | `int` | Actual frame height in pixels (e.g., 1080). Used to convert zone % coords → pixels. |

#### Helper Functions Used (from `app/utils/geometry.py`)

```python
def bbox_bottom_center(bbox: dict) -> Tuple[float, float]:
    """Foot position: (center_x, y2) — bottom edge of bbox."""
    cx = (bbox["x1"] + bbox["x2"]) / 2.0
    return (cx, bbox["y2"])

def bbox_center(bbox: dict) -> Tuple[float, float]:
    """Body center: midpoint of bbox."""
    cx = (bbox["x1"] + bbox["x2"]) / 2.0
    cy = (bbox["y1"] + bbox["y2"]) / 2.0
    return (cx, cy)

def point_in_polygon(point, polygon_points) -> bool:
    """Uses shapely to check if a point is inside a polygon."""
    p = Point(point)
    poly = Polygon(polygon_points)
    return poly.contains(p)

def polygon_from_json(polygon_json) -> Optional[List[Tuple]]:
    """Converts DB format {"points": [[x1,y1],...]} to tuple list."""
    points = polygon_json.get("points", [])
    return [(p[0], p[1]) for p in points] if len(points) >= 3 else None
```

#### Line-by-line Breakdown

```
Line 140:    now = utc_now()
```
Get current UTC timestamp. Used to record `zone_enter_times` and calculate `dwell_seconds`.

```
Line 141-142: if not track.bbox: return
```
Guard clause: if the track has no bounding box (shouldn't happen, but safety check), skip entirely.

```
Line 147-148:
    bottom = bbox_bottom_center(track.bbox)   # (cx, y2) — feet
    centre = bbox_center(track.bbox)           # (cx, cy) — body center
```
Calculate the TWO check points for this track. Example:
```
If bbox = {"x1": 400, "y1": 200, "x2": 600, "y2": 900}:
    bottom = (500, 900)   ← feet position
    centre = (500, 550)   ← body center
```

```
Line 150:    current_zone_ids = set()
```
Initialize empty set. This will hold all zone UUID strings the person is currently in for THIS frame.

```
Line 152:    for zone in zones_data:
```
Iterate through every active zone for this camera.

```
Line 153:        zone_id = str(zone["id"])
```
Convert UUID to string — all zone tracking uses string keys internally.

```
Line 154:        poly_points = polygon_from_json(zone.get("polygon"))
```
Extract the polygon points from the zone's `polygon` JSONB field.
```
Example zone from DB:
  zone["polygon"] = {"points": [[20, 30], [60, 30], [60, 70], [20, 70]]}
  → poly_points = [(20, 30), (60, 30), (60, 70), (20, 70)]
```

```
Line 156-157:
    if not poly_points:
        continue
```
Skip zones with no polygon or fewer than 3 points (need at least 3 for a polygon).

```
Line 159-162:  COORDINATE CONVERSION (percentage → pixels)
    pixel_poly = [
        (p[0] * frame_width / 100.0, p[1] * frame_height / 100.0)
        for p in poly_points
    ]
```
Zones are stored as 0-100 percentages (camera/resolution agnostic). Convert to actual pixel coordinates:
```
Example with frame_width=1920, frame_height=1080:
  poly_points = [(20, 30), (60, 30), (60, 70), (20, 70)]
  pixel_poly  = [(384, 324), (1152, 324), (1152, 756), (384, 756)]
  → A rectangle covering center-left portion of the frame
```

```
Line 165:  POINT-IN-POLYGON CHECK (dual-point strategy)
    in_zone = point_in_polygon(bottom, pixel_poly) or point_in_polygon(centre, pixel_poly)
```
This is the KEY design decision. Check BOTH the feet AND the body center. If EITHER is inside the polygon → person is "in" the zone.

**Why check both?** A person standing behind a billing counter might have their **feet** outside the drawn polygon (feet are behind the counter), but their **upper body** is clearly visible inside the billing zone. Using both points ensures the person is correctly detected as "in" the zone.

```
Point check visualization:

Frame (1920 x 1080)
┌──────────────────────────────────────────────┐
│                                              │
│     ┌──────────────────────────┐             │
│     │  ZONE: Billing Counter   │             │
│     │  (polygon 384,324 →      │             │
│     │           1152,756)      │             │
│     │                          │             │
│     │     ┌──────────┐         │             │
│     │     │  PERSON  │         │             │
│     │     │  bbox    │         │             │
│     │     │  400x200 │         │             │
│     │     │  →900    │         │             │
│     │     └────┬─────┘         │             │
│     │     centre=(500,550)  ✓  │  ← INSIDE zone
│     │          │                │
│     │     bottom=(500,900)  ✗   │  ← OUTSIDE zone (behind counter)
│     └──────────┼───────────────┘
│              Counter                         │
└──────────────────────────────────────────────┘

Result: in_zone = True  (centre is inside, even though bottom is outside)
```

```
Line 168-177:  IF in_zone:
    current_zone_ids.add(zone_id)          ← add to current zones set

    # Track zone entry time
    if zone_id not in track.zone_enter_times:
        track.zone_enter_times[zone_id] = now   ← FIRST time seen in this zone
```

When the person is in the zone, add it to the current set. If this is the FIRST time they've been seen in this zone (i.e., `zone_id` is not in `zone_enter_times`), record the entry time. This is used later to calculate dwell duration.

```
Line 177:
    enter_time = track.zone_enter_times[zone_id]
    track.dwell_seconds[zone_id] = seconds_since(enter_time)
```
Every frame the person is still in the zone, update `dwell_seconds` to the elapsed time since they first entered.

```
Line 178-181:  ELSE (NOT in_zone):
    if zone_id in track.zone_enter_times:
        del track.zone_enter_times[zone_id]   ← person LEFT this zone
```
If the person was previously in this zone but is no longer, delete the entry time. This signals "zone exit" to the `ZoneEventDetector` in the next stage.

```
Line 183:  track.current_zones = current_zone_ids
```
Replace the track's zone set with the freshly computed set for this frame.

#### Complete Walkthrough Example

**Setup:**
- Camera resolution: 1920 x 1080
- 2 zones configured:
  - Zone A (id: "aaa", type: "billing_zone"): rectangle at 20-60% x, 30-70% y
  - Zone B (id: "bbb", type: "entry_exit"): rectangle at 60-90% x, 20-80% y
- 1 person detected at bbox: `{x1: 400, y1: 200, x2: 600, y2: 900}`

```
Frame T=0 (person first appears at frame left side):
─────────────────────────────────────────────────────────
Track.bbox = {x1:400, y1:200, x2:600, y2:900}
  → bottom = (500, 900)
  → centre = (500, 550)

Zone A (pixel_poly): [(384,324), (1152,324), (1152,756), (384,756)]
  → point_in_polygon((500,900), zoneA) = True   (bottom inside zone)
  → point_in_polygon((500,550), zoneA) = True   (centre inside zone)
  → in_zone_A = True

Zone B (pixel_poly): [(1152,216), (1728,216), (1728,864), (1152,864)]
  → point_in_polygon((500,900), zoneB) = False
  → point_in_polygon((500,550), zoneB) = False
  → in_zone_B = False

After update_zones:
  track.current_zones = {"aaa"}
  track.zone_enter_times = {"aaa": T=0}
  track.dwell_seconds = {"aaa": 0.0}

ZoneEventDetector (next stage) detects:
  _last_zones[track_id] was {} → now {"aaa"}
  → zone_enter event for zone "aaa"  ✓


Frame T=2s (person still in Zone A):
─────────────────────────────────────────────────────────
Same bbox, same zone membership.

After update_zones:
  track.current_zones = {"aaa"}
  track.zone_enter_times = {"aaa": T=0}     ← unchanged
  track.dwell_seconds = {"aaa": 2.0}        ← updated

ZoneEventDetector:
  _last_zones = {"aaa"}, current = {"aaa"}
  → No event (same zones as before)


Frame T=5s (person walks right, enters Zone B):
─────────────────────────────────────────────────────────
Track.bbox = {x1:1000, y1:200, x2:1300, y2:900}
  → bottom = (1150, 900)
  → centre = (1150, 550)

Zone A (pixel_poly): [(384,324), (1152,324), (1152,756), (384,756)]
  → point_in_polygon((1150,900), zoneA) = False  (bottom ON edge)
  → point_in_polygon((1150,550), zoneA) = True   (centre inside)
  → in_zone_A = True  ← STILL in Zone A (centre is inside!)

Zone B (pixel_poly): [(1152,216), (1728,216), (1728,864), (1152,864)]
  → point_in_polygon((1150,900), zoneB) = False
  → point_in_polygon((1150,550), zoneB) = True   (centre inside)
  → in_zone_B = True  ← NOW in Zone B too!

After update_zones:
  track.current_zones = {"aaa", "bbb"}         ← in BOTH zones
  track.zone_enter_times = {"aaa": T=0, "bbb": T=5}
  track.dwell_seconds = {"aaa": 5.0, "bbb": 0.0}

ZoneEventDetector:
  _last_zones = {"aaa"}, current = {"aaa", "bbb"}
  → zone_enter event for zone "bbb"  ✓ (new zone entered)


Frame T=10s (person fully inside Zone B, left Zone A):
─────────────────────────────────────────────────────────
Track.bbox = {x1:1200, y1:200, x2:1500, y2:900}
  → bottom = (1350, 900)
  → centre = (1350, 550)

Zone A (pixel_poly): [(384,324), (1152,324), (1152,756), (384,756)]
  → point_in_polygon((1350,900), zoneA) = False
  → point_in_polygon((1350,550), zoneA) = False  (centre past x=1152)
  → in_zone_A = False  ← LEFT Zone A

Zone B (pixel_poly): [(1152,216), (1728,216), (1728,864), (1152,864)]
  → point_in_polygon((1350,900), zoneB) = True
  → point_in_polygon((1350,550), zoneB) = True
  → in_zone_B = True   ← Still in Zone B

After update_zones:
  track.current_zones = {"bbb"}
  track.zone_enter_times = {"bbb": T=5}      ← Zone A entry REMOVED
  track.dwell_seconds = {"bbb": 5.0}

ZoneEventDetector:
  _last_zones = {"aaa", "bbb"}, current = {"bbb"}
  → zone_exit event for zone "aaa" (dwell=10.0s)  ✓


Frame T=30s (person still in Zone B):
─────────────────────────────────────────────────────────
After update_zones:
  track.current_zones = {"bbb"}
  track.zone_enter_times = {"bbb": T=5}
  track.dwell_seconds = {"bbb": 25.0}

ZoneEventDetector:
  _last_zones = {"bbb"}, current = {"bbb"}
  → zone_dwell_milestone for zone "bbb" at 30s threshold
  → fires when dwell_seconds reaches exactly 30s


Frame T=40s (person leaves frame entirely):
─────────────────────────────────────────────────────────
Track goes STALE (>5s without detection).
cleanup_stale_tracks() removes this track.

CameraWorker._close_track_session():
  → Fires "person_left_view" event
  → Total visit: 40s in frame, 35s in Zone B

ZoneEventDetector cleanup:
  → Removes track from _last_zones and _fired_milestones
```

#### Key Design Insights

1. **Percentage storage → pixel evaluation.** Zones are stored as 0-100 percentages so they work on any resolution camera. Conversion happens at runtime using the actual frame dimensions.

2. **Dual-point check (feet + body).** Using both `bottom_center` and `bbox_center` makes zone detection robust to occlusion (counters, shelves blocking feet) and gives more accurate "in zone" detection.

3. **Entry time tracking.** `zone_enter_times` stores the FIRST timestamp a person entered each zone. This is used to compute `dwell_seconds` and is the basis for all dwell-related events.

4. **Cleanup on exit.** When a person leaves a zone, `zone_enter_times` for that zone is deleted. This ensures if they re-enter later, a new entry time is recorded (new visit, fresh dwell counter).

5. **Multiple zones simultaneously.** A person can be in MANY zones at once (e.g., a queue zone overlapping with a billing zone). `current_zones` is a set of all active zone memberships.

6. **No DB in this function.** Everything is in-memory. Zone membership is computed fresh each frame. No database queries here — this function runs at 5 FPS for every tracked person.

#### Relationship to Downstream Stages

```
update_zones() output           → Used by...
─────────────────────────────────────────────────────
track.current_zones             → ZoneEventDetector.detect()  (stage 3)
                                 → RuleEvaluator.evaluate()   (stage 4)
                                 → _persist_observations()    (stage 6d)

track.zone_enter_times          → dwell_seconds calculation
                                 → (not persisted directly)

track.dwell_seconds             → zone_dwell_milestone events (stage 3)
                                 → zone_dwell rule (stage 4)
                                 → billing_interaction rule (stage 4)
                                 → possible_purchase rule (stage 4)
```

---

## 5. Hardcoded Events & Rules

### Hardcoded Events (always fire, no configuration needed)

These are embedded directly in CameraWorker's `_create_track_session()` and `_close_track_session()` methods.

| Event | Trigger | Severity | Where Defined |
|-------|---------|----------|---------------|
| `person_entered_view` | New TrackSession created (first detection of a person) | LOW | `camera_worker.py:_create_track_session()` |
| `person_left_view` | TrackSession closed (person left or went stale) | LOW | `camera_worker.py:_close_track_session()` |

`person_entered_view` is initially created with `person_identity_id=None`. When ReID later resolves the identity, the event is updated with the resolved `person_identity_id` and description.

### Automatic Zone Events (ZoneEventDetector)

These fire automatically based on zone polygon intersection, no rule configuration needed.

| Event | Trigger | Metadata |
|-------|---------|----------|
| `zone_enter` | Person's bbox first intersects zone polygon (current_zones - prev_zones ≠ ∅) | `local_track_id` |
| `zone_exit` | Person's bbox leaves zone polygon (prev_zones - current_zones ≠ ∅) | `local_track_id`, `dwell_seconds` |
| `zone_dwell_milestone` | Dwell reaches 30s, 60s, or 120s in a zone | `local_track_id`, `dwell_seconds`, `threshold_reached` |

Key behaviors:
- `zone_exit` fires only once when the person leaves (not every frame).
- `zone_dwell_milestone` fires only once per threshold per track:zone pair. Fires the highest matching threshold first (e.g., if dwell jumps from 0 to 65s, fires 60s milestone, not 30s).
- Memory is cleaned up when tracks go stale.

### Rule-Based Events (RuleEvaluator — 6 hardcoded `rule_type` enums)

These are configurable via the `/api/rules` CRUD API and stored in the `rules` table. Each rule links a zone to a behavior with parameters.

#### 1. `line_crossing`

```
Purpose: Detect when a person crosses a line (2-point line shape zone).

How it works:
  - Uses prev_bbox and curr_bbox (tracked frame-to-frame)
  - Computes bbox_bottom_center for both frames
  - Normalizes to [0-1] range to match stored line coordinates
  - line_crossing_check() determines if the segment between prev and curr
    intersects the line, and in which direction

Event fired: line_crossing_{direction}  (left/right)
Severity: from rule config (default "info")
Metadata: { direction, zone_type }

Required zone: shape="line" with 2 points in polygon
Required config: none (uses zone's line_config derived from polygon)
```

#### 2. `zone_dwell`

```
Purpose: Alert when a person stays in a zone longer than a threshold.

How it works:
  - Checks track.dwell_seconds for the zone
  - If dwell ≥ dwell_threshold_seconds → fire

Event fired: zone_dwell_exceeded
Severity: from rule config (default "warning")
Metadata: { dwell_seconds, threshold }

Config parameters:
  - dwell_threshold_seconds (default: 60)
  - cooldown_seconds (default: 30)
```

#### 3. `billing_interaction`

```
Purpose: Detect when a person interacts with a billing counter.

How it works:
  - Checks if person is currently in the zone
  - If dwell ≥ dwell_threshold_seconds → fire
  - ALSO creates a BillingInteraction row in the billing_interactions table

Event fired: billing_interaction
Severity: "info"
Metadata: { dwell_seconds }

Config parameters:
  - dwell_threshold_seconds (default: 10)
  - cooldown_seconds (default: 30)

Additional DB record:
  INSERT INTO billing_interactions:
  camera_id, person_identity_id, track_session_id, zone_id,
  entered_at, dwell_seconds, interaction_type="billing_counter",
  metadata_json
```

#### 4. `queue_count`

```
Status: PLACEHOLDER — not implemented per-track.

The _eval_queue_count() method returns None.
Queue counting would need to be evaluated at the camera level
(aggregating across all tracks), not per-track.
```

#### 5. `possible_purchase`

```
Purpose: Detect possible purchase when a person dwells in a billing/purchase zone.

How it works:
  - Checks if person is currently in the zone
  - If dwell ≥ dwell_threshold_seconds → fire

Event fired: purchase
Severity: HIGH (hardcoded, not from rule config)
Metadata: { dwell_seconds, local_track_id }

Config parameters:
  - dwell_threshold_seconds (default: 15)
  - cooldown_seconds (default: 30)

Note: This is the only rule type with severity=HIGH.
```

#### 6. `restricted_zone`

```
Purpose: Alert when a person enters a restricted area.

How it works:
  - Checks if person is currently in the zone
  - Fires immediately (no dwell threshold)

Event fired: restricted_zone_intrusion
Severity: "alert" (hardcoded)
Metadata: { zone_type: "restricted_zone" }

Config parameters:
  - cooldown_seconds (default: 30)

Note: No dwell threshold — fires on first detection of entry.
```

### Rule Cooldown Mechanism

Each rule has `cooldown_seconds` (default 30s). When a rule fires for a specific `[rule_id, track_id]` pair, it won't fire again for that pair until the cooldown expires.

Implementation:
```
cooldown_key = f"{rule['id']}:{track.local_track_id}"
cooldown_secs = rule.get("cooldown_seconds", 30)

if is_within_cooldown(cooldown_tracker.get(cooldown_key), cooldown_secs):
    continue  # skip this rule+track combo

cooldown_tracker[cooldown_key] = utc_now()  # record fire time
```

The cooldown tracker is periodically pruned (every 60s) to remove expired entries and prevent memory leaks.

### How Rules are Loaded

```
1. CameraWorker.start()
   → WorkerSupervisor.start_camera(camera_id)
   → load_runtime_config(db, camera_id)
   → config_loader.py:
      - load_camera_config(db, camera_id) → camera dict
      - load_zones_for_camera(db, camera_id) → active zones for this camera
      - load_active_rules(db, camera_id) → enabled rules (camera-scoped + global)
      - zones_by_id = {str(z["id"]): z for z in zones}
   → Returns: { camera, zones, zones_by_id, rules }

2. CameraWorker.apply_runtime_config(runtime_config):
   - self.zones = runtime_config["zones"]
   - self.rule_evaluator.cache.load(rules, zones_by_id)

3. On each frame:
   - rule_evaluator.evaluate() reads from cache (no DB query)
   - cache.get_rules_for_camera(camera_id): filters rules by camera_id

4. Refresh triggered by:
   - POST /api/runtime/reload-config
   - When a zone/rule is created/updated/deleted via API
```

### Event Type Summary

| Event Type | Source | Severity | Configurable |
|-----------|--------|----------|-------------|
| `person_entered_view` | Hardcoded (CameraWorker) | LOW | No |
| `person_left_view` | Hardcoded (CameraWorker) | LOW | No |
| `zone_enter` | ZoneEventDetector (auto) | LOW | No |
| `zone_exit` | ZoneEventDetector (auto) | LOW | No |
| `zone_dwell_milestone` | ZoneEventDetector (auto) | LOW | Thresholds (30/60/120s) |
| `line_crossing_{direction}` | RuleEvaluator (line_crossing) | From rule | Yes |
| `zone_dwell_exceeded` | RuleEvaluator (zone_dwell) | From rule | Yes |
| `billing_interaction` | RuleEvaluator (billing_interaction) | info | Yes |
| `purchase` | RuleEvaluator (possible_purchase) | HIGH | Yes |
| `restricted_zone_intrusion` | RuleEvaluator (restricted_zone) | alert | Yes |

---

## 6. Models Used in the Pipeline

| Stage | Model | Framework | Input | Output | Shared? |
|-------|-------|-----------|-------|--------|---------|
| Detection + Tracking | **YOLOv8n** | Ultralytics | Full frame (BGR) | TrackedDetections: track_id, bbox, confidence, class | Yes (singleton) |
| ReID (Body) | **OSNet** | torchreid | Person crop (from bbox) | 512-dim body embedding | Yes (singleton) |
| Face Analysis | **InsightFace buffalo_l** | InsightFace | Person crop (from bbox) | gender, age, age_group, face_embedding (512-dim), face_score, face_crop | Yes (singleton) |
| Identity Matching | **pgvector cosine** | PostgreSQL | 512-dim embedding | Best match PersonIdentity + confidence score | N/A (DB query) |
| Crop Quality Filter | **Custom** | OpenCV + numpy | Person crop | quality_score (0-1): blur, size, aspect ratio | N/A (utility) |

### Model Details

**YOLOv8n (Detection + Tracking)**
- File: `app/modules/detection/yolo_detector.py`
- Model: `yolov8n.pt` (nano variant, lightest)
- Method: `model.track(frame, persist=True)` — combines detection + ByteTrack tracking in one forward pass
- Confidence threshold: Configurable via `YOLO_CONFIDENCE_THRESHOLD`
- Allowed classes: Person only (class_id=0, filtered from COCO 80 classes)
- GPU: Runs on CUDA if available, falls back to CPU
- Shared: `get_shared_detector()` returns a singleton

**OSNet (Body Re-Identification)**
- File: `app/modules/reid/osnet_extractor.py`
- Model: OSNet (torchreid)
- Input: Person crop image (resized to model input size)
- Output: 512-dim normalized embedding vector
- Accumulation: 5 frames collected before mean pooling + decision
- Quality gate: `assess_crop_quality()` rejects blurry/small crops before extraction
- Shared: `get_shared_extractor()` returns a singleton

**InsightFace buffalo_l (Face Analysis + Demographics)**
- File: `app/modules/reid/insightface_analyzer.py`
- Model: buffalo_l (InsightFace model zoo)
- Input: Person crop image
- Output: Face detection + alignment + 512-dim face embedding + demographics (gender, age, age_group)
- Only runs when `demographic_enabled=True` on the camera
- Face score: Confidence of face detection (used for quality ranking)
- Best face is kept per track (highest face_score)
- Shared: `get_shared_analyzer()` returns a singleton

**IdentityDecisionEngine (pgvector Matching)**
- File: `app/modules/reid/identity_decision_engine.py`
- Method: Cosine similarity search on `person_embeddings` (body) and `person_face_embeddings` (face)
- Uses pgvector's `<->` operator for efficient cosine distance
- Decision logic:
  - If match score > threshold → existing person (re-sighting)
  - If no match → creates new PersonIdentity (temporary, promoted on re-sighting)
  - Temporary IDs can be pruned if never re-sighted
  - Face embeddings provide additional confidence for identity matching

**InferencePool**
- File: `app/modules/ai_runtime/inference_pool.py`
- Purpose: Runs model inference in a ThreadPoolExecutor to avoid blocking the asyncio event loop
- Usage: `await run_inference(model_fn, *args)` → runs the model callable in a thread
- All models (YOLO, OSNet, InsightFace) run through this pool

### Deployment Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│ DOCKER COMPOSE                                                         │
│                                                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │
│  │ FastAPI App      │  │ PostgreSQL      │  │ MediaMTX            │   │
│  │ (retail-ai)      │  │ + pgvector      │  │ (RTSP→WebRTC/HLS)   │   │
│  │                  │  │                 │  │                     │   │
│  │ Port: 8000       │  │ Port: 5432      │  │ Port: 8889 (API)    │   │
│  │                  │  │                 │  │ Port: 8554 (RTSP)   │   │
│  │ - REST API       │  │ - All data      │  │ Port: 8888 (HLS)    │   │
│  │ - Camera Workers │  │ - pgvector idx  │  │ Port: 8189 (WHEP)   │   │
│  │ - Stream Manager │  │ - JSONB columns │  │                     │   │
│  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘   │
│           │                    │                       │               │
│  ┌────────┴────────┐           │                       │               │
│  │ MinIO            │           │                       │               │
│  │ (S3-compatible)  │           │                       │               │
│  │                  │           │                       │               │
│  │ Port: 9000 (API) │           │                       │               │
│  │ Port: 9001 (UI)  │           │                       │               │
│  │                  │           │                       │               │
│  │ - Crop images    │           │                       │               │
│  │ - Face crops     │           │                       │               │
│  │ - Snapshots      │           │                       │               │
│  └──────────────────┘           │                       │               │
│                                 │                       │               │
│  ┌──────────────────────────────┴───────────────────────┴───────────┐ │
│  │ External RTSP Cameras / NVR                                       │ │
│  │ rtsp://camera-ip:554/stream                                       │ │
│  │ → Pulled by LatestFrameBuffer (cv2.VideoCapture)                  │ │
│  │ → Republished by MediaMTX (WebRTC/HLS) for browser playback       │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
```

### Stream Flow

```
RTSP Camera
    │
    ├──→ LatestFrameBuffer (cv2.VideoCapture, background thread)
    │       │
    │       └──→ CameraWorker._process_frame() → AI pipeline
    │               │
    │               └──→ StreamBroadcaster (if burnin_enabled)
    │                       │
    │                       └──→ ffmpeg pipe → MediaMTX → WebRTC/HLS
    │
    └──→ StreamManager (if NOT burnin_enabled)
            │
            └──→ ffmpeg RTSP pull → republish → MediaMTX → WebRTC/HLS
```

When `burnin_enabled=True` (default), the StreamBroadcaster draws bounding boxes and zone overlays on the annotated frames before pushing to MediaMTX. The viewer sees the AI-processed feed with annotations.

When `burnin_enabled=False`, StreamManager pulls the raw RTSP stream and republishes it to MediaMTX without annotations. The viewer sees the raw feed.