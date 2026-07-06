# Camera Worker Architecture & Event Lifecycle

This document explains the internal mechanics of the Retail AI Platform's Camera Worker. The worker processes real-time RTSP video streams to detect, track, identify (via face/body), and generate analytical events for retail environments.

> **Last Updated:** July 6, 2026 — Face-first identity, multi-face storage, per-camera YOLO isolation, inference pool fix.

---

## Pipeline Data Flow

The following diagram illustrates the frame-by-frame data flow within a single camera worker.

```text
+----------------+
| RTSP Stream    |
+-------+--------+
        |
        v
+-------+--------+
| YOLO11 Detect  |
| + ByteTrack    |
+-------+--------+
        | Bounding Boxes + track_id
        v
+-------+--------+
| Track Manager  | ---> [Zone Event Detector] ---> Generate Zone Events
+-------+--------+
        |
        | bbox height >= 100px, should_run_reid()
        v
+-------+--------+
| ReID Pipeline  |
| (5-frame accum |
|  windows)      |
+-------+--------+
        |
        | YOLO-Pose keypoints → Crop Quality → OSNet body emb
        | InsightFace → demographics + face emb + frontality check
        v
+-------+--------+
| Face-First      |
| Deferred        | ---> good_face_count < FACE_IDENTITY_MIN_DETECTIONS?
| Resolution Gate |      → YES: skip decision, keep accumulating
+-------+--------+      → NO (+ face >= quality threshold): proceed
        |
        | Enough good faces OR existing identity
        v
+-------+--------+
| Identity Engine | ---> Face search (multi-face, up to 5 per person)
+-------+--------+ ---> Face contradiction gate on body candidates
        |               ---> Body search (48h window, demoted confidence)
        v
+-------+--------+
| PostgreSQL DB   |
+----------------+
```

---

## How the Worker Works

### 1. Frame Ingestion & YOLO Tracking
The worker runs asynchronously at a target FPS (e.g., 5 FPS). It retrieves the absolute latest frame from the RTSP stream using a dedicated background thread buffer. The frame is passed into the camera's **own YOLO model instance** (not shared), coupled with ByteTrack (`persist=True`), to generate temporally consistent bounding boxes and `local_track_ids`.

> **CRITICAL:** Each camera MUST have its own YOLO model instance. ByteTrack's `persist=True` maintains internal state (Kalman filters, track ID assignments) that is NOT thread-safe. Sharing one model across cameras corrupts track IDs, causing massive track churn (new IDs every few seconds, `total_frames=1` per track). See `CRITICAL_FINDINGS.md` for details.

### 2. Track Management
Detected persons are managed via `TrackManager` which keeps in-memory state per track: bounding box history, zone membership, dwell times, confidence, stability score. All persons are tracked regardless of zone configuration — zones are only required for zone-based rules and billing analytics.

### 3. Face-First Deferred ReID Pipeline

This is the core identity resolution pipeline, redesigned July 2026 to require clear face evidence before creating a `PersonIdentity`.

#### 3.1 Per-Frame Processing (5-frame accumulation windows)
When a track is eligible (`bbox_height >= 100px`, not throttled), the system:
1. **Crops** the person from the frame
2. **YOLO-Pose** checks torso keypoint visibility (>= 50% required)
3. **Crop quality** assessment (weighted: keypoints 35%, YOLO conf 25%, sharpness 15%, size 10%, aspect 10%, brightness 5%)
4. **InsightFace (buffalo_l)** runs on EVERY crop regardless of body quality:
   - Face detection + frontality filtering (score >= 0.50, width >= 30px)
   - Age, gender, age_group extraction
   - 512-dim face embedding (ArcFace)
5. **OSNet** extracts 512-dim body embedding (only if crop quality >= 0.30)

Each frame's results are accumulated into a 5-frame window.

#### 3.2 Face Quality Tracking (across all windows)
Within each accumulation window:
- `good_face_count` tracks how many frames had a frontal face with `face_score >= FACE_IDENTITY_MIN_SCORE` (0.60)
- The best face embedding, best face score, and best face crop path are preserved across ALL windows on the track
- The best body embedding is also preserved

This means even if face-quality frames span multiple 5-frame windows, they all contribute.

#### 3.3 Deferred Resolution Gate
After each 5-frame window:
- **If track has no identity AND `good_face_count < FACE_IDENTITY_MIN_DETECTIONS` (2):**
  → **Defer.** Skip `decide_identity()`. Clear the window but keep accumulating in the next window.
- **If track has an identity OR `good_face_count >= 2`:**
  → **Proceed.** Call `decide_identity()` with the best face and body from ALL windows.

#### 3.4 Final Resolution on Track Close
When a track goes stale (>5s unseen), `_close_track_session()` performs a final identity resolution attempt if the track accumulated enough good faces but hadn't yet triggered a window decision. If still no identity after the final attempt, the track remains anonymous (no `PersonIdentity` created).

---

## Identity Decision Engine

### Matching Priority (highest to lowest)

1. **Face matching (highest priority)**
   - Searches ALL face embeddings per person (up to `MAX_FACE_EMBEDDINGS_PER_PERSON = 5`)
   - Multi-angle matching: best per-person similarity aggregated from all stored faces
   - Match threshold: `FACE_MATCH_THRESHOLD` (0.48)
   - If face matches → identity resolved, body ReID bypassed

2. **Body ReID matching (fallback, demoted)**
   - Searches `person_embeddings` within 48-hour window
   - Face contradiction gate: candidates whose stored faces don't match the track's face are excluded
   - Match threshold: `REID_MATCH_THRESHOLD` (0.80)
   - **Body-only matches are never marked as confident** (`reid_confident = False` unless face confirmed)

3. **New identity creation (gated)**
   - Requires ALL of:
     - `REQUIRE_FACE_FOR_IDENTITY = True` (face embedding present)
     - `face_score >= FACE_IDENTITY_MIN_SCORE` (0.60)
     - `good_face_count >= FACE_IDENTITY_MIN_DETECTIONS` (2+ good face detections)

### Multi-Face Embedding Storage

Previously only 1 face embedding was stored per person (the best one, overwritten). Now:

- Up to 5 face embeddings stored per `PersonIdentity`
- New faces are **appended** (not replacing)
- **Deduplication:** Before appending a face to the track's `face_embedding_list`, cosine similarity is compared against all existing entries. If > 0.95 with any existing face → skipped (same crop/angle from consecutive windows)
- When cap exceeded, lowest-quality faces are pruned
- After identity resolution, all accumulated faces are stored EXCEPT the one already stored by `decide_identity` (similarity > 0.95 to best → skipped to avoid duplicates)
- Face similarity search aggregates all stored faces per person for multi-angle robustness
- Contradiction checking uses the BEST match across all stored faces of a candidate
- IVFFlat index on `person_face_embeddings.embedding` for fast pgvector cosine search

### Face Contradiction Gate (Body ReID Safety)

When body ReID finds a candidate match, the system checks ALL stored face embeddings of that candidate against the track's face:
- If ANY stored face matches the track's face well (cosine similarity >= 0.48) → candidate is accepted
- If NO stored face matches → candidate is **excluded** (prevents clothing-based false merges)

---

## Event Generation Lifecycle

### 1. `person_entered_view`
- **When:** Frame 1 (exact moment a new `local_track_id` is created)
- **How:** Fired immediately with `person_identity_id = null`
- **Refinement:** Once ReID resolves (either during a window or on track close), an `UPDATE` populates the `person_identity_id`

### 2. `new_person_registered`
- **When:** Identity decision creates a new `PersonIdentity`
- **Gating:** Requires face quality >= 0.60 AND 2+ good face detections
- **Data:** `face_score` and `crop_quality_score` stored in `metadata_json`

### 3. `zone_enter` & `zone_exit`
- Continuously evaluated on every frame by `ZoneEventDetector`
- Both bottom-center (feet) and bbox-center checked for zone membership

### 4. `person_left_view`
- When track goes stale (>5s unseen)
- Includes total frame count and duration in `metadata_json`

---

## Configuration Reference

| Setting | Value | Description |
|---------|-------|-------------|
| `REQUIRE_FACE_FOR_IDENTITY` | `True` | Must have face to create identity |
| `FACE_IDENTITY_MIN_SCORE` | `0.60` | Minimum face quality for identity creation |
| `FACE_IDENTITY_MIN_DETECTIONS` | `2` | Minimum good face detections across track |
| `MAX_FACE_EMBEDDINGS_PER_PERSON` | `5` | Multi-angle face embeddings stored |
| `FACE_MATCH_THRESHOLD` | `0.48` | Face cosine similarity threshold |
| `REID_MATCH_THRESHOLD` | `0.80` | Body cosine similarity threshold |
| `BODY_ONLY_CONFIDENCE_LIMIT` | `0.95` | Body-only match confidence ceiling |
| `MAX_WORKERS` | `10` (default), `12` in `.env` | Inference pool threads (6 per camera × 2) |
| `REID_ACCUMULATION_FRAMES` | `5` | Frames per accumulation window |
| `REID_CROP_QUALITY_THRESHOLD` | `0.30` | Minimum crop quality for OSNet |
| `FACE_MIN_DET_SCORE` | `0.50` | Minimum face detection score |
| `FACE_MIN_SIZE_PX` | `30` | Minimum face width in pixels |
| `FACE_SEARCH_THRESHOLD` | `0.65` | Face quality to skip body search |

---

## Identity Resolution Decision Tree

```
Track starts (local_track_id assigned)
    ↓
Frame processing (per-frame, 5-frame windows)
    ↓
    ┌─ InsightFace finds frontal face with score >= 0.60?
    │   → YES: good_face_count++
    │   → NO:  continue accumulating
    │
    ├─ 5-frame window complete, good_face_count >= 2?
    │   → NO (+ no existing identity): DEFER. Continue next window.
    │   → YES: Run decide_identity()
    │       │
    │       ├─ Face match in DB (cosine >= 0.48)?
    │       │   → Assign existing PersonIdentity. reid_confident = True.
    │       │
    │       ├─ No face match, body match in DB (cosine >= 0.80)?
    │       │   → Assign existing PersonIdentity. reid_confident = False (body-only).
    │       │
    │       └─ No match at all?
    │           → Create NEW PersonIdentity (gated on face quality).
    │
    ├─ Track stale (>5s unseen)?
    │   → NO:  Continue processing frames.
    │   → YES: _close_track_session()
    │       ├─ Has accumulated good faces but no identity?
    │       │   → Final resolve attempt with best accumulated data.
    │       └─ Close session. Fire person_left_view.
```

---

## Database Tables Involved

| Table | Purpose |
|-------|---------|
| `track_sessions` | Per-person tracking session (entry/exit) |
| `track_observations` | Per-frame bbox position snapshots (every 2s) |
| `person_identities` | Unique identity (created only with face evidence) |
| `person_embeddings` | Body ReID vectors (OSNet, 512-dim, up to 10 per person, 48h window) |
| `person_face_embeddings` | Face recognition vectors (InsightFace/ArcFace, 512-dim, up to 5 per person, no time limit) |
| `events` | Lifecycle events (entered_view, left_view, new_person_registered, zone_enter/exit) |
| `billing_interactions` | Checkout counter dwell/purchase events |
