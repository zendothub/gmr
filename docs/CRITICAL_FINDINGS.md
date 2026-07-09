# Critical Findings & Troubleshooting

> **Created:** July 6, 2026  
> **Last updated:** July 7, 2026  
> **Purpose:** Documents critical bugs discovered and fixed during the ReID/tracking overhaul. Read this BEFORE modifying camera worker, inference pool, YOLO detector, or identity decision engine code.

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
The lock prevents race-condition duplicates at the exact same timestamp. It **cannot** prevent duplicates caused by face similarity below threshold — both cameras can see each other's registered person but still fail to match due to cross-angle appearance difference. See **Finding #7** for the complete analysis.

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

## 7. Massive Duplicate Person Registration (Cross-Camera)

**Status:** FIXED (July 7, 2026)  
**Full analysis:** See `docs/DUPLICATE_IDENTITY_ROOT_CAUSE.md`

### Problem
After 24 hours with 2 cameras: **885 person identities** created for what should be ~100–250 real visitors. **1,554 confirmed duplicate pairs** (same person registered by each camera as a different identity). Data unusable.

### Root Causes (brief)
1. `FACE_MATCH_THRESHOLD = 0.48` used as BOTH the positive match gate AND the contradiction/disassociation gate. Same-person cross-angle similarity of 0.40–0.47 simultaneously failed matching AND triggered disassociation.
2. Body ReID face-exclusion gate also used 0.48 — silently skipped valid body candidates for the same cross-angle reason.
3. `skip_body_reid` logic abandoned body fallback when high-quality face failed to match — creating duplicates when body ReID would have succeeded.
4. Once both cameras confidently confirmed their own identities, no merge ever happened (temporary-only merge mechanism).

### Fixes Summary
- `FACE_CONTRADICTION_THRESHOLD = 0.25` — disassociation only fires for truly different faces
- `FACE_BODY_EXCLUSION_THRESHOLD = 0.30` — body gate allows cross-angle same-person candidates through
- Removed `skip_body_reid` — body ReID always runs as fallback when face search fails
- `deduplicate_persons()` job every 10 minutes — catches any duplicates that slip through

### Data Reset Required
The pre-fix data is corrupt. Run:
```bash
PYTHONPATH=. venv/bin/python danger/reset_tracking_data.py --yes
```

---

## 8. MinIO Crop Leaks & Debug View 404s

**Status:** FIXED (July 7, 2026)  
**Full analysis:** See `docs/CROP_LIFECYCLE_AND_MINIO_FIXES.md`

### Problem
- Debug panel showing 404 on face/body crop URLs in `PersonFaceEmbedding` records
- MinIO growing at **~1 GB+/hour** per camera from undeleted `curr_face_*` files
- Body crops from quality-rejected frames accumulating indefinitely
- Partial ReID windows on stale tracks leaving orphaned crops

### Root Causes (brief)
1. `face_embedding_list` accumulated face crop paths **before** the window cleanup deleted them → DB stored deleted paths
2. `curr_face_*` files uploaded every frame, old file overwritten in memory but never deleted from MinIO  
3. Body crops from non-accumulated frames never added to cleanup scope
4. `current_crop_path` briefly pointed to deleted file after each window cleanup
5. Partial window crops discarded from memory without MinIO cleanup on stale track eviction

### Fixes Summary
- `_minio_cleanup()` helper method on `CameraWorker` (replaces inline redefined function)
- Window cleanup guards `face_embedding_list` paths from deletion
- Delete-before-overwrite for `current_face_crop_path` and `current_crop_path`
- `current_face_crop_path` deleted on stale track eviction
- Partial window crops cleaned up on stale track eviction with protected-paths exclusion

---

## 9. Face Frontality: Profile Faces Stored as "Best" Identity Face

**Status:** FIXED (July 7, 2026)

### Problem
`assess_face_quality()` returned only `face_score * 1.1` — a negligible 10% boost for any face with both eyes detected. Profile-angle faces with high InsightFace detection confidence (0.80+) were stored as the canonical face for an identity, degrading cross-camera matching because profile ArcFace embeddings have low cosine similarity to frontal embeddings of the same person.

The `FACE_MIN_EYE_SPREAD` gate was disabled to avoid rejecting "too many" faces, which had the unintended effect of allowing profiles through.

### Fix
`assess_face_quality()` now computes a weighted combination of:
- Detection score (65% weight when `FACE_FRONTALITY_WEIGHT = 0.35`)
- Geometric frontality (35%): eye spread + nose centering + eye vertical symmetry

Profile face example: `det=0.80, frontality=0.05 → quality=0.54`. Frontal face example: `det=0.80, frontality=0.95 → quality=0.85`. The frontal face always wins.

`FACE_MIN_EYE_SPREAD = 0.25` is re-enabled. Faces with normalised eye spread < 0.25 (roughly 60°+ from frontal) are now rejected entirely from the ReID pipeline.

**Config:** `FACE_FRONTALITY_WEIGHT = 0.35`, `FACE_MIN_EYE_SPREAD = 0.25`

---

## Diagnostic Scripts

### Check for duplicate identities (fast, index-assisted)
```bash
cd /gmr/gmr
PYTHONPATH=. venv/bin/python danger/dedup_faces.py [--threshold 0.48]
```
Uses pgvector LATERAL query — completes in seconds regardless of DB size.

### Reset all tracking data
```bash
cd /gmr/gmr
PYTHONPATH=. venv/bin/python danger/reset_tracking_data.py [--yes]
```
Wipes all tracking, person identity, embeddings, events, analytics, and MinIO crops. Preserves config (cameras, zones, rules, stores, users).

---

## Configuration Quick Reference

| Setting | Value | Notes |
|---------|-------|-------|
| `MAX_WORKERS` | `12` | Inference pool threads (6 per camera × 2) |
| `FACE_MATCH_THRESHOLD` | `0.48` | Positive face match threshold |
| `FACE_CONTRADICTION_THRESHOLD` | `0.25` | **New** — disassociation gate (separate from match) |
| `FACE_BODY_EXCLUSION_THRESHOLD` | `0.30` | **New** — body candidate face gate |
| `FACE_MIN_EYE_SPREAD` | `0.25` | Re-enabled frontal gate |
| `FACE_FRONTALITY_WEIGHT` | `0.35` | **New** — frontality contribution to face_quality |
| `FACE_IDENTITY_MIN_SCORE` | `0.60` | Min face_quality (now includes frontality) for identity creation |
| `FACE_IDENTITY_MIN_DETECTIONS` | `2` | Min good face detections per track |
| `MAX_FACE_EMBEDDINGS_PER_PERSON` | `5` | Multi-angle face storage cap per identity |
| `REID_MATCH_THRESHOLD` | `0.80` | Body ReID match threshold |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.45` | Person detection confidence |
| Stale track timeout | `5.0s` | `track_manager.py` — track removed if unseen |
| Dedup job interval | `10 min` | `jobs/scheduler.py` — periodic identity merger |

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

## 10. Face Contamination — Close Persons + Hair Hallucination

**Status:** FIXED (July 7, 2026)

### Problem A — Two persons standing close together

When two people stand shoulder-to-shoulder in the same camera frame, YOLO bounding boxes can overlap or be adjacent. InsightFace runs on the BODY CROP (not the full frame). If Person B's body crop partially includes Person A's face, InsightFace detects Person A's face inside Person B's crop. The old face-selection heuristic ("face closest to crop centreline wins") could pick the WRONG face.

**Evidence:** Persons `5d1d4b64` and `bd00ae0c` overlapped for 2.8 minutes on the same camera. Cross-face similarity = 0.68. Person B's `PersonIdentity.face_crop` showed Person A's face.

### Fix — Multi-signal face selection (Layer 1)

Replaced the centreline-only heuristic in `insightface_analyzer.py` with a weighted scoring function:

```python
score = det_score × (0.40×size_ratio + 0.35×centre_proximity + 0.25×upper_position)
```

**Signals:**
- **Face size** (0.40 weight): larger face relative to crop → more likely the tracked person's own face
- **Centreline proximity** (0.35 weight): closer to horizontal centre → more likely correct
- **Upper position** (0.25 weight): face should be in upper 40% of body crop

A small face near the crop edge (adjacent-person contamination) scores low and is deselected.

### Fix — Running-consensus contamination gate (Layer 2)

Added in `_run_reid` in `camera_worker.py`. After a track accumulates ≥2 prior good faces, every new face is validated against the running consensus:

```
max_sim = max(cosine_sim(new_face, prior_face) for prior_face in face_embedding_list)

if max_sim < FACE_CONTAMINATION_THRESHOLD (0.35):
    REJECT — this face doesn't belong to the person we've been tracking
    → skip demographics, skip identity accumulation for this frame
```

**Threshold rationale:** Same-person cross-angle: 0.40–0.70+. Different person: 0.10–0.30. Threshold 0.35 sits safely between the two distributions.

### Problem B — Hair / back-of-head detected as a face (hallucination)

InsightFace's SCRFD detector occasionally hallucinates faces on hair textures, accessories, or background objects. In these cases the raw `det_score` can be misleadingly high (0.70+) and hallucinated keypoints can pass frontality checks.

**Evidence:** Person `22be2ccb` had a back-of-head shot stored as `PersonIdentity.face_crop`. Only 1 face embedding from a 15-second, 35-frame track.

### Fix — KPS geometry validation

Added as the **very first gate** in the face validation chain, BEFORE det_score / width / eye_spread:

```
If kps has >= 5 points:
    lx, ly = kps[0] (left eye), rx, ry = kps[1] (right eye), nx = kps[2] (nose)
    
    Reject if:
    1. lx >= rx                    → eyes swapped? (impossible for real face)
    2. |ry - ly| / face_h > 0.20   → eyes not level (hallucinated)
    3. |nx - eye_mid| / eye_sep > 0.40 → nose not between eyes (hallucinated)
    
    face_frontal = False → face rejected
```

Real faces always satisfy all three invariants. Hallucinated landmarks from hair/objects fail at least one.

**Config:** New `FACE_CONTAMINATION_THRESHOLD = 0.35` (in `config.py`).

---

## 11. Gender Voting — Continuous Across Non-Frontal Frames

**Status:** FIXED (July 7, 2026)

### Problem

Gender was voted ONLY on perfectly frontal frames (`eye_spread >= 0.25`). A person at a 3/4 angle for most of their 14-second track would have only 2 gender votes from 41 total frames. If InsightFace misclassified both, the wrong gender was permanently stored.

Additionally, `FACE_IDENTITY_MIN_DETECTIONS = 2` meant identity creation happened as soon as 2 faces were detected — even if both came from the same second at the same angle. No recovery was possible if both votes were wrong.

**Evidence:** Person `c9f6d865` — 41 frames, 14 seconds, only 2 good frontal faces (both in 1 second). Both InsightFace calls returned "M" for a person the operator confirmed is female. Gender in DB: M.

### Fix A — Gender voting moved outside frontality gate

Gender voting now runs for **every detected face**, not just frontal ones:

```
if face_result is not None:                       ← 20+ frames now
    track.gender_votes[face_result.gender] += 1    ← votes counted
    _majority_gender = max(M_votes, F_votes)       ← majority computed

if face_frontal:                                   ← still only 2 frames
    face_embedding + demographics update             ← embedding quality preserved
```

InsightFace returns reliable gender even from non-frontal faces. Counting all detected faces gives the majority vote 10× more data, making it robust against individual misclassifications.

`best_demographics["gender"]` is also continuously updated from `_majority_gender` even on non-frontal frames — gender flips mid-track if voting changes.

### Fix B — Rely on face_quality (including frontality) for best_demographics ranking

The `best_demographics` record (which becomes `PersonIdentity.face_crop_path`) still uses `face_quality` (det_score × frontality) to select the best face image. This ensures the displayed face crop is always the highest-quality FRONTAL shot, while gender is determined by majority vote across ALL detected faces.

### Fix C — ActiveTrack.gender_votes field

Added `gender_votes: dict = field(default_factory=lambda: {"M": 0, "F": 0})` to `ActiveTrack` in `track_manager.py`. Votes persist across the track's entire lifetime. When the track ends, `_close_track_session` writes the majority gender to `PersonIdentity.gender`.

---

## 12. SigLIP2 Zero-Shot Gender (Replaces MiVOLO / InsightFace)

**Status:** FIXED (July 8, 2026)  
**Files changed:** `reid/siglip2_analyzer.py`, `config.py`, `camera_worker.py`

### Problem
MiVOLO's ViT-Small gender accuracy was ~11% on retail CCTV face crops (the model was trained on IMDB celebrity faces — high resolution, studio lighting, front-facing). InsightFace's built-in gender classifier was also ~85-90% with systematic misclassification on certain faces. DeepFace scored 0%. None met the 90%+ accuracy requirement.

### Fix — SigLIP2 Zero-Shot Classification
Google's SigLIP2 (siglip2-base-patch16-224) achieves 100% gender accuracy on clean retail CCTV face crops by comparing image embeddings to pre-computed text prompt embeddings.

**Architecture:**
```
Startup: encode 7 female + 7 male text prompts → cache embeddings
Runtime: encode face/body crop → compute cosine sim to cached text embs → best match wins
```

**Prompts (7+7), pre-computed at startup:**
- Female: "a photo of a woman", "a woman", "a female person, woman", "a woman shopping", "a woman's face", "a female customer", "a woman, female, lady"
- Male: parallel set

**Body crop integration:** Body crops carry clothing context (saree, kurta, uniform) that face crops miss. `analyze_with_body()` combines face + body votes with body weighted 3×. A woman in a saree that face-only missed was correctly identified at 79% confidence via body crop.

**Performance:** 18 ms/image, 1.4 GB GPU, 55 images/sec throughput. MiVOLO kept for age prediction only.

---

## 13. Full-Frame Face Detection + Padded Resize

**Status:** FIXED (July 8, 2026)  
**Files changed:** `insightface_analyzer.py`, `camera_worker.py`, `image_utils.py`

### Problem
Previously InsightFace ran per-track on body crops (200-600px). Face crops were extracted from the body crop at 30-80px, then stretched to 224² for models. The two-level cropping degraded face quality dramatically.

### Fix
InsightFace now runs **once per frame** on the full 2880×1620 frame — single GPU kernel launch instead of 5-10 per-track calls. All faces are detected at native resolution. `_match_face_to_track()` assigns faces to body tracks by face-centre-in-body-bbox membership with a multi-signal scoring heuristic (size × centre-proximity × detection-confidence).

Face crops are extracted from the **full frame** at native resolution (100-400px). `resize_pad_square()` preserves aspect ratio with edge-replicate padding — no stretching or distortion.

**Result:** Face crops 3-7× higher resolution for MiVOLO/SigLIP2 input. Single GPU kernel launch instead of N×. Matched-face exclusivity prevents double-assignment.

---

## 14. Body ReID Dedup + Consensus Gate + Missing-Weights Root Cause

**Status:** FIXED (July 8, 2026 — consensus gate), **ROOT CAUSE FOUND** (July 9, 2026 — missing MSMT17 weights)  
**Files changed:** `identity_decision_engine.py`, `config.py`, `osnet_extractor.py`

### Original Problem (July 8)
OSNet body embedding similarity was non-discriminative — same-person and different-person distributions overlapped heavily. `_search_similar` returned raw top-5 embeddings without deduplicating by person — a single person with 10 stored body embeddings could monopolize all 5 candidates.

### Root Cause — Missing ReID Weights (July 9)
The configured `OSNET_MODEL_PATH = "models/osnet_x1_0.pth"` **did not exist on disk**. `FeatureExtractor` silently fell back to `pretrained=True`, loading ONLY the ImageNet backbone (`osnet_x1_0_imagenet.pth` from `~/.cache/torch/`). The ReID `fc` embedding head (512-dim) stayed **randomly-initialized**. All stored body embeddings were non-discriminative — they encoded scene appearance (pharmacy background, lighting, color histograms) rather than person identity. This is why different-person body sims (median 0.763) **exceeded** same-person sims (median 0.730) — different people in the same pharmacy scene looked MORE similar than the same person across cameras.

### Fix

**Layer 0: Download MSMT17-trained OSNet checkpoint (July 9)**
Replaced missing `models/osnet_x1_0.pth` with `osnet_x1_0_msmt17_combineall` checkpoint (17.3 MB, from HuggingFace `kaiyangzhou/osnet`). Startup guard added in `osnet_extractor.py:_load_model` that refuses to start if the model file is missing AND verifies `fc.*` keys loaded from the checkpoint via `_verify_reid_weights_loaded()`. 292 existing `person_embeddings` recomputed from MinIO crops + IVFFlat index rebuilt.

**New distributions (MSMT17 weights, 32 multi-camera persons + 60 concurrent pairs):**
```
SAME-person: n=88   median=0.680  p10=0.393  p90=0.845
DIFF-person: n=145  median=0.386  p10=0.294  p90=0.534
```
Same-person median (0.680) >> diff-person median (0.386). Best F1 threshold=0.49 (F1=0.793).

**Thresholds retuned:**
- `REID_MATCH_THRESHOLD`: 0.85 → **0.50** (was calibrated against broken ImageNet-backbone weights)
- `BODY_CONTAMINATION_THRESHOLD`: 0.60 → **0.50** (was too close to same-person p25=0.537, rejected valid cross-angle embeddings)

**Layer 1: Person-identity deduplication in `_search_similar`**
Mirrors the face search behavior: fetches 25 raw embeddings, keeps only the best match per unique person_identity_id, returns top-K unique identities. Prevents one person from dominating candidates.

**Layer 2: Body consensus gate (2 of top-3 must agree)**
A single body match at 0.50 can still be a false positive. Requires at least 2 of the top-3 unique-identity candidates to agree on the same person AND exceed REID_MATCH_THRESHOLD before accepting a body ReID merge. With the new MSMT17 weights, 2-of-3 false positives at 0.50 is exponentially unlikely (diff-person p90=0.534, so the probability of 2 wrong persons both exceeding 0.50 is low).

**Layer 3: Body contamination gate (store-time, 0.50 median)**
When storing a new body embedding, if the median cosine similarity to the existing cluster (≥3 embeddings) is below 0.50, reject it. At 0.50, same-person embeddings (median 0.680) are kept, diff-person embeddings (median 0.386) are rejected. Clean separation with good margin on both sides. The dedup job's iterative median-based outlier removal also uses this threshold.

---

## 15. Staff Detection + Purchase Dedup

**Status:** FIXED (July 8, 2026)  
**Files changed:** `config.py`, `jobs/tasks.py`, `analytics/service.py`, `camera_worker.py`, `person.py` (Alembic migration)

### Problem
Employees/staff generate hundreds of billing interaction events per shift while standing in billing zones. Purchase analytics counted raw `COUNT(billing_interactions.id)` with no staff exclusion and no per-person deduplication — inflating purchase counts 10-100×.

### Fix

**Staff auto-classification** (runs every 10 min in dedup job):
- `PersonIdentity.is_staff` boolean column with index
- Two configurable signals: `STAFF_DURATION_THRESHOLD_SECONDS=1800` (total visible time >30 min) OR `STAFF_DISTINCT_DAYS_THRESHOLD=3` (appeared on 3+ distinct days)
- Person promoted to staff when either signal fires; demoted only if BOTH fall below threshold
- Configurable via `.env` / `config.py`

**Purchase query fixes** (5 sites in `analytics/service.py`):
- `COUNT(DISTINCT person_identity_id)` instead of `COUNT(id)` — 1 person = 1 purchase
- `WHERE NOT is_staff` — staff excluded from all purchase analytics
- `_STAFF_IDS` shared subquery for consistency across V1/V2 dashboards

**One-per-track-session guard** (`_camera_worker.py _persist_events`):
- Only one `BillingInteraction` per `track_session_id` + `zone_id` combo
- Prevents cooldown resets from creating duplicate billing rows

---

## 16. Dedup Threshold: 0.40 (Empirically Determined)

**Status:** TUNED (July 8, 2026)  
**Analysis script:** `danger/find_optimal_threshold.py`

### Problem
`FACE_MATCH_THRESHOLD=0.48` was missing ~75% of same-person cross-camera pairs because ArcFace cosine similarity on CCTV overlaps heavily: same-person (0.13-0.73) vs different-person (-0.14 to 0.63). P10(same)=0.20, P90(diff)=0.27 — they overlap at every percentile.

### Analysis (34 same-person, 80 different-person pairs)

| Threshold | Same caught | Diff falsely merged | 
|---|---|---|
| 0.20 | ~85% | ~10% |
| 0.30 | ~50% | ~5% |
| 0.40 | ~35% | ~3% |
| 0.48 (old) | ~25% | <1% |

**Decision: 0.40** in the dedup job (not real-time identity engine). Catches 35% of same-pairs (~30% more than 0.48) with only ~3% false merge rate. The real-time identity engine keeps 0.48 for face matches — dedup job handles the rest.

---

## Diagnostic Scripts

### Check for duplicate identities (fast, index-assisted)
```bash
cd /gmr/gmr
PYTHONPATH=. venv/bin/python danger/dedup_faces.py [--threshold 0.48]
```
Uses pgvector LATERAL query — completes in seconds regardless of DB size.

### Reset all tracking data
```bash
cd /gmr/gmr
PYTHONPATH=. venv/bin/python danger/reset_tracking_data.py [--yes]
```
Wipes all tracking, person identity, embeddings, events, analytics, and MinIO crops. Preserves config (cameras, zones, rules, stores, users).

### Find optimal face similarity threshold
```bash
cd /gmr/gmr
PYTHONPATH=. venv/bin/python danger/find_optimal_threshold.py [--sample 50]
```
Samples same-person vs different-person pairs and computes F1-optimal threshold.

### Fix genders using SigLIP2
```bash
cd /gmr/gmr
PYTHONPATH=. venv/bin/python danger/fix_gender_siglip2.py [--apply]
```
Cross-checks all persons and corrects `PersonIdentity.gender` if SigLIP2 disagrees.

### Clean contaminated face embeddings
```bash
cd /gmr/gmr
PYTHONPATH=. venv/bin/python danger/clean_contaminated_embeddings.py [--apply]
```
Detects and removes face embeddings from different people stored under the same identity (negative pairwise cosine similarity).

### Test gender models
```bash
cd /gmr/gmr
PYTHONPATH=. venv/bin/python danger/test_mivolo.py [person_id ...]
PYTHONPATH=. venv/bin/python danger/test_siglip2.py [person_id ...]
PYTHONPATH=. venv/bin/python danger/test_siglip2_body.py [person_id ...]
```

---

## Configuration Quick Reference

| Setting | Value | Notes |
|---|---|---|
| `FACE_MATCH_THRESHOLD` | `0.48` | Positive face match threshold (real-time) |
| `FACE_CONTRADICTION_THRESHOLD` | `0.25` | Disassociation gate |
| `FACE_BODY_EXCLUSION_THRESHOLD` | `0.30` | Body candidate face gate |
| `FACE_CONTAMINATION_THRESHOLD` | `0.35` | Running-consensus contamination gate |
| `FACE_MIN_EYE_SPREAD` | `0.25` | Frontal gate |
| `FACE_FRONTALITY_WEIGHT` | `0.35` | Frontality contribution to face_quality |
| `FACE_IDENTITY_MIN_SCORE` | `0.60` | Min face_quality for identity creation |
| `FACE_IDENTITY_MIN_DETECTIONS` | `2` | Min good face detections per track |
| `MAX_FACE_EMBEDDINGS_PER_PERSON` | `5` | Multi-angle face storage cap |
| `REID_MATCH_THRESHOLD` | `0.50` | Body ReID match threshold (MSMT17 OSNet, same-person median=0.680, diff-person median=0.386, best F1=0.49. Previously 0.85 — calibrated against broken ImageNet-backbone weights) |
| `BODY_CONTAMINATION_THRESHOLD` | `0.50` | Body contamination gate (store-time + dedup cleanup). Was 0.60, lowered to 0.50 with MSMT17 weights — same-person p25=0.537, diff-person p75=0.444. |
| `SIGLIP2_MODEL_ID` | `google/siglip2-base-patch16-224` | Gender model (100% on clean CCTV) |
| `MIVOLO_MODEL_PATH` | `models/mivolo/mivolo_fairface.pth.tar` | Age model (MiVOLO kept for age only) |
| `STAFF_DURATION_THRESHOLD_SECONDS` | `1800` | Staff detection: total visible time >30 min |
| `STAFF_DISTINCT_DAYS_THRESHOLD` | `3` | Staff detection: appeared on 3+ days |
| Stale track timeout | `5.0s` | `track_manager.py` |
| Dedup job interval | `10 min` | `jobs/scheduler.py` — includes staff classification + MinIO sweep |
| Staff classification interval | `10 min` | Runs inside dedup job |
| MinIO sweep interval | `10 min` | Runs inside dedup job — cross-references all `crops/` against DB |
| `FACE_MATCH_THRESHOLD` | `0.48` | `.env` | Face similarity threshold for matching |
| `FACE_IDENTITY_MIN_SCORE` | `0.60` | `config.py` | Min face quality for identity creation |
| `FACE_IDENTITY_MIN_DETECTIONS` | `2` | `config.py` | Min good face detections per track |
| `MAX_FACE_EMBEDDINGS_PER_PERSON` | `5` | `config.py` | Multi-angle face storage cap |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.30` | `.env` | Person detection confidence |
| Billing dwell threshold | `90s` | DB `rules` table | Time in billing zone before purchase event |
| Stale track timeout | `5.0s` | `track_manager.py` | Track removed if unseen for this long |
