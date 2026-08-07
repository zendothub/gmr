# Retail Eye Insights — Documentation

On-premises AI CCTV retail analytics for pharmacies. The system processes live
RTSP camera feeds through a per-frame AI pipeline (YOLO detection, face/body
ReID, gender/age classification), stores person identities with pgvector
similarity search, and generates analytics events (zone enter/exit, billing
interactions, dwell time, staff detection).

**Stack:** FastAPI (Python 3.10) + PostgreSQL/pgvector + MinIO + RTX 4070 Ti GPU
**Frontend:** React 19 + TanStack Start (separate repo)
**Deployment:** Bare-metal Linux, 2 cameras, 3 systemd processes

---

## System Architecture (High-Level)

```
                        ┌─────────────────────────────────────────────────┐
                        │              RTSP CAMERAS (2)                    │
                        │   5502bd64 (entry)   704189a2 (counter)          │
                        └──────────────────┬──────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  retail-ai.service  (API Server, ~3.4 GB GPU)                                │
│                                                                              │
│  RTSP → LatestFrameBuffer → YOLOv8+ByteTrack → InsightFace (full frame)     │
│      → Occlusion IoU flag → Hungarian face-to-track (harden + ambig reject) │
│      → SigLIP2 gender (face-only, margin δ=0.5) + InsightFace age (median)  │
│      → OSNet body ReID (0% pad; skip if occluded) → IdentityDecisionEngine  │
│      → Zone event detection → Rule evaluation → Persist to DB + MinIO        │
│      → StreamBroadcaster (FFmpeg NVENC → MediaMTX → WebRTC)                  │
│                                                                              │
│  Models: YOLOv8n, InsightFace buffalo_l (+genderage), OSNet MSMT17, SigLIP2 │
│  State:  ByteTrack tracks, face/gender-margin/age-sample accumulation        │
└──────────────────────────┬───────────────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
┌──────────────────────┐    ┌──────────────────────────┐
│  PostgreSQL +        │    │  MinIO (S3)              │
│  pgvector            │    │                          │
│                      │    │  crops/ (body + face)    │
│  person_identities   │    │  snapshots/ (events)     │
│  person_embeddings   │    │  clips/ (recordings)      │
│  person_face_emb     │    └──────────────────────────┘
│  track_sessions      │
│  events              │
│  billing_interactions│
│  zones, rules        │
└──────────────────────┘
              ▲
              │
┌─────────────┴─────────────────────────────────────────────────────────────────┐
│  retail-ai-worker.service  (Background Jobs, ~100 MB, NO GPU)                 │
│                                                                               │
│  - deduplicate_persons (10 min): merge duplicates, absorb embeddings,         │
│    clean contamination, sweep MinIO, classify staff                            │
│  - close_stale_track_sessions (5 min)                                         │
│  - probe_cameras (2 min)                                                       │
│  - daily_analytics (00:15) + storage_cleanup (02:00)                          │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│  reextract-faces.timer  (GPU oneshot, every 20 min)                           │
│                                                                               │
│  Re-extracts face embeddings from MinIO track crops for persons left          │
│  faceless by contamination cleanup. Deletes person if no face found in any    │
│  crop. Loads InsightFace (1.5 GB GPU), frees on exit.                         │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## Identity Decision Flow

This is the core matching logic that decides whether a tracked person matches
an existing identity or creates a new one. It runs inside the API server's
camera worker, once per ReID window (every 5 frames accumulated).

```
Frame N+5 (ReID window fires):
  │
  ├── Step 1: FACE SEARCH (highest priority)
  │   │
  │   │  For each accumulated face embedding:
  │   │    search_similar_face() → best per-person candidate + last_seen_at
  │   │
  │   ├── face_sim >= 0.40 (FACE_MATCH_THRESHOLD)?
  │   │     → MATCH (strict, any age of candidate, confident)
  │   │
  │   ├── face_sim >= 0.35 (FACE_MATCH_THRESHOLD_RECENT)
  │   │   AND candidate last_seen within 5 min (RECENT_WINDOW)?
  │   │     │
  │   │     ├── >= 3 total cross-pairs?
  │   │     │     → check median of ALL cross-pairs
  │   │     │     ├── median >= 0.30 → MATCH (recent, grey-zone validated)
  │   │     │     └── median <  0.30 → REJECT (single lucky crop, diff person)
  │   │     │
  │   │     └── < 3 cross-pairs?
  │   │           → MATCH (too few to check median, fall back to best-pair)
  │   │
  │   └── No face match → fall through to body search
  │
  ├── Step 2: BODY ReID (fallback, with face contradiction gate)
  │   │
  │   │  search_similar() → top-5 unique-person body candidates
  │   │  (then median vs that person's FULL body gallery, n_bodies ≥ 2)
  │   │
  │   │  For each candidate:
  │   │    ├── Face exclusion gate:
  │   │    │     recent candidate (< 5 min): bar = 0.25 (relaxed)
  │   │    │     older candidate:            bar = 0.30 (strict)
  │   │    │     face_sim < bar → SKIP (face contradicts body)
  │   │    │
  │   │    └── Score body_median to gallery
  │   │
  │   ├── Top body_median ≥ 0.50 AND not ambiguous vs 2nd?
  │   │     → MATCH (strict body, non-confident)  [Body Match]
  │   │
  │   └── Else recent override:
  │         candidate last_seen within 5 min?
  │         body_median >= 0.55 AND >= 2 body embeddings?
  │           → MATCH (recent body-only, non-confident)
  │         else → no match
  │
  └── No face or body match?
        → CREATE NEW PERSON (requires face: score >= 0.60, >= 2 good faces)
        (SAME_CAM reject / MATCH STALE → no create; leave unassigned)
```

**Key thresholds:**

| Setting | Value | Purpose |
|---|---|---|
| `FACE_MATCH_THRESHOLD` | 0.40 | Strict face match (any time) |
| `BODY_CROP_PADDING_PCT` | 0.0 | Body crop padding for OSNet (env-overridable) |
| `OCCLUSION_IOU_THRESHOLD` | 0.10 | Pairwise IoU → mark tracks occluded |
| `ENABLE_HUNGARIAN_FACE_ASSIGN` | True | Hungarian face↔track assignment |
| `ENABLE_STAFF_REATTACH` | True | Staff-only recent body reattach (blur/side face fragments) |
| `STAFF_REATTACH_BODY_MEDIAN` | 0.70 | Min median body sim for staff reattach |
| `STAFF_REATTACH_FACE_MIN` | 0.30 | Face floor; below → reject staff reattach |
| `STAFF_REATTACH_REQUIRE_FACE` | True | Reject faceless body-only staff reattach |
| `FACE_MATCH_THRESHOLD_RECENT` | 0.35 | Relaxed face match within 5-min window |
| `FACE_MATCH_MEDIAN_THRESHOLD` | 0.30 | Median of all cross-pairs must pass (grey-zone validation) |
| `REID_MATCH_THRESHOLD` | 0.50 | Body ReID consensus threshold |
| `RECENT_BODY_SINGLE_MATCH_THRESHOLD` | 0.55 | Body-only override within recent window |
| `FACE_CONTRADICTION_THRESHOLD` | 0.25 | Face disassociation + recent non-contradiction bar |
| `FACE_BODY_EXCLUSION_THRESHOLD` | 0.30 | Body candidate face exclusion (strict, old candidates) |
| `FACE_CONTAMINATION_THRESHOLD` | 0.35 | Face contamination cleanup (median) |
| `BODY_CONTAMINATION_THRESHOLD` | 0.50 | Body contamination cleanup (median) |
| `RECENT_WINDOW_MINUTES` | 5 | Recent window for relaxed matching |

**Key principles:**
- **Face is the authority.** Body ReID is a backup when face is absent/blurred.
- **Body ReID is clothing-dependent.** Only reliable within the same visit (same clothes). Unreliable across days (different clothes) and unreliable between staff wearing similar uniforms.
- **Median, not best.** Grey-zone face matches require median of ALL cross-pairs >= 0.30. A single lucky crop hitting 0.35 is rejected if the rest are low.
- **Contamination gates.** Face and body embeddings are checked for cluster fit at store time AND at dedup absorb time. Embeddings that don't fit the winner's cluster are dropped, not moved.

---

## Documentation Index

### 1. Architecture & Pipeline

| Document | Description |
|---|---|
| [architecture_and_pipeline.md](architecture_and_pipeline.md) | Full system architecture: camera onboarding, per-frame AI pipeline (8 stages), DB schema, zones, rules, models. Contains the master ASCII pipeline diagram. **Start here for understanding the system.** |
| [camera_worker_architecture.md](camera_worker_architecture.md) | Camera worker internals: lifecycle, track management, face-first identity, multi-face storage, per-camera YOLO isolation, inference pool. |
| [COMPLETE_SYSTEM_FLOW.md](COMPLETE_SYSTEM_FLOW.md) | End-to-end data flow: camera processing, ReID, multi-camera synchronization. Covers how identities are shared across cameras. |

### 2. Identity & ReID

| Document | Description |
|---|---|
| [RECENT_WINDOW_AND_CONTAMINATION_FIX.md](RECENT_WINDOW_AND_CONTAMINATION_FIX.md) | Recent-window two-tier matching (face + body), face median check for grey-zone matches, contamination cleanup (median-outlier removal), absorb contamination gates. **Latest identity work (July 9-10, 2026).** |
| [CRITICAL_FINDINGS.md](CRITICAL_FINDINGS.md) | Critical bugs: ByteTrack state corruption from shared YOLO instances, inference pool threading issues. Read before modifying camera worker or detection code. |
| [DUPLICATE_IDENTITY_ROOT_CAUSE.md](DUPLICATE_IDENTITY_ROOT_CAUSE.md) | Root cause of 885 duplicate person identities: broken union-find, face-only dedup threshold gap, contamination. Documents all dedup fixes. |
| [REID_PRODUCTION_INTEGRATION.md](REID_PRODUCTION_INTEGRATION.md) | Body ReID production enhancements: torso-keypoint crop-quality assessment, weighted scoring formula. |
| [CROP_LIFECYCLE_AND_MINIO_FIXES.md](CROP_LIFECYCLE_AND_MINIO_FIXES.md) | MinIO crop lifecycle: deferred deletion, orphan sweep, 404 fixes for face/body crop URLs. |

### 3. Analytics & Dashboard

| Document | Description |
|---|---|
| [DASHBOARD_V2_DATA_FLOW.md](DASHBOARD_V2_DATA_FLOW.md) | Dashboard V2 API: how the endpoint extracts and processes data (router → service layer → SQL queries). |
| [ANALYTICS_DASHBOARD_CHANGES.md](ANALYTICS_DASHBOARD_CHANGES.md) | Footfall counting fix: switched from total track sessions to COUNT(DISTINCT person_identity_id). |
| [BILLING_ZONE_SQL_QUERIES.md](BILLING_ZONE_SQL_QUERIES.md) | SQL recipes for manually creating billing zones and rules to enable purchase tracking. |
| [PURCHASE_TRACKING_SETUP.md](PURCHASE_TRACKING_SETUP.md) | User-facing setup guide: create a billing zone, attach rules, verify dashboard purchase count. |

### 4. Fixes & Incidents

| Document | Description |
|---|---|
| [TIMEZONE_FIX.md](TIMEZONE_FIX.md) | Dashboard crash caused by `Asia/Calcutta` → `Asia/Kolkata` timezone fix for PostgreSQL/asyncpg. |
| [../DATABASE_CORRUPTION_FIX.md](../DATABASE_CORRUPTION_FIX.md) | Recovery guide for corrupted DB: Alembic version tracking out of sync with missing tables. |

### 5. Capacity & Cost

| Document | Description |
|---|---|
| [HOST_CAPACITY_AND_CAMERA_COST.md](HOST_CAPACITY_AND_CAMERA_COST.md) | Live host RAM/CPU/GPU/storage present vs used, how many cameras can be added, ₹/cam economics (₹17k OpEx). |

---

## Data Flow Diagrams

### Per-Frame AI Pipeline

```
RTSP Stream
    │
    ▼
LatestFrameBuffer (background thread, always keeps latest frame)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: YOLO Detection + ByteTrack Tracking                    │
│   detector.track(frame) → [TrackedDetection(track_id, bbox)]   │
│   Model: YOLOv8n (per-camera instance, NOT shared)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: Track State Update + Zone Membership                    │
│   track_manager.update_track() → ActiveTrack (bbox history,     │
│   stability score, confidence)                                   │
│   track_manager.update_zones() → point-in-polygon for each zone │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: Zone Event Detection                                    │
│   zone_event_detector.detect() → zone_enter / zone_exit /       │
│   zone_dwell_milestone (30s, 60s, 120s)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: Rule Evaluation                                         │
│   rule_evaluator.evaluate() → billing_interaction,              │
│   line_crossing, zone_dwell, restricted_zone, possible_purchase  │
│   Cooldown: 30s per [rule_id, track_id] pair                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 5: Stale Track Cleanup                                    │
│   Remove tracks not seen for > 5 seconds                        │
│   Close DB track_sessions for removed tracks                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 6: Persist Batch (single DB transaction)                   │
│                                                                  │
│  6a. Create track_session (new tracks)                          │
│      → Save initial crop to MinIO                                │
│      → Insert "person_entered_view" event                        │
│                                                                  │
│  6b. Run ReID (every 5 accumulated frames):                      │
│      → extract_crop(padding=BODY_CROP_PADDING_PCT=0.0)          │
│      → OSNet.extract (skip if track.is_occluded)                 │
│      → Face from pre-assigned full-frame match                   │
│      → SigLIP2 gender (face+margin δ=0.5) + IF genderage age     │
│      → Face contamination gate (track + store level)            │
│      → Body contamination gate (median to cluster)               │
│      → IdentityDecisionEngine.decide_identity()                  │
│      → Store embeddings in person_embeddings /                   │
│        person_face_embeddings (pgvector)                        │
│      → UPDATE track_sessions.person_identity_id                  │
│                                                                  │
│  6c. Persist events (rule + zone events)                         │
│      → Save frame snapshot to MinIO                              │
│      → INSERT events, billing_interactions                       │
│                                                                  │
│  6d. Persist observations (every 2s per track)                   │
│      → INSERT track_observations (bbox, zone_ids)                │
│                                                                  │
│  6e. Close track sessions (stale tracks)                         │
│      → UPDATE is_active=False, ended_at, best_crop               │
│      → Insert "person_left_view" event                          │
│                                                                  │
│  6f. db.commit()                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 7: Stream Broadcast (burn-in)                              │
│   latest_tracks → StreamBroadcaster (separate thread)            │
│   → Draw bboxes + zone polygons on frame                        │
│   → FFmpeg NVENC → MediaMTX → WebRTC/HLS                        │
└─────────────────────────────────────────────────────────────────┘
```

### Dedup Job Flow (background worker, every 10 min)

```
deduplicate_persons()
    │
    ├── Step 1: Find duplicate pairs (pgvector LATERAL, face >= 0.40)
    │
    ├── Step 2: Union-find connected components (recursive path compression)
    │
    ├── Step 3: For each component, pick winner (highest face_score)
    │
    ├── Step 4: Merge each loser into winner (per-merge SAVEPOINT):
    │   │
    │   ├── Reassign track_sessions, events, billing → winner
    │   ├── Absorb face embeddings (contamination-gated: median fit >= 0.35)
    │   ├── Absorb body embeddings (contamination-gated: median fit >= 0.50)
    │   ├── Re-vote person-level gender (majority across all tracks)
    │   ├── Absorb visit_count + first_seen_at
    │   └── DELETE loser (cascade: person_embeddings, person_face_embeddings)
    │
    ├── Step 5: Clean contaminated face embeddings (median-outlier, >= 0.35)
    │
    ├── Step 6: Clean contaminated body embeddings (median-outlier, >= 0.50)
    │
    ├── Step 7: Sweep orphaned MinIO crops (cross-ref DB vs MinIO)
    │
    └── Step 8: Classify staff (duration > 30 min OR 3+ distinct days)
```

### Crop Lifecycle (MinIO)

```
                  ┌─────────────┐
                  │  RTSP Frame │
                  └──────┬──────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        ┌──────────┐ ┌────────┐ ┌────────────┐
        │ Body crop│ │Face    │ │ Snapshot   │
        │ (OSNet)  │ │crop    │ │ (events)   │
        └────┬─────┘ └───┬────┘ └─────┬──────┘
             │           │            │
             ▼           ▼            │
        MinIO        MinIO            │
        crops/       crops/           │
        crop_*       face_*           │
             │           │            ▼
             │           │       MinIO snapshots/
             │           │            │
             ▼           ▼            ▼
        ┌─────────────────────────────────┐
        │  DEFERRED DELETION QUEUE        │
        │  (class-level set of paths)     │
        │  Never delete immediately —     │
        │  always defer to periodic sweep │
        └──────────────┬──────────────────┘
                       │
                       ▼ (every 10 min, dedup job)
              ┌────────────────────┐
              │  Sweep: cross-ref  │
              │  ALL MinIO crops/  │
              │  vs live DB paths  │
              │  → delete only     │
              │    unreferenced    │
              └────────────────────┘

DB references (protected from deletion):
  - person_embeddings.crop_path
  - person_face_embeddings.face_crop_path
  - person_identities.face_crop_path
  - track_sessions.best_crop_path
```

---

## Models

| Task | Model | Why |
|---|---|---|
| Person detection + tracking | YOLOv8n + ByteTrack | Lightweight, fast on GPU. Per-camera instance (NOT shared). |
| Face detection + embedding + age | InsightFace buffalo_l (SCRFD + ArcFace + genderage) | Full-frame detection (1 call). Age years from genderage head. |
| Body ReID | OSNet 512-dim (MSMT17 checkpoint) | ReID-finetuned. Same-person median=0.69, diff-person median=0.38. |
| Gender | SigLIP2 face-only + margin δ=0.5 | 7+7 prompts. M only if `(male−female)>0.5`. Body path off (F→M bias). |
| Age | InsightFace genderage + multi-face **median** | Replaced MiVOLO FairFace (young collapse). Product gender remains SigLIP2. |

**GPU memory:** ~4.3–4.5 GB (InsightFace+genderage ~1.5GB + YOLO 300MB + OSNet 100MB + SigLIP2 1.4GB + overhead; MiVOLO not loaded)

---

## File Structure

| Concern | Primary file(s) |
|---|---|
| AI Pipeline (per-camera) | `app/modules/ai_runtime/camera_worker.py` |
| Identity decisions | `app/modules/reid/identity_decision_engine.py` |
| Face detection | `app/modules/reid/insightface_analyzer.py` |
| Gender classification | `app/modules/reid/siglip2_analyzer.py` (`SIGLIP2_GENDER_MARGIN_DELTA`) |
| Age prediction | `app/modules/reid/insightface_analyzer.py` (genderage head + median) |
| Body ReID (OSNet) | `app/modules/reid/osnet_extractor.py` |
| Track management | `app/modules/tracking/track_manager.py` |
| Rule engine | `app/modules/rule_engine/rule_evaluator.py` |
| Zone events | `app/modules/rule_engine/zone_event_detector.py` |
| Analytics queries | `app/modules/analytics/service.py` |
| Background jobs (dedup, staff) | `app/modules/jobs/tasks.py` |
| Config / thresholds | `app/config.py` |
| Stream broadcaster | `app/modules/ai_runtime/stream_broadcaster.py` |
| YOLO detector | `app/modules/detection/yolo_detector.py` |
| Inference pool | `app/modules/ai_runtime/inference_pool.py` |
| Crop helpers | `app/utils/image_utils.py` |
| DB models | `app/core/db/models/` |
| Danger scripts | `danger/` |
