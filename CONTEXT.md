# CONTEXT.md — Retail Eye Insights Cross-Session Memory

> **Last updated:** July 8, 2026  
> **Purpose:** Every AI session MUST read this file first. It contains all architectural decisions, model choices, threshold values, known issues, and the reasoning behind every critical change made to the system.

---

## Project Summary

**Retail Eye Insights** — On-premises AI CCTV retail analytics for pharmacies.  
**Deployment:** Bare-metal Linux, 2× camera (Apolo counter + Apolo entry), RTX 4070 Ti GPU.  
**Backend:** FastAPI (Python 3.10, single uvicorn worker), PostgreSQL + pgvector, MinIO.  
**Frontend:** React 19 + TanStack Start + Recharts + Tailwind v4.

---

## AI Pipeline (Per Camera, Per Frame)

```
Frame (2880×1620) 
  → YOLOv8 + ByteTrack (per-camera model, NOT shared — CRITICAL)
  → InsightFace SCRFD on FULL FRAME (1 call, not per-track)
  → Match faces to body tracks (centre-in-bbox + scoring heuristic)
  → Extract face crop from full frame at native res → resize_pad_square(224²)
  → SigLIP2 gender (pre-computed 7+7 text prompts) + MiVOLO age
  → BodyReID (OSNet 512-dim, deduplicated by person, 2-of-3 consensus, 0.85 threshold)
  → Gender voting (continuous, all detected faces, not just frontal)
  → pgvector face/body embedding storage (48h body window, infinite face window)
  → Rule engine (zone dwell, line crossing, billing)
  → Stream broadcaster (annotated 15fps → FFmpeg NVENC → MediaMTX → WebRTC)
```

---

## Model Choices & Why

| Task | Model | Why | Accuracy |
|---|---|---|---|
| **Face detection + embedding** | InsightFace buffalo_l (detection + recognition only) | Best face detector, ArcFace embeddings for re-ID | — |
| **Gender** | **SigLIP2** (google/siglip2-base-patch16-224) | 100% on clean retail CCTV. 7 female + 7 male text prompts pre-computed at startup (~1.4GB GPU) | ~18ms/img |
| **Age** | **MiVOLO** (ViT-Small, FairFace 3-class checkpoint) | Kept for age prediction only (SigLIP2 doesn't provide age) | ~10-30ms/img |
| **Body ReID** | **OSNet** (512-dim) via torchreid | Lightweight, but overlapping same/diff distributions (0.58-0.83 vs 0.10-0.40) | — |
| ~~Gender~~ | ~~MiVOLO ViT-Small~~ | 11% accuracy on CCTV — replaced by SigLIP2 | — |
| ~~Gender~~ | ~~DeepFace~~ | 0% accuracy — unusable | — |
| ~~Gender~~ | ~~InsightFace buffalo_l genderage~~ | 85-90%, systematic misclassification — replaced by SigLIP2 | — |

---

## Critical Thresholds — DO NOT CHANGE WITHOUT ANALYSIS

| Setting | Value | Where Used | Rationale |
|---|---|---|---|
| `FACE_MATCH_THRESHOLD` | `0.48` | Real-time identity matching | Same-person X-angle 0.40-0.70. Below 0.48 misses too many. Above 0.48 increases false merges |
| `FACE_CONTRADICTION_THRESHOLD` | `0.25` | Disassociation gate | Same-person X-angle ≥0.40. 0.25 avoids false disassociation |
| `FACE_BODY_EXCLUSION_THRESHOLD` | `0.30` | Body candidate face gate | More permissive than match — allows body fallthrough for X-angle faces |
| `FACE_CONTAMINATION_THRESHOLD` | `0.35` | Running-consensus contamination check | Reject face if sim < 0.35 to ALL prior faces (different person 0.10-0.30, same 0.40+) |
| `REID_MATCH_THRESHOLD` | `0.85` | Body ReID match | Raised from 0.80. OSNet self-sim floor 0.58. Need ≥0.85 for clean separation |
| `DEDUP_THRESHOLD` (dedup job) | `0.40` | Periodic dedup job only | Empirically determined. Catches 35% of same-pairs with 3% false merge rate |
| `FACE_MIN_EYE_SPREAD` | `0.25` | Frontal gate | 3/4-view accepted, profile rejected |
| `FACE_IDENTITY_MIN_SCORE` | `0.60` | Person creation gate | Min face_quality (det_score × frontality) |
| `FACE_IDENTITY_MIN_DETECTIONS` | `2` | Person creation gate | Good face count before identity created |
| `STAFF_DURATION_THRESHOLD_SECONDS` | `1800` | Staff detection | Total visible session time >30 min |
| `STAFF_DISTINCT_DAYS_THRESHOLD` | `3` | Staff detection | Appeared on 3+ distinct calendar days |
| `MAX_FACE_EMBEDDINGS_PER_PERSON` | `5` | Face storage cap | Pruned when exceeded |
| `MAX_EMBEDDINGS_PER_PERSON` (body) | `10` | Body storage cap | Pruned when exceeded |
| `BODY_ONLY_CONFIDENCE_LIMIT` | `0.95` | Body-only match confidence | Body-only matches demoted to non-confident |

---

## Identity Decision Flow

```
Frame N+4 (window fires):
  1. Face search (SigLIP2 gender + ArcFace embedding)
     → sim ≥ FACE_MATCH_THRESHOLD (0.48) → MATCH (confident if face-matched)
     → sim < 0.48 → fall through to body search

  2. Body search (OSNet, deduplicated by person, top-5 unique identities)
     → 2 of top-3 candidates must agree on same person_id
     → sim ≥ REID_MATCH_THRESHOLD (0.85) → MATCH (non-confident, temporary)
     → sim < 0.85 → CREATE NEW PERSON (requires face: 3 gates above)
```

**Key rules:**
- Face-confirmed identities → permanent (`is_confident = True`)
- Body-only identities → temporary (`is_confident = False`)
- Temporary identities merged or deleted by dedup job
- `REQUIRE_FACE_FOR_IDENTITY = True` — NO person created without face
- Contamination gate: new face rejected if min_sim_to_existing < 0 (different person's face)

---

## Staff Detection & Purchase Dedup

**Staff auto-classification** (runs every 10 min in dedup job):
- `PersonIdentity.is_staff` boolean, indexed
- Duration > 30 min OR 3+ distinct days → promoted to staff
- If BOTH signals fall below → demoted

**Purchase counting** (5 query sites in analytics):
- `COUNT(DISTINCT person_identity_id)` — not raw row count
- `WHERE NOT is_staff` — staff excluded
- One `BillingInteraction` per `track_session_id` + `zone_id` combo

---

## MinIO Crop Lifecycle

**All deletions are DEFERRED.** The `_minio_cleanup()` function adds paths to a class-level set. The periodic sweep (every 10 min in dedup job) cross-references ALL MinIO `crops/` files against live DB references and only deletes truly unreferenced files.

**Known-safe references immune to deletion:**
- `person_face_embeddings.face_crop_path`
- `person_identities.face_crop_path`
- `person_embeddings.crop_path`
- `track_sessions.best_crop_path`

**Key invariant:** NO immediate MinIO deletion anywhere in the codebase. Every delete path goes through the deferred queue.

---

## File Structure — What Lives Where

| Concern | Primary File(s) |
|---|---|
| AI Pipeline (per-camera) | `camera_worker.py` |
| Face detection (INSIGHTFACE) | `insightface_analyzer.py` |
| Gender (SIGLIP2) | `siglip2_analyzer.py` |
| Age (MIVOLO) | `mivolo_analyzer.py` |
| Body ReID (OSNet) | `osnet_extractor.py` |
| Identity decisions | `identity_decision_engine.py` |
| Track management | `track_manager.py` |
| Rule engine | `rule_evaluator.py`, `zone_event_detector.py` |
| Analytics queries | `analytics/service.py` |
| Background jobs | `jobs/tasks.py`, `jobs/scheduler.py` |
| Staff detection | `jobs/tasks.py` (deduplicate_persons) |
| MinIO sweep | `jobs/tasks.py` (_sweep_orphaned_crops) |
| Crop helpers | `image_utils.py` |
| Config/Thresholds | `config.py` |
| Stream broadcaster | `stream_broadcaster.py` |
| Inference pool | `inference_pool.py` |
| YOLO detector | `yolo_detector.py` |

---

## Danger Scripts (Diagnostics & Fixes)

| Script | Purpose |
|---|---|
| `danger/dedup_faces.py` | Find duplicate identities (LATERAL query) |
| `danger/find_optimal_threshold.py` | Same vs different-person similarity distribution |
| `danger/fix_gender_siglip2.py` | Cross-check + update all genders via SigLIP2 |
| `danger/fix_gender_mivolo.py` | Cross-check + update all genders via MiVOLO (deprecated) |
| `danger/clean_contaminated_embeddings.py` | Remove contaminated face embeddings (negative sim) |
| `danger/reset_tracking_data.py` | Full data reset (preserves config) |
| `danger/test_mivolo.py` | Test MiVOLO on specific persons |
| `danger/test_siglip2.py` | Test SigLIP2 face-only on specific persons |
| `danger/test_siglip2_body.py` | Test SigLIP2 body similarity on specific persons |
| `danger/test_deepface.py` | Test DeepFace on specific persons (0% accuracy — deprecated) |

---

## Known Issues

1. **OSNet body ReID overlap** — Same-person sim 0.58-0.83 overlaps with different-person 0.10-0.40. No single threshold perfectly separates. Current mitigation: 0.85 + 2-of-3 consensus gate + temporary-only body identities.

2. **SigLIP2 face-only misses culturally distinctive attire** — A woman in a saree was missed by face-only (37-40% F prob). Mitigated by body crop integration (3× weight, 79% confidence with body context). Still may miss edge cases.

3. **ArcFace cross-angle degradation** — Cross-camera same-person sim can drop to 0.20. Mitigated by 0.40 dedup threshold in periodic job.

4. **Face contamination during close-proximity** — When 2+ people stand shoulder-to-shoulder, faces from adjacent persons can appear in body crops. Mitigation: full-frame face detection + contamination gate (reject neg-sim) + contaminated embedding cleaner script.

5. **MinIO sweep deletes old crops** — Body crops from merged/deleted persons cascade-delete. The sweep correctly cleans them up. Face crops from the `_prune_face_embeddings` path are protected by the deferred-deletion + reference check.

---

## Current State (as of last update)

- **Gender:** SigLIP2 running live with 7+7 prompts, face+body combined
- **Age:** MiVOLO FairFace 3-class checkpoint
- **Body ReID:** OSNet with 0.85 threshold + 2-of-3 consensus gate
- **Staff detection:** Running every 10 min, 30-min duration threshold
- **Purchase counts:** Deduplicated per person, staff excluded
- **Face detection:** Full-frame, 1 call per frame, padded resize
- **Contamination gate:** Active in `_store_face_embedding`
- **Dedup job:** 0.40 threshold, runs every 10 min
- **MinIO:** Deferred deletion, sweep every 10 min
- **GPU:** ~4.5 GB used (InsightFace ~1.5GB + YOLO ~300MB + OSNet ~100MB + MiVOLO ~100MB + SigLIP2 ~1.4GB + Torch overhead)
