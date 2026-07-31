# CONTEXT.md — Retail Eye Insights Cross-Session Memory

> **Last updated:** July 31, 2026 (Billing visit repair post-dedup: fragmented dwell sum + null BI fill)  
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
  → Mark occluded tracks (pairwise body IoU ≥ OCCLUSION_IOU_THRESHOLD 0.10)
  → GLOBAL face-to-track assignment (Hungarian; greedy fallback)
    → Membership: face centre inside ORIGINAL body bbox (no expansion)
    → Immature tracks (good_face_count < 2): face in upper 45% of body + ≥70% face area inside body + score ≥ 0.35
    → Ambiguous reject: if top2/top1 scores for same face ≥ 0.85 → assign face to no one this frame
    → Scoring: det_score × (size + centre + temporal continuity; continuity disabled while occluded)
    → last_face_center updated only when assigned AND not occluded
  → Extract face crop from full frame at native res → resize_pad_square(224²)
  → SigLIP2 gender face-only + margin δ=0.5 (mean margin over track faces)
  → InsightFace genderage age sample each face → track median → person estimated_age
  → Body crop with BODY_CROP_PADDING_PCT=0.0 (tight YOLO box); skip OSNet body ReID when track.is_occluded
  → BodyReID (OSNet 512-dim, MSMT17; live body **median** n≥2, thr 0.50 / recent 0.55 + ambiguity)
  → Face contamination gate (normalized cosine sim, ≥2 prior faces in track)
  → pgvector face/body embedding storage (48h body window, infinite face window)
    → Face embeddings L2-normalized at store time (norm=1.0)
    → Body contamination gate (median sim to existing cluster, ≥3 prior, 0.50 threshold)
  → Rule engine (zone dwell, line crossing, billing)
  → Stream broadcaster (annotated 15fps → FFmpeg NVENC → MediaMTX → WebRTC)
```

---

## Model Choices & Why

| Task | Model | Why | Accuracy |
|---|---|---|---|
| **Face detection + embedding + age** | InsightFace buffalo_l (`detection` + `recognition` + `genderage`) | SCRFD + ArcFace re-ID; age years from genderage (product **gender** still SigLIP2) | — |
| **Gender** | **SigLIP2** face-only + **margin δ=0.5** | Face-only (body disabled). M only if `(male_best−female_best)>0.5`. Fixes female→male bias; ~98% on labeled set. Body×3 and hair prompts hurt. | ~18ms/img |
| **Age** | **InsightFace buffalo_l genderage** + multi-face **median** | FairFace MiVOLO collapsed ~74% under-18 on live DB; IF median gives realistic 25–44 mix. Gender still SigLIP2. | free with face det |
| **Body ReID** | **OSNet** (512-dim) via torchreid, **MSMT17 checkpoint** (`osnet_x1_0_msmt17_combineall`) | Lightweight, discriminative with proper ReID-finetuned weights. Same-person cross-camera median=0.680, diff-person median=0.386 | — |
| ~~Age~~ | ~~MiVOLO FairFace 3-class~~ | Young bias (midpoints 10/40/75) + single best face — removed from live pipeline Jul 2026. Weights under `models/mivolo/` offline only. | — |
| ~~Gender~~ | ~~MiVOLO ViT-Small~~ | 11% accuracy on CCTV — replaced by SigLIP2 | — |
| ~~Gender~~ | ~~DeepFace~~ | 0% accuracy — unusable | — |
| ~~Gender primary~~ | ~~InsightFace genderage~~ | OK for age; gender secondary if needed. Product gender = SigLIP2. | — |

---

## Critical Thresholds — DO NOT CHANGE WITHOUT ANALYSIS

| Setting | Value | Where Used | Rationale |
|---|---|---|---|
| `FACE_MATCH_THRESHOLD` | `0.40` | Real-time identity matching | Same as dedup. Same-person X-angle 0.40–0.70. (Docs formerly said 0.48 — code/env is 0.40.) |
| `FACE_MATCH_THRESHOLD_RECENT` | `0.35` | Live recent-window face | Grey zone [0.35, 0.40); attach uses same thr via `match_tier` (bugged until 2026-07-17) |
| `BODY_MATCH_AMBIGUITY` | `0.03` | Live body match | Reject if top-2 body medians within gap |
| `FACE_CONTRADICTION_THRESHOLD` | `0.25` | Disassociation gate | Same-person X-angle ≥0.40. 0.25 avoids false disassociation |
| `FACE_BODY_EXCLUSION_THRESHOLD` | `0.30` | Body candidate face gate | More permissive than match — allows body fallthrough for X-angle faces |
| `FACE_CONTAMINATION_THRESHOLD` | `0.35` | Running-consensus contamination check (track + store + dedup) | Reject face if cosine sim < 0.35 to cluster (different person 0.10-0.30, same 0.40+). All comparisons use L2-normalized vectors via `_face_sim()`. |
| `BODY_CONTAMINATION_THRESHOLD` | `0.50` | Body store gate vs **recent-window** cluster (≥3 embs in last 5m) | Never deletes older day bodies. Multi-day clothing change allowed to store. Same-visit stranger reject only. |
| `REID_MATCH_THRESHOLD` | `0.50` | Customer body match (recent gallery median) | Only bodies with `captured_at` in RECENT_WINDOW. + ambiguity 0.03. |
| `DEDUP_THRESHOLD` (dedup job) | `0.40` | Periodic dedup job only | Empirically determined. Catches 35% of same-pairs with 3% false merge rate |
| `FACE_MIN_EYE_SPREAD` | `0.25` | Frontal gate | 3/4-view accepted, profile rejected |
| `FACE_IDENTITY_MIN_SCORE` | `0.60` | Person creation gate | Min face_quality (det_score × frontality) |
| `FACE_IDENTITY_MIN_DETECTIONS` | `2` | Person creation gate | Good face count before identity created |
| `STAFF_DURATION_THRESHOLD_SECONDS` | `1800` | Staff detection | Total visible session time >30 min |
| `STAFF_DISTINCT_DAYS_THRESHOLD` | `3` | Staff detection | Appeared on 3+ distinct calendar days |
| `MAX_FACE_EMBEDDINGS_PER_PERSON` | `5` | Face storage cap | Pruned when exceeded |
| `MAX_EMBEDDINGS_PER_PERSON` (body) | `10` | Body storage cap | Pruned when exceeded |
| `BODY_ONLY_CONFIDENCE_LIMIT` | `0.95` | Body-only match confidence | Body-only matches demoted to non-confident |
| `ENABLE_RECENT_WINDOW_MATCHING` | `True` | Live engine feature flag | Toggles the recent-window relaxed matching (Step D). Rollback switch. |
| `RECENT_WINDOW_MINUTES` | `5` | Live engine + backfill | Same-visit window. Dry-run window sweep: 7–15m does not improve purchase DISTINCT vs 5m. |
| `BODY_MATCH_USE_RECENT_GALLERY_ONLY` | `True` | Live customer body ANN + median | Match only recent bodies; keep old rows (clothing change). |
| `STAFF_BODY_USE_FULL_GALLERY` | `True` | Staff reattach | Activity-recent staff: full lifetime body gallery (uniform stable). |
| `FACE_MATCH_THRESHOLD_RECENT` | `0.35` | Live engine recent-window face path | Relaxed from 0.40 within the window. Catches cross-angle handoffs (best face pair 0.35-0.40) that would otherwise create duplicates. Same metric as existing dedup job LATERAL MAX() — uses best cross-pair, not median (cross-angle faces need the best angle). |
| `RECENT_BODY_SINGLE_MATCH_THRESHOLD` | `0.55` | Live body recent path | Median vs **recent** bodies, ≥2 recent bodies, face non-contradiction, activity-recent (track/emb not stale last_seen alone). |
| `FACE_MATCH_MEDIAN_THRESHOLD` | `0.30` | Live engine + backfill recent face path | When recent face best-pair is in grey zone [0.35, 0.40), require median of ALL cross-pairs ≥ this. Same-person min median=0.401, diff-person p50=0.200. At 0.30: 0% same-person rejected, 97.5% diff-person rejected. Only checked when ≥3 total cross-pairs. Catches single lucky crops from different people that hit 0.35+ on one pair. |
| `OCCLUSION_IOU_THRESHOLD` | `0.10` | Face assign + body ReID | Pairwise body IoU ≥ this → both tracks `is_occluded=True` |
| `FACE_ASSIGN_UPPER_BODY_FRAC` | `0.45` | Face assign (immature) | Face centre must sit in top 45% of body height |
| `FACE_ASSIGN_MIN_OVERLAP` | `0.70` | Face assign (immature) | ≥70% of face bbox area inside body bbox |
| `FACE_ASSIGN_MIN_SCORE_IMMATURE` | `0.35` | Face assign (immature) | Composite score floor for tracks with <2 good faces |
| `FACE_ASSIGN_AMBIGUITY_RATIO` | `0.85` | Face assign | top2/top1 ≥ this for same face → reject face this frame |
| `ENABLE_HUNGARIAN_FACE_ASSIGN` | `True` | Face assign | Hungarian bipartite matching; False → legacy greedy |
| `SKIP_BODY_REID_WHEN_OCCLUDED` | `True` | Body ReID | No OSNet embedding on occluded frames (face-only fallback OK) |
| `BODY_CROP_PADDING_PCT` | `0.0` | Body crop / OSNet | Padding on body crops for ReID. 0.0 = tight YOLO box. Raise via env (e.g. 0.05) if OSNet degrades. Face crops stay at 0.30. |
| `ENABLE_STAFF_REATTACH` | `True` | Identity engine | Staff-only recent body reattach when face fails (blur/side staff fragments) |
| `STAFF_REATTACH_BODY_MEDIAN` | `0.70` | Staff reattach | Median body sim to staff (raised from 0.55 after uniform FPs / b33 pollution) |
| `STAFF_REATTACH_MIN_BODIES` | `2` | Staff reattach | Min stored bodies on staff identity |
| `STAFF_REATTACH_FACE_MIN` | `0.30` | Staff reattach | face_sim below this → reject (raised from 0.20) |
| `STAFF_REATTACH_REQUIRE_FACE` | `True` | Staff reattach | Faceless tracks never reattach on body alone |
| `STAFF_REATTACH_AMBIGUITY` | `0.03` | Staff reattach | Reject if top-2 staff body medians within this gap |
| `ENABLE_SAME_CAMERA_OVERLAP_GATE` | `True` | Live match + dedup | Reject identity match/merge if candidate has concurrent track on same camera (cross-cam OK). |
| `SAME_CAMERA_OVERLAP_MIN_SECONDS` | `1.0` | Live match + dedup | Min overlap seconds to reject (ignore 1-frame glitches). |
| `ENABLE_CONTRADICTION_SAME_ID_BLOCK` | `True` | Live identity | Never rematch the same person_id after face contradiction (blocks self-rematch into mixed galleries). |
| `ENABLE_FACE_MATCH_CLUSTER_MEDIAN` | `True` | Live face match | Require median sim to full face gallery when gallery ≥2 and cross-pairs ≥3. |
| `FACE_MATCH_CLUSTER_MEDIAN_THRESHOLD` | `0.35` | Live face match | Median floor (= contamination thr); lucky best-pair into mixed ID rejected. |
| `SIGLIP2_GENDER_MARGIN_DELTA` | `0.5` | SigLIP2 gender | M only if `(male_best−female_best)>δ`. Top of sweep: ~98% acc, fixes hard F→M. Higher δ (1.0+) mass-flips males to F. |
| `SIGLIP2_USE_BODY_FOR_GENDER` | `False` | SigLIP2 gender | Body×3 path increased female→male errors on this CCTV. Face-only. |

---

## Identity Decision Flow

```
Frame N+4 (window fires):
  1. Face search (SigLIP2 gender + ArcFace embedding)
     → sim ≥ FACE_MATCH_THRESHOLD (0.48) → MATCH (confident if face-matched)
     → sim < 0.48 → fall through to body search

   2. Body search (OSNet, unique persons; median vs gallery)
      → body_median ≥ REID_MATCH_THRESHOLD (0.50), n_bodies≥2, ambiguity OK → MATCH (non-confident)
      → recent window: body_median ≥ 0.55 → MATCH (body_recent)
      → else if face create gates OK → CREATE NEW PERSON (requires face)
```

**Key rules:**
- Face-confirmed identities → permanent (`is_confident = True`)
- Body-only identities → temporary (`is_confident = False`)
- Temporary identities merged or deleted by dedup job
- `REQUIRE_FACE_FOR_IDENTITY = True` — NO person created without face
- Face embeddings are L2-normalized at store time (norm=1.0) so `np.dot()` = cosine similarity
- All in-memory face comparisons use `_face_sim()` which normalizes both vectors before dot product
- Contamination gate: new face rejected if cosine sim < 0.35 to ALL prior faces (different person)
- Body contamination gate: new body rejected if median cosine sim < 0.50 to existing cluster (≥3 embeddings)
- Face dedup: new face skipped if cosine sim > 0.95 to existing face in track (same angle, not a new angle)

---

## Staff Detection & Purchase Dedup

**Staff auto-classification** (runs every 10 min in dedup job):
- `PersonIdentity.is_staff` boolean, indexed
- Duration > 30 min OR 3+ distinct days → promoted to staff
- If BOTH signals fall below → demoted

**Periodic dedup job** (every 10 min, in-process via APScheduler):
1. Merge duplicate persons (face sim ≥ 0.40, union-find for connected components)
2. Clean contaminated face embeddings (remove faces with sim < 0.35 to cluster)
  3. Clean contaminated body embeddings (iterative median-based outlier removal, 0.50 threshold)
4. Sweep orphaned MinIO crops
5. Classify staff (duration + distinct days)
- Steps 2-3 run numpy in `asyncio.to_thread()` to avoid blocking the event loop at 1k+ persons

**Face re-extraction / deletion** (every 20 min, SEPARATE PROCESS via systemd timer):
- For persons left with 0 face embeddings after contamination cleanup:
  1. Download track crops from MinIO
  2. Run InsightFace on each crop to extract a face
  3. If face found → store normalized embedding
  4. If NO face found in ANY crop → DELETE the person entirely (orphan tracks, events, billing)
- Runs as `danger/reextract_or_delete_faceless.py` via `reextract-faces.timer`
- Separate process to avoid loading InsightFace (1.5GB GPU) in the API event loop

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
| Gender (SIGLIP2 face + margin) | `siglip2_analyzer.py` (`SIGLIP2_GENDER_MARGIN_DELTA`) |
| Age (InsightFace genderage) | `insightface_analyzer.py` (`estimate_age_from_crop`, age on detections) |
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
| `danger/clean_contaminated_embeddings.py` | Remove contaminated face AND body embeddings (iterative median-outlier removal, 0.35 face / 0.50 body). `--apply` / `--face-only` / `--body-only`. |
| `danger/normalize_face_embeddings.py` | One-time migration: L2-normalize all existing face embeddings in DB |
| `danger/reextract_or_delete_faceless.py` | Re-extract faces from track crops or delete faceless persons (runs via systemd timer) |
| `danger/diagnose_persons.py` | Deep-dive diagnostic: cross-person face/body sim, track metadata, contamination check |
| `danger/verify_crops.py` | Re-compute embeddings from MinIO crops to verify stored values |
| `danger/measure_body_reid.py` | Measure OSNet same/diff body sim distributions from fresh crop embeddings |
| `danger/recompute_body_embeddings.py` | Recompute all person_embeddings from MinIO crops with fixed OSNet weights + rebuild IVFFlat index |
| `danger/uncontaminate_tracks.py` | Disassociate tracks whose face contradicts assigned person |
| `danger/delete_persons.py` | Hard-delete contaminated PersonIdentity + related tracks (dry-run default; `--apply` / `--ids`). NULLs event/billing FKs; MinIO via sweep. |
| `danger/cleanup_same_camera_overlap.py` | Same-cam concurrent-track pollution fix. Orphans conflicting tracks for **staff and visitors** (keep person + primary cluster). Never full-deletes visitors. Dry-run default; `--apply` / `--ids` / `--min-overlap-seconds`. |
| `danger/cleanup_mixed_identity_tracks.py` | Sequential mix cleanup: temporal primary cluster + duration-weighted gender majority + optional `--with-faces` cluster fit. Orphan only. |
| `danger/merge_staff_reattach_duplicates.py` | Historical backfill for staff reattach: non-staff fragments → is_staff if within RECENT_WINDOW + body_median ≥ 0.55 + face_min 0.20. Dry-run default; `--apply` / `--ids` / `--staff-ids`. |
| `danger/merge_recent_window_duplicates.py` | Historical backfill — merge same-visit duplicates within 5-min window (combined face/body rule, contamination-gated absorb). Dry-run-first. |
| `danger/reset_tracking_data.py` | Full data reset (preserves config) |
| `danger/test_mivolo.py` | Offline MiVOLO only (not used live) |
| `danger/fix_demographics_oneshot.py` | Backfill IF median age all + known gender F→M list |
| `danger/sweep_siglip2_gender_margin.py` | Calibrate `SIGLIP2_GENDER_MARGIN_DELTA` |
| `danger/test_insightface_age.py` | Offline IF genderage age distribution |
| `danger/compare_gender_models.py` | SigLIP2 vs IF genderage error rates |
| `danger/test_siglip2.py` | Test SigLIP2 face-only on specific persons |
| `danger/test_siglip2_body.py` | Test SigLIP2 body similarity on specific persons |
| `danger/test_deepface.py` | Test DeepFace on specific persons (0% accuracy — deprecated) |

---

## Known Issues

1. **OSNet body ReID — ROOT CAUSE FOUND AND FIXED (2026-07-09)** — The originally documented "overlapping same/diff distributions (0.58-0.83 vs 0.10-0.40)" were caused by **missing ReID-finetuned weights**. The configured `OSNET_MODEL_PATH = "models/osnet_x1_0.pth"` did not exist on disk. `FeatureExtractor` silently fell back to `pretrained=True`, loading ONLY the ImageNet backbone (`osnet_x1_0_imagenet.pth` from `~/.cache/torch/`). The ReID `fc` embedding head (512-dim) stayed **randomly-initialized**, producing non-discriminative embeddings that encoded scene appearance (background, lighting) rather than person identity. **Fix:** Downloaded `osnet_x1_0_msmt17_combineall` checkpoint to `models/osnet_x1_0.pth`. Added startup guard in `osnet_extractor.py` that refuses to start if the model file is missing AND verifies the `fc.*` keys loaded from the checkpoint. **New distributions (MSMT17 weights, 32 multi-camera persons + 60 concurrent pairs):** same-person median=0.680 (p10=0.393, p90=0.845), diff-person median=0.386 (p10=0.294, p90=0.534). Best F1 threshold=0.49. Thresholds retuned: `REID_MATCH_THRESHOLD` 0.85→0.50, `BODY_CONTAMINATION_THRESHOLD` 0.60→0.50. All 292 existing `person_embeddings` recomputed from MinIO crops + IVFFlat index rebuilt.

2. **SigLIP2 face-only misses culturally distinctive attire** — A woman in a saree was missed by face-only (37-40% F prob). Mitigated by body crop integration (3× weight, 79% confidence with body context). Still may miss edge cases.

3. **ArcFace cross-angle degradation** — Cross-camera same-person sim can drop to 0.20. Mitigated by 0.40 dedup threshold in periodic job.

4. **Face contamination during close-proximity** — When 2+ people stand shoulder-to-shoulder, faces from adjacent persons can be matched to wrong tracks. Mitigation: full-frame face detection + global optimal assignment (not greedy per-track) + original bbox membership (no expansion) + temporal continuity scoring + track-level contamination gate (normalized, ≥2 prior faces) + store-time contamination gate (normalized, 0.35 threshold) + dedup face cleanup.

5. **MinIO sweep deletes old crops** — Body crops from merged/deleted persons cascade-delete. The sweep correctly cleans them up. Face crops from the `_prune_face_embeddings` path are protected by the deferred-deletion + reference check.

6. **Faceless persons after contamination cleanup** — If all face embeddings are removed by the dedup cleanup, the person has no face. The separate `reextract_or_delete_faceless.py` process (systemd timer, every 20 min) re-extracts faces from track crops or deletes the person entirely if no face is found. Tracks are orphaned (NULL person_identity_id) — they'll be re-identified on the next visit.

7. **Face embedding normalization** — InsightFace buffalo_l embeddings are NOT L2-normalized (norms 12-27). All in-memory comparisons use `_face_sim()` which normalizes both vectors. New embeddings are normalized at store time. Existing DB embeddings were migrated via `normalize_face_embeddings.py`.

8. **pgvector returns string from raw SQL** — When querying pgvector columns via `text()` (raw SQL), the embedding comes back as a string `"[0.1, 0.2, ...]"` in some asyncpg codec configurations. Must use `eval()` before `np.array()`: `np.array(eval(raw), dtype=np.float32)` if `isinstance(raw, str)`. The ORM handles this correctly via the pgvector type. Places fixed: `_store_face_embedding`, `_store_embedding`, `_clean_contaminated_face_embeddings`, `_clean_contaminated_body_embeddings`, `_absorb_face_embeddings`, `_absorb_body_embeddings`.

9. **IVFFlat index must be rebuilt after normalizing embeddings** — The `normalize_face_embeddings.py` migration modifies all embedding values. Without `REINDEX INDEX idx_person_face_embeddings_embedding`, the IVFFlat cluster assignments are stale, causing the LATERAL query to return non-deterministic results. The migration script and `reset_tracking_data.py` now both run `REINDEX` + `VACUUM ANALYZE` automatically.

10. **Dedup merge must ABSORB embeddings, not cascade-delete** — Previously, when merging person B into A, the dedup job CASCADE-DELETED B's face and body embeddings. This caused an infinite duplicate cycle: merge destroys alternate angle, next visit from that angle creates duplicate, merge destroys it, repeat. Now embeddings are MOVED to the winner (up to caps), and gender is re-voted from all reassigned tracks.

11. **Multiple raw `np.dot()` bugs on face embeddings** — Six locations used `np.dot()` on unnormalized InsightFace embeddings (norm ~20), comparing against thresholds designed for cosine similarity (0.25-0.48). This made all face gates effectively dead code. All fixed with `_face_sim()` or explicit normalization.

12. **Background jobs must run in separate process** — The dedup job's heavy DB queries, MinIO batch deletions, and numpy computations blocked the API event loop for 1-2 minutes. Moved all APScheduler jobs to `retail-ai-worker.service` (separate process). The API server now only handles HTTP + camera workers.

13. **Dedup union-find `find()` reverted every union (FIXED 2026-07-09)** — The iterative `find()` in `deduplicate_persons()` (`app/modules/jobs/tasks.py:232-236`) had a broken path-compression line:

   ```python
   # BROKEN — reverts every union
   def find(x: str) -> str:
       while parent.get(x, x) != x:
           parent[x] = parent.get(parent.get(x, x), x)  # ← BUG
           x = parent.get(x, x)
       return x
   ```

   After `union(A, B)` sets `parent[A] = B`, calling `find(A)` computes `parent.get(parent.get(A, A), A)` = `parent.get(B, A)` = `A` (B is a root, not a dict key, so `dict.get` defaults to `A`). This overwrites `parent[A] = A` — **reverting the union**. Every component collapsed to size 1, so `len(members) < 2` (line 269) skipped all merges.

   **The failure was completely silent:** the job ran every 10 min, logged `"found N duplicate pair(s) — merging..."`, built the union-find, then produced 0 merges with no error. No exception, no warning. The only symptom was duplicate `person_identity` records persisting indefinitely despite the job "running successfully". **The dedup job had never successfully merged anyone since this code was written.**

   Fixed with correct recursive path compression:
   ```python
   def find(x: str) -> str:
       p = parent.get(x, x)
       if p != x:
           parent[x] = find(p)
       return parent.get(x, x)
   ```

   Verified: first run after fix merged `3ce4e929 → d9ab4071` (face sim 0.4254). **Restart required:** running worker processes keep the old buggy code in memory until `sudo systemctl restart retail-ai-worker.service`.

14. **Face-only dedup threshold gap: cross-angle same-person pairs missed + diff-person false merges (OPEN)** — The dedup job uses **face embeddings only** (threshold 0.40, `DEDUP_THRESHOLD` in `deduplicate_persons()`). Body ReID (`person_embeddings`, OSNet) is NOT used for merging — only for contamination cleanup. An empirical analysis on July 9 2026 (16 persons, 09:15–09:30 window, BEFORE the find() fix was applied) revealed two problems:

   **(a) Same-person pairs that fall below 0.40 (missed merges):**
   ```
   d9ab4071  ↔ da526bb8   face=0.3883  body=0.7505   (same person, NO merge)
   3ce4e929  ↔ da526bb8   face=0.3655  body=0.7304   (same person, NO merge)
   2ce51a66  ↔ da526bb8   face=0.1568  body=0.7460   (same person, NO merge)
   2ce51a66  ↔ 3ce4e929   face=N/A     body=0.7212   (same person, NO merge)
   ```
   This is ArcFace cross-angle degradation (issue #3). da526bb8 needs a better face angle on a future visit; 2ce51a66 has a contaminated face (issue #15).

   **(b) Diff-person pairs at face ≥ 0.40 (false merges — these WOULD be merged by the fixed dedup):**
   ```
   6fe53c77  ↔ ac252e23   face=0.6085  body=0.8958   (DIFFERENT people)
   1abd845e  ↔ 9b6053ac   face=0.5885  body=0.8118   (DIFFERENT people)
   1abd845e  ↔ f08e6d35   face=0.5846  body=0.7841   (DIFFERENT people)
   9b6053ac  ↔ f08e6d35   face=0.5449  body=0.7584   (DIFFERENT people)
   1abd845e  ↔ 2ce51a66   face=0.4661  body=0.7845   (DIFFERENT — 2ce51a66 contaminated)
   ```
   ~5-6 false merges per busy time window. These are likely **contaminated or misassigned face embeddings** (faces from adjacent persons matched to wrong tracks during close-proximity, issue #4), NOT genuinely similar faces of different people.

   **Similarity distributions (16 persons, July 9 09:15–09:30 window):**
   ```
   FACE sim:
     same-person:  n=4   min=0.157  max=0.425  avg=0.334  median=0.388
     diff-person:  n=63  min=0.073  max=0.609  avg=0.204  median=0.164

   BODY (OSNet) sim:
     same-person:  n=5   min=0.603  max=0.751  avg=0.710  median=0.730
     diff-person:  n=93  min=0.593  max=0.896  avg=0.755  median=0.763
   ```
    Note: diff-person body sims (median 0.763) **exceed** same-person body sims (median 0.730) — see issue #16. **These measurements were taken with broken OSNet weights (ImageNet backbone + random fc head). With MSMT17 weights, same-person median=0.680 >> diff-person median=0.386 — see issue #16 resolution.**

   **Body-confirmation experiment (REJECTED with broken weights — RE-EVALUATE with MSMT17 weights):** A combined rule `face >= 0.40 OR (face >= 0.35 AND body >= T)` was tested against live data. The premise was that body ReID could confirm borderline face pairs (0.35–0.40) that face alone misses. **It failed** because OSNet body ReID was non-discriminative (the model was running on ImageNet backbone + random fc head — see issue #16 for the root cause). Threshold sweep (with BROKEN weights):
   ```
   T=0.70:  same-pairs-merged=3/4   diff-FALSE-merged=6
   T=0.72:  same-pairs-merged=3/4   diff-FALSE-merged=6
   T=0.74:  same-pairs-merged=2/4   diff-FALSE-merged=6
   T=0.75:  same-pairs-merged=2/4   diff-FALSE-merged=6
   T=0.76:  same-pairs-merged=1/4   diff-FALSE-merged=6
   ```
   All 6 false merges come from the `face >= 0.40` branch — body ReID cannot prevent them. The body branch only catches 1-3 same-person pairs at low T, at the cost of more false merges. **NOTE (2026-07-09): This conclusion was based on broken OSNet weights. With the MSMT17 checkpoint now loaded (issue #16 resolved), body ReID IS discriminative (same-person median=0.680, diff-person median=0.386). This experiment should be re-run with the fixed weights — body ReID may now successfully confirm borderline face pairs.**

   **Current accepted state (as of 2026-07-09):**
   - Threshold stays 0.40 (face-only)
   - ~5-6 false merges per busy time window are accepted
   - Safety net: face contamination cleanup (0.35, ≥2 faces) + `reextract_or_delete_faceless.py` timer (every 20 min) may correct contaminated embeddings over time
   - da526bb8 (face 0.388) and 2ce51a66 (contaminated face) remain as separate duplicate registrations

   **Future investigation needed:** Why do diff-person face sims reach 0.47–0.61? This is far above the documented different-person range (0.10–0.30 in CONTEXT.md thresholds). Likely root cause is contaminated/misassigned face embeddings from close-proximity frame assignment (issue #4), NOT genuinely similar faces. If the face-to-track assignment contamination can be reduced, the false-merge problem shrinks. Run `danger/diagnose_persons.py` on the false-merge pairs (6fe53c77/ac252e23/1abd845e/9b6053ac/f08e6d35) to confirm contamination vs genuine similarity.

15. **Single contaminated face embedding causes false dedup merge (PARTIALLY MITIGATED 2026-07-09)** — `2ce51a66` (same person as `d9ab4071`) had a single face embedding that was **contaminated** — it belonged to a different person matched to the wrong track during close-proximity (issue #4). Its face sim to its true cluster was 0.066–0.187 (different-person range), but its sim to a stranger `1abd845e` was 0.4661 (above 0.40). At face threshold 0.40, the dedup job (once the find() bug was fixed) would have merged 2ce51a66 into the **WRONG person**.

   **The false 6-person component that was prevented:**
   2ce51a66's contaminated face linked it (via 1abd845e at sim 0.4661) into a connected component of 6 strangers, all from 08:00–08:53 — a completely different time window:
   ```
   Component (size=6, root=b277bc00):
     2ce51a66  (09:20, the contaminated target)
     1abd845e  (08:03, stranger — linked via face 0.4661)
     b277bc00  (08:53, stranger)
     9b6053ac  (08:09, stranger — highest face_score 0.869, would WIN)
     ed6838d4  (08:09, stranger)
     f08e6d35  (08:00, stranger)
   ```
   If this had merged, winner = 9b6053ac (score 0.869). 2ce51a66's tracks, visit count, and the contaminated face would have been absorbed into a stranger — **permanently losing the real person's visit data** and polluting the stranger's identity.

   **Manual fix applied:** Deleted the single contaminated face embedding from 2ce51a66 before the dedup run. 2ce51a66 now has 0 faces, 2 body embeddings, 1 track session — intact. The `reextract_or_delete_faceless.py` timer (every 20 min, separate process) will download 2ce51a66's track crops from MinIO and run InsightFace to re-extract a clean face. If a clean face is found → stored, 2ce51a66 may match d9ab4071 on the next dedup cycle (if sim ≥ 0.40). If no face is found in ANY crop → 2ce51a66 is deleted entirely, tracks orphaned (NULL person_identity_id) for re-identification on next visit.

   **The dedup contamination cleanup's blind spot:**
   `_clean_contaminated_face_embeddings()` (`tasks.py:429`) detects contaminated faces by comparing each embedding against the person's other faces — it requires **≥2 face embeddings** per person to identify outliers. A person with a single contaminated face is **invisible** to the cleanup: there's nothing to compare against. This is a structural gap:

   - Persons with ≥2 faces: contamination detectable (outlier removed if sim < 0.35 to cluster majority)
   - Persons with exactly 1 face: contamination undetectable by the cleanup job
   - Single-face contamination only surfaces when it causes a **false dedup merge** (as it did here) or when the `reextract_or_delete_faceless.py` timer re-extracts from crops

   **Mitigation status:** The immediate false-merge was prevented (manual face deletion). The structural gap remains — any single-face person with a contaminated embedding will false-merge into whoever the contaminated face matches. The `reextract_or_delete_faceless.py` timer is the long-term safety net but runs on a 20-min cycle and requires GPU (separate process).

16. **OSNet body ReID distributions inverted vs documented ranges (RESOLVED 2026-07-09)** — Issue #1 documented OSNet same-person sims 0.83–0.93 and diff-person sims 0.54–0.63. The July 9 2026 empirical analysis (16 persons, 09:15–09:30 window) showed the **opposite pattern**: diff-person body sims **exceed** same-person body sims.

    ```
    BODY (OSNet) sim distributions (16 persons, July 9 2026, BROKEN weights):
      same-person:  n=5   min=0.603  max=0.751  avg=0.710  median=0.730
      diff-person:  n=93  min=0.593  max=0.896  avg=0.755  median=0.763
    ```

    **Root cause:** The `OSNET_MODEL_PATH` file (`models/osnet_x1_0.pth`) did not exist. `FeatureExtractor` fell back to `pretrained=True`, loading the ImageNet backbone only. The `fc` ReID embedding head was randomly-initialized. Embeddings encoded scene appearance (pharmacy background, lighting, color histograms) rather than person identity — so different people in the same pharmacy scene looked MORE similar than the same person across cameras/angles. This is why diff-person sims exceeded same-person sims: the dominant signal was "same scene" not "same person".

    **Resolution (2026-07-09):**
    - Downloaded `osnet_x1_0_msmt17_combineall` checkpoint (17.3 MB, from HuggingFace `kaiyangzhou/osnet`) to `models/osnet_x1_0.pth`.
    - Added startup guard in `osnet_extractor.py:_load_model` — refuses to start if model file missing; verifies `fc.*` keys loaded via `_verify_reid_weights_loaded()`.
    - Recomputed all 292 existing `person_embeddings` from MinIO crops with the fixed model (`danger/recompute_body_embeddings.py`). IVFFlat index rebuilt.
    - Re-measured distributions (32 multi-camera persons + 60 concurrent diff-camera pairs, `danger/measure_body_reid.py`):
    ```
    BODY (OSNet) sim distributions (MSMT17 weights, July 9 2026):
      same-person:  n=88   min=0.309  max=0.923  p10=0.393  p50=0.680  p90=0.845  mean=0.641
      diff-person:  n=145  min=0.214  max=0.900  p10=0.294  p50=0.386  p90=0.534  mean=0.399
    ```
    Same-person median (0.680) >> diff-person median (0.386). Distributions properly separated. Best F1 threshold=0.49 (F1=0.793). Overlap region [0.39, 0.53] — much narrower than the inverted distribution.
    - Retuned thresholds: `REID_MATCH_THRESHOLD` 0.85→0.50, `BODY_CONTAMINATION_THRESHOLD` 0.60→0.50.
    - The body-confirmation dedup experiment (issue #14) can now be re-evaluated — body ReID IS discriminative with proper weights.

    **Previous incorrect hypotheses (now superseded):** "Pharmacy customers wear similar clothing", "low-resolution CCTV degrades OSNet", "checkpoint may not be the best variant". The actual problem was far more fundamental: no ReID training was present in the running weights at all.

17. **Dedup absorb had NO contamination gate — winner identities silently polluted (FIXED 2026-07-09)** — `_absorb_face_embeddings` / `_absorb_body_embeddings` (`jobs/tasks.py`) moved ALL of a loser's embeddings into the winner during a dedup merge with only a ">0.95 duplicate-angle" check — zero cluster-fit validation. Every false merge (issue #14 documented ~5-6 per busy window) directly injected a stranger's face+body into the winner. Staff identities were hit hardest (they win most merges via highest face_score/visit_count). On 2026-07-09 an empirical audit found 2 persons with severe face contamination (min pairwise face sim 0.14-0.24, deep "different person" range): `9b6053ac` (staff, 94 visits, clean 2-person split with a borderline bridge) and `cf793282` — both exactly the issue-#14 false-merge participants. **Fix:** added a contamination gate to both absorb functions — a loser embedding is only moved if its median similarity to the winner's *existing* cluster clears the threshold (0.35 face / 0.50 body); otherwise it is DROPPED (logged as rejected) instead of moved. Body uses median (≥2 winner embs) so a single borderline-bridge winner face cannot chain a stranger in.

18. **Face contamination cleanup used greedy single-linkage clustering — chained contamination through bridges (FIXED 2026-07-09)** — `_clean_contaminated_face_embeddings` (`jobs/tasks.py` and `danger/clean_contaminated_embeddings.py`) used "keep if similarity ≥0.35 to ANY already-kept embedding". Hand-verified this FAILED on both real contaminated identities: a borderline "bridge" embedding (similar ~0.35-0.49 to BOTH real people) let contamination chain straight through undetected. **Fix:** replaced with the same **iterative median-outlier removal** already used for body (`_clean_contaminated_body_embeddings`): repeatedly remove the embedding whose median similarity to the rest is lowest, until the worst remaining median ≥ threshold or the cluster drops below 2 (faces) / 3 (bodies). Aggressive-reject tuning: any embedding below the median bar is removed (no leniency) — deleting a borderline same-person face is recoverable via `reextract_or_delete_faceless.py`; storing a contaminated one is not. The standalone `danger/clean_contaminated_embeddings.py` was rewritten to use the shared median approach and clean BOTH face and body (with `--apply` / `--face-only` / `--body-only`). Applied 2026-07-09: removed 2 faces + 2 bodies; face contamination below gate went 2→0.

19. **Dedup merge loop aborted the whole batch on one failure (FIXED 2026-07-09)** — the per-merge try/except did `await db.rollback(); return`, so a single bad pair (e.g. a SQL error on one loser) aborted ALL remaining merges AND skipped the downstream contamination-cleanup / sweep / staff-classification steps for the entire 10-min cycle — a poison-pill that could block dedup indefinitely. **Fix:** each merge now runs in its own SAVEPOINT (`async with db.begin_nested()`); a failure rolls back ONLY that merge and the batch continues. MinIO deletion of merged losers' crops is deferred to AFTER the commit so a savepoint rollback can never delete a MinIO object whose DB reference survived.

20. **Recent-window two-tier matching (NEW 2026-07-09, median check + last_seen_at fix 2026-07-10)** — Same physical person seen briefly per camera (median track-session duration in this store is **13 seconds**; 40% of persons are single-visit fragments) was repeatedly registered as separate identities because: (a) cross-angle face best-pair falls in [0.35, 0.40) just under the strict 0.40 face match, and (b) the 2-of-3 body consensus gate is structurally impossible when the store is quiet (only 1 candidate exists, so consensus can never reach 2). **Fix:** two-tier matching. Within `RECENT_WINDOW_MINUTES=5` of a candidate's `last_seen_at` (changed from `first_seen_at` 2026-07-10 — a staff member who arrived 6h ago but was tracked 30s ago IS recent): face accepts at relaxed `0.35` (best cross-pair, matching the dedup job's MAX() semantics), AND a body single-candidate override accepts at median `0.55` (≥2 bodies each side, non-overlapping tracks on the *same* camera, faces don't contradict at 0.25). Body-only valid ONLY within the window (clothing constant → body reliable; small candidate pool). Outside: strict 0.40 face / 0.50 body + 2-of-3 consensus unchanged. **Face median check (added 2026-07-10):** when recent face best-pair is in grey zone [0.35, 0.40), require median of ALL cross-pairs ≥ `FACE_MATCH_MEDIAN_THRESHOLD` (0.30). Calibration (2026-07-10, 196 persons): same-person min median=0.401, diff-person p50=0.200. At 0.30: 0% same-person rejected, 97.5% diff-person rejected. Only checked when ≥3 total cross-pairs. Catches single lucky crops from different people. **Critical safety findings:** (1) body-alone is NOT trustworthy even in a short window — uniformed staff (`b33a2586`↔`d01adabc` body median 0.586, confirmed different people) are indistinguishable by OSNet. Non-contradiction gate is the safety net. (2) `_is_recent` must use `last_seen_at` not `first_seen_at` — a staff member who arrived 6h ago but was tracked 30s ago IS recent; `first_seen_at` incorrectly blocked the relaxed face path causing duplicate registrations (e.g. `86e763dd` split from `fe34af9d` after a 10-min tracking gap). Feature-flagged via `ENABLE_RECENT_WINDOW_MATCHING`. Camera-aware overlap: only same-camera overlap blocks a merge; cross-camera overlap is expected (same person visible on entry+counter simultaneously). **Critical safety finding from live data:** body-alone is NOT trustworthy even in a short window — a "body-chameleon" person (`f9392afa`) matched 5+ strangers at body_median 0.6-0.7 in the same 10-min span while their faces contradicted (<0.20). The non-contradiction gate (faces must not contradict at 0.25) is what prevents those false merges. Feature-flagged via `ENABLE_RECENT_WINDOW_MATCHING`. **Camera-aware overlap (2026-07-09 fix):** the backfill's overlap gate was originally global; this blocked true same-person pairs visible on entry+counter cameras simultaneously. Now the gate only blocks same-camera overlap (two different people cannot occupy the same camera at the same time); cross-camera overlap is allowed. Validated against the `488c3308`+`ac865cdf` pair (cross-camera overlap at 18:07, same person — face_max 0.395).

21. **Historical backfill merge script (NEW 2026-07-09)** — `danger/merge_recent_window_duplicates.py` applies the same combined recent-window rule to ALL historical persons (not just since last restart) as a one-off cleanup. Uses the FIXED union-find + FIXED contamination-gated absorb functions so it does not re-inject contamination. Must run AFTER contamination cleanup (Step B). Dry-run-first with full evidence per pair. Applied 2026-07-09: 100→95 persons, 5 merges (incl. the acid-test triple). Body-only path requires non-contradiction: a faceless side OR face_max ≥ 0.30, which collapsed an initial 20-pair candidate set (that chained 6 strangers via body-only with face_max 0.07-0.18) down to 5 defensible merges.

 22. **Worker process logging invisible (FIXED 2026-07-09)** — `app/worker.py` (the separate background-worker process running APScheduler) never called the loguru sink setup that `app/main.py` does, so dedup-job activity was only in `sudo journalctl` (which the `retaileye` service user cannot read without sudo). **Fix:** extracted `app/logging_config.py::setup_logging()` (shared, idempotent); `worker.py` now calls it so dedup runs appear in `logs/ai_processing.log` alongside the API server.

 23. **Body crop padding for ReID (SHIPPED 2026-07-10 — default 0%)** — Body crops for OSNet now pass `padding_pct=settings.BODY_CROP_PADDING_PCT` (default **0.0** = tight YOLO box) for both initial track crop and ReID crop. Face crops remain at 0.30. Tunable via env without code change (e.g. `BODY_CROP_PADDING_PCT=0.05`). Existing MinIO body crops/embeddings were generated at the old 10% padding — historical re-embed via `danger/recompute_body_embeddings.py` is optional; new frames only use 0%. If same-person body medians drop after deploy, raise padding slightly and re-measure with `danger/measure_body_reid.py`.

 24. **Face-to-track misassignment under occlusion (MITIGATED Phase 1 2026-07-10)** — Root intake path for contamination is face↔track assignment under side-by-side / occlusion. **Phase 1 ship:** (1) pairwise IoU ≥ `OCCLUSION_IOU_THRESHOLD` (0.10) marks `track.is_occluded`; (2) immature tracks (`good_face_count < 2`) require face centre in upper `FACE_ASSIGN_UPPER_BODY_FRAC` (0.45) of body + ≥ `FACE_ASSIGN_MIN_OVERLAP` (0.70) of face area inside body + score ≥ `FACE_ASSIGN_MIN_SCORE_IMMATURE` (0.35); (3) if two tracks compete for the same face with top2/top1 ≥ `FACE_ASSIGN_AMBIGUITY_RATIO` (0.85), assign face to **no one** that frame; (4) Hungarian bipartite assignment (`ENABLE_HUNGARIAN_FACE_ASSIGN`, scipy) replaces pure greedy; (5) continuity bonus + `last_face_center` update **disabled while occluded**; (6) `SKIP_BODY_REID_WHEN_OCCLUDED` drops OSNet on occluded frames (face-only accum OK). Remaining residual: interior mature-track swaps with clear faces + no IoU overlay; store/dedup contamination gates + single-face blind spot (issue #15) still apply. **Future:** same-camera temporal-overlap hard block in `decide_identity` (Phase 3); optional pose-aligned body crop.

---
## Process Architecture

```
+-----------------------------+   +------------------------------+   +------------------------------+
| retail-ai.service           |   | retail-ai-worker.service     |   | reextract-faces.timer        |
| (API Server, ~3.4 GB GPU)   |   | (Background Jobs, ~100 MB)   |   | (GPU Worker, oneshot)        |
|                             |   |                              |   |                              |
| - FastAPI HTTP endpoints    |   | - deduplicate_persons (10m)  |   | - Re-extract face from crops |
| - Camera workers (YOLO,     |   | - close_stale_tracks (5m)    |   | - Delete faceless persons    |
|   InsightFace, OSNet, etc.) |   | - probe_cameras (2m)         |   | - Every 20 min               |
| - Stream broadcasters       |   | - daily_analytics (00:15)    |   | - Loads InsightFace (1.5GB)  |
| - In-memory track state     |   | - storage_cleanup (02:00)    |   | - Frees GPU on exit          |
| - NO background jobs        |   | - NO GPU, NO camera          |   |                              |
+-----------------------------+   +------------------------------+   +------------------------------+
```

All three share PostgreSQL + MinIO. The worker process handles the heavy dedup/sweep so the API server never freezes. The face re-extraction script loads InsightFace as a separate process to avoid GPU memory contention.

---
## Current State (as of last update)

- **Gender:** SigLIP2 face-only, 7+7 prompts, `SIGLIP2_GENDER_MARGIN_DELTA=0.5`, `SIGLIP2_USE_BODY_FOR_GENDER=False`. Track mean margin across faces → gender. Dedup merge still re-votes person gender from track-level genders.
- **Age:** InsightFace buffalo_l `genderage` (modules: detection+recognition+genderage); track `age_samples` → median. IF under-reports true young children (often 22–28); accepted limitation for now (no FairFace / geometry child gate without recalibration).
- **Backfill (applied 2026-07-11):** `danger/fix_demographics_oneshot.py` — re-aged all face IDs via IF median; set 8 known F→M ids to F. Dry-run default; `--apply` writes.
- **~~Age MiVOLO~~ / ~~body gender×3~~:** removed from live worker Jul 2026. MiVOLO weights under `models/mivolo/` offline only.
- **Body ReID:** OSNet with MSMT17 checkpoint. Live match uses **median body sim** (n_bodies≥2) ≥0.50 strict / ≥0.55 recent; top-2 ambiguity gap `BODY_MATCH_AMBIGUITY=0.03`. Old unique-person 2-of-3 vote gate was dead (FIXED 2026-07-17).
- **Staff detection:** Running every 10 min, 30-min duration threshold
- **Purchase counts:** Deduplicated per person, staff excluded
- **Face detection:** Full-frame, 1 call per frame, padded resize
- **Face-to-track assignment:** Hungarian (greedy fallback) + immature geo harden + ambiguous reject + occlusion IoU flag; continuity disabled while occluded
- **Body crop padding:** `BODY_CROP_PADDING_PCT=0.0` (config); skip OSNet when occluded
- **Staff reattach:** face/body customer paths fail → body median ≥0.70 + recent + `is_staff` + **required face** sim ≥0.30 → reattach; mid face (0.30–0.35) dropped from gallery; faceless rejected (raised after b33 uniform pollution)
- **Face embedding normalization:** All face embeddings L2-normalized (norm=1.0) at store time + existing DB migrated
- **Face comparison:** All in-memory face sims use `_face_sim()` (normalizes before dot product). pgvector string format handled with `isinstance` check.
- **Face contamination gate (track-level):** Normalized cosine sim, >=2 prior faces, 0.35 threshold
- **Face contamination gate (store-time):** Normalized cosine sim to cluster, 0.35 threshold
- **Face dedup (track):** Normalized cosine sim > 0.95 = duplicate angle (skip)
- **Body contamination gate (store-time):** Median cosine sim to existing cluster, >=3 embeddings, 0.50 threshold
- **Face/body contamination cleanup (dedup job + danger script):** Both use iterative median-outlier removal (face 0.35, body 0.50). Aggressive-reject. Previously face used greedy single-linkage that chained contamination through bridges — FIXED (issue #18).
- **Dedup absorb (contamination-gated):** `_absorb_face_embeddings`/`_absorb_body_embeddings` now DROP a loser embedding if its median similarity to the winner's existing cluster is below threshold (was: moved ALL with no check — issue #17). Stops false-merge contamination injection.
- **Dedup job:** 0.40 merge threshold + face cleanup (0.35, median) + body cleanup (0.50, median). Per-merge SAVEPOINT isolation — one bad pair no longer aborts the batch (issue #19). Absorbs embeddings (contamination-gated) + re-votes gender. Runs in separate worker process. union-find `find()` recursive path compression (fixed).
- **Recent-window matching:** Live engine, 5-min via `last_seen_at`. Face grey 0.35 + median cluster gate; body recent 0.55 median. `match_tier` drives CASE1/2 accept thr so grey face actually attaches (FIXED 2026-07-17). Outside: face 0.40 / body median 0.50 + ambiguity.
- **SAME_CAM after reject:** No create-new (leave unassigned). Prevents staff/visitor clone factory when concurrent track blocks attach (FIXED 2026-07-17).
- **MATCH STALE / store FK (P5 FIXED 2026-07-17):** Person may vanish between search and store (reextract delete). Exist check + `FOR SHARE` + SAVEPOINT attach; create suppressed on stale; no create-on-exception poison. `IDENTITY_ADVISORY_LOCK_KEY=1001` shared with `reextract_or_delete_faceless`.
- **Create gates INFO:** `[CREATE BLOCKED] reason=NO_FACE|LOW_SCORE|LOW_GOOD_COUNT`.
- **Historical backfill:** `danger/merge_recent_window_duplicates.py` — one-off cleanup applying the combined recent-window rule to all historical persons (issue #21). Applied 2026-07-09 (100→95 persons).
- **Face re-extraction/deletion:** Separate process via systemd timer, every 20 min; **must** take advisory lock 1001 before DELETE
- **Multi-angle face accumulation:** Working — tracks accumulate 5-11 unique face angles, 2-5 stored per person
- **Background jobs:** Separate `retail-ai-worker.service` process (no GPU, ~100 MB)
- **MinIO:** Deferred deletion, sweep every 10 min
- **GPU:** ~4.3–4.5 GB (InsightFace+genderage ~1.5GB + YOLO ~300MB + OSNet ~100MB + SigLIP2 ~1.4GB + Torch overhead; MiVOLO no longer loaded)
- **Purchase rule:** Billing Counter Interaction, `dwell_threshold_seconds=50`, cooldown 600s, zone Counter on Apollo counter. Analytics = `COUNT(DISTINCT person_identity_id)` excluding `is_staff`.

---

## 25. Live identity match dead-paths (FIXED 2026-07-17)

**Root causes (found via logs + code audit, connected to purchase undercount / null-id tracks):**

1. **Body 2-of-3 consensus was structurally impossible** — `_search_similar` returns unique persons; votes per person never ≥2. Failure path set `best_candidate=None`, which **killed** the recent body override. Logs pre-fix: 0× `[Body Consensus]` / 0× `[Body RECENT single]`.
2. **Recent face grey zone logged then failed** — Step1 accepted face at 0.35–0.40, then CASE1 required `FACE_MATCH_THRESHOLD=0.40` → create/miss.
3. **SAME_CAM reject → create clone** — including high face_sim to known staff (staff fragment factory).

**Fix (`identity_decision_engine.py`):**
- Body match on **gallery median** (n≥2), strict 0.50 / recent 0.55; `BODY_MATCH_AMBIGUITY=0.03`.
- `match_tier` + `_accept_threshold()` for face_strict / face_recent / body / body_recent / staff_reattach.
- `same_cam_blocked` suppresses `_create_new_person`.
- Create blocks at INFO with reason codes.

**Post-restart smoke (≈46 min):** Body Match 143×; Face Match RECENT 202×; SAME_CAM create suppressed 72×; Matched logs all include `tier=`.

**Tests:** `tests/test_identity_decision_p0_p3.py`. **Deploy:** restart `retail-ai.service`.

---

## 26. Identity store FK race + session poison (FIXED 2026-07-17 — P5)

**Symptom after P0 body path went live:** ~88× `Identity decision failed` ForeignKeyViolation on `person_embeddings.person_identity_id` → cascade ~2k ERROR (`Session rolled back` on face store / ReID / frame loop).

**Root cause (not permanent orphans — DB audit showed 0 orphan emb rows):**
```
T0  search JOIN person_identities → person P exists
T1  concurrent delete of P (reextract_or_delete_faceless / merge)
T2  _store_embedding flush → IntegrityError
T3  exception swallowed / create-on-fallback on poisoned session → ERROR storm
```
Cross-process concurrency: live `pg_advisory_xact_lock(1001)` did **not** cover reextract delete.

**Fix:**
1. `_person_exists(..., for_share=True)` before attach; `match_stale_blocked` → no create.
2. `_attach_embeddings` in SAVEPOINT (`db.begin_nested`); stores raise `IdentityStoreError` on FK (no silent poison).
3. Outer `decide_identity` **never** create-on-exception after fail.
4. `camera_worker` wraps `decide_identity` + extra face stores in SAVEPOINT.
5. `reextract_or_delete_faceless` takes `IDENTITY_ADVISORY_LOCK_KEY` (1001) before DELETE.

**Tests:** `tests/test_identity_persistence_p5.py`. Restart `retail-ai.service` after deploy; reextract timer picks up lock on next run.

**Invariant for future sessions:** Any process that DELETEs `person_identities` while cameras run **must** take advisory lock 1001 (or wait for off-peak). Do not call `db.rollback()` deep inside store if saved by outer SAVEPOINT — let nested CM clean up.

---

## 27. Purchase / counter dwell audit (2026-07-17 — analysis only)

**Store vs DB (non-staff DISTINCT primary purchasers):**

| Window | DB purchases | Store said | Notes |
|--------|-------------:|-----------:|------|
| 16 Jul full day | **46** | **106** | ~43% of store |
| 17 Jul →13:40 IST | **39** | **44** | close |
| 17 Jul →~15:17 | **44** | ~44 at ~2pm reported earlier | |

**Camera offline on Jul 16?** **No.** Counter + entry tracks/events every hour; **0** zero-activity 15‑min bins 08:00–24:00 IST. Under-count is not camera downtime.

**Rule at time of audit:** `Billing Counter Interaction`, dwell **50s**, cooldown 600s, zone `Counter` / BID `57250117-…`, camera Apollo counter.

**Counter zone max dwell (non-staff persons, from event metadata):**

| Cohort Jul 16 | n | avg | median |
|---------------|--:|----:|-------:|
| All in Counter zone | 88 | 82.9s | **52.3s** |
| Has BillingInteraction | 46 | 135.7s | **91.2s** |
| At counter no BI | 42 | 25.1s | **23.7s** |

**Threshold sweep (distinct non-staff with max_dwell ≥ T):** even **T=0 only reaches 88** on Jul 16 — **cannot hit store 106** by lowering dwell alone. Identity ceiling (null person_id tracks ~60–70% of sessions) is the binding constraint.

| T | Jul 16 | Jul17→13:40 |
|--:|------:|-------------:|
| 0 | 88 | 66 |
| 40 | 54 | 46 |
| 45 | 50 | **43 ≈ store** |
| 50 | 48 | 42 |

**Operational takeaway:** keep dwell **45–50s** for purity (today aligns). Closing Jul 16 106 gap needs **identity coverage** (P0–P5), not thr→0. Store 106 is likely **bills**, DB is **unique people once** — conceptual mismatch remains.

**Danger:** `danger/backfill_billing_dwell.py` for threshold backfill only after rule change + explicit apply.

---

## 28. Null billing person_id after deferred identity (FIXED 2026-07-29)

**Symptom:** Guest at counter long enough for `billing_interaction` (dwell≥50s) but analytics undercount — BI row exists with `person_identity_id=NULL`. Case e.g. `3235f0e9` Jul 24: short assigned slices + long unassigned fragments; also pure deferred-identity races.

**Root causes:**
1. Frame order: zone/rule eval **before** ReID → BI insert snapshots `track.person_identity_id` (often still None).
2. Later ReID updated `track_sessions` + `person_entered_view` only — **never** prior `billing_interactions` / zone events.
3. Analytics: `COUNT(DISTINCT person_identity_id)` skips NULL.
4. Separate issue: ByteTrack fragmentation resets per-track dwell (no live track-stitch yet). Jul 27 audit: gated same-cam body stitch recovers only ~+4–6 guest purchases/day; thr 50→30 recovers ~+11 without merge — still short of store bill counts.

**Live fix (`camera_worker.py`):**
- `_refresh_event_person_ids` after ReID, before `_persist_events` (same-frame).
- `_backfill_null_person_fks` on ReID resolve + track close: `UPDATE billing_interactions` / `events` SET person WHERE session matches AND person IS NULL.

**Historical:** `danger/backfill_null_billing_person.py` (dry-run default; `--apply`). Joins null BI → session with person.

**Also not in #28 alone:** same-cam track stitch / dwell carry across ByteTrack IDs. See #29.

**Deploy:** `sudo systemctl restart retail-ai.service`.

---

## 29. Fragmented counter visit billing repair (NEW 2026-07-31)

**Bugs:**
1. ByteTrack splits one continuous counter stay into N `track_sessions`; each `ActiveTrack.dwell_seconds` resets to 0 → no fragment hits rule thr → **missed purchase**.
2. Null `billing_interactions.person_identity_id` when identity later attaches only to another fragment (live `_backfill_null_person_fks` is **same session only**).
3. Live `_refresh_event_person_ids` only patches pending events same batch — not historical multi-session visits.

**Fix (post-dedup job):** `repair_fragmented_billing_visits()` in `app/modules/jobs/tasks.py`, invoked at end of `deduplicate_persons()` (every 10 min). Also `danger/repair_fragmented_billing_visits.py` (`--apply` to write).

Steps:
1. Fill null BI/event `person_identity_id` from `track_sessions` (lookback `BILLING_VISIT_LOOKBACK_HOURS=48`).
2. Per enabled `billing_interaction` rule/zone: load sessions with zone dwell evidence in events; group by **same `person_identity_id` + same `camera_id`** with inter-fragment gap ≤ `BILLING_VISIT_STITCH_GAP_SECONDS=60`; **reject same-camera overlap** (different people).
3. `total_dwell = sum(max event dwell per session)`; if ≥ rule thr and no existing BI for person+zone in visit window → insert one `billing_interactions` + matching event (`metadata.billing_visit_repair=true`, fragment ids, sum dwell). Skip `is_staff`.

**Safety:** person_id grouping only — **no body-only stitch** in this job (staff uniform FP). Faceless null sessions still need body-only identity path / backfill separately.

**Config:** `ENABLE_BILLING_VISIT_REPAIR`, `BILLING_VISIT_LOOKBACK_HOURS`, `BILLING_VISIT_STITCH_GAP_SECONDS`, `BILLING_VISIT_DEFAULT_DWELL_THRESHOLD`.

**Tests:** `tests/test_billing_visit_repair.py` (cluster pure function).

---

## Decision log — identity & analytics (must not be lost)

| Decision | Value / action | Why |
|----------|----------------|-----|
| Live `FACE_MATCH_THRESHOLD` | **0.40** (not 0.48) | Code/env truth; CONTEXT older rows said 0.48 |
| Face recent thr | **0.35** + median grey check; accept via `match_tier` | Grey zone was dead without match_tier |
| Body live match | **Median** n≥2 recent bodies only; 0.55 (body_recent); ambiguity 0.03 | Clothing-dependent — no full-lifetime customer body match |
| Body-only create | `ENABLE_BODY_ONLY_IDENTITY_CREATE`; q≥0.55, nearest<0.45, staff<0.48 | Faceless bypass of REQUIRE_FACE; non-confident |
| Backfill body-only | `danger/backfill_body_only_identity.py` days 2026-07-19+20 applied | +24 DISTINCT non-staff purchasers (85→109) |
| Body store contamination | Gate vs **recent** cluster only; never delete old day bodies | Multi-day outfit change OK |
| Staff reattach gallery | Full lifetime if activity-recent | Uniform stable across days |
| Activity-recent | track overlap OR body emb in window (not stale last_seen alone) | Dedup grafts left last_seen Jul-11 while tracks Jul-20 |
| SAME_CAM reject | **No create** | Cloned staff/visitors |
| MATCH STALE / attach FK | **No create**; SAVEPOINT; FOR SHARE; lock 1001 with reextract | Session poison + false IDs |
| Create gates | face ≥0.60 + good_face≥2; INFO log | Keep; match can attach without relaxing create |
| Purchase metric | DISTINCT person_id, not is_staff | Critical findings #15 staff inflation fix |
| Purchase dwell | rule DB thr (live was 25s Jul 31); do not drop only to chase store bills | Identity ceiling + bill vs person |
| Counter visit repair | post-dedup `repair_fragmented_billing_visits`; person+cam+gap 60s; sum dwell | fragmentation missed BI; no body-only group |
| Null BI person backfill | Live on ReID/close + `danger/backfill_null_billing_person.py` | BI fired pre-identity was permanent undercount |
| Counter track stitch | Live body stitch **not** shipped; **visit repair** sums dwell by person_id post-dedup (#29) | body-only stitch staff FP; person-id group safer |
| retail-ai-worker | Dedup/sweep/staff/probes/analytics | API freezes if jobs in uvicorn |
| Track session debug log | `bbox_history` JSON object on close: boxes + best_crop_quality + torso_visibility_ratio + best_face_*; legacy rows stay bare arrays | No schema migration; debug tab `/api/v2/debug/track-sessions` |
| MinIO protect | Also `bbox_history->>'best_face_crop_path'` when object-shaped | Unassigned face crops would otherwise be swept |

**Required restarts after identity engine changes:** `sudo systemctl restart retail-ai.service` (camera workers load engine in-process). Worker only if jobscode changed.
