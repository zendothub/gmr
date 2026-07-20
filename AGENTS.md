# AGENTS.md — Retail Eye Insights AI Agent Instructions

> Every AI session working on this codebase MUST read these instructions first.

---

## First Action: Read CONTEXT.md

**Before ANY code change, debugging, or analysis, read `CONTEXT.md` (repo root `/gmr/gmr/CONTEXT.md`, or host `/gmr/CONTEXT.md` if mirrored).** This file contains:

- The complete project architecture and AI pipeline
- All model choices and why they were made (SigLIP2 face+margin for gender, InsightFace genderage median for age, etc.)
- Every critical threshold and the empirical analysis behind each value
- The identity decision flow
- Staff detection and purchase counting logic
- MinIO crop lifecycle and deferred deletion architecture
- Known issues with their current mitigations
- All danger/diagnostic scripts

**Do NOT propose changes to thresholds without reading the rationale in CONTEXT.md.** Each threshold was set based on empirical data from this specific CCTV setup.

---

## Critical Rules

1. **Single uvicorn worker MANDATORY** — `--workers 1` is required. Camera workers share in-process state that breaks with multiple workers.

2. **Per-camera YOLO models MANDATORY** — Each camera gets its own YOLO+ByteTrack instance. Sharing causes ByteTrack state corruption (track ID explosions).

3. **No immediate MinIO deletion** — All deletions go through `CameraWorker._minio_cleanup()` → deferred set → periodic sweep. Never call `client.remove_object()` directly.

4. **Face required for identity creation** — `REQUIRE_FACE_FOR_IDENTITY = True`. No person exists without a face. Body-only tracks get NULL `person_identity_id`.

5. **Test before deploying model changes** — Use `danger/test_*.py` scripts to validate any model change on real data before integrating into the pipeline.

6. **Keep CONTEXT.md updated** — After any significant change (new model, threshold adjustment, architectural decision), update CONTEXT.md with the rationale, data, and date.

7. **Recent-window matching is feature-flagged** — `ENABLE_RECENT_WINDOW_MATCHING` in `config.py`. Customer body match uses **only bodies with `captured_at` in `RECENT_WINDOW_MINUTES` (5)**. Older body rows stay stored — mismatch means **do not merge**, not delete. Activity-recent = track activity or body emb in window (not stale `person.last_seen` alone). Staff reattach: activity-recent + full body gallery (`STAFF_BODY_USE_FULL_GALLERY`). Face **0.40** outside window / **0.35** recent. Body median ≥**0.55** recent (n_bodies_recent≥2) + `BODY_MATCH_AMBIGUITY=0.03`. Body ReID is clothing-dependent — never trust body alone across days.

8. **Camera-aware overlap** — Cross-camera overlap (entry + counter simultaneously) is expected for the same person. Only same-camera overlap blocks a merge. After SAME_CAM reject: **do not create** a new person (leave unassigned). The backfill script (`danger/merge_recent_window_duplicates.py`) must distinguish cameras, not use a global overlap check.

9. **Contamination cleanup uses median, not greedy** — `_clean_contaminated_face_embeddings` and `_clean_contaminated_body_embeddings` both use iterative median-outlier removal. Greedy single-linkage chains contamination through bridge embeddings. Never revert to the greedy approach.

10. **Absorb must check cluster fit** — `_absorb_face_embeddings` / `_absorb_body_embeddings` must DROP a loser embedding that doesn't fit the winner's cluster (median sim < 0.35 face / < 0.50 body). Never move embeddings without this check — it was the root cause of staff-identity pollution.

11. **Worker logging uses shared setup** — `app/worker.py` calls `app/logging_config.py::setup_logging()`. Do not add ad-hoc loguru sinks in the worker; use the shared helper so background job activity stays in `logs/ai_processing.log`.

12. **Identity delete lock** — Processes that DELETE `person_identities` while cameras run must take `IDENTITY_ADVISORY_LOCK_KEY` (1001), same as live `decide_identity` (see CONTEXT #26 / `reextract_or_delete_faceless.py`).

---

## Key Files to Know

| When working on... | Start here |
|---|---|
| Gender classification | `app/modules/reid/siglip2_analyzer.py` (margin δ, face-only) |
| Age prediction | `app/modules/reid/insightface_analyzer.py` (genderage head + median) |
| Face detection | `app/modules/reid/insightface_analyzer.py` |
| Identity matching (incl. recent-window) | `app/modules/reid/identity_decision_engine.py` |
| Camera pipeline | `app/modules/ai_runtime/camera_worker.py` |
| Analytics | `app/modules/analytics/service.py` |
| Background jobs (dedup, cleanup, staff) | `app/modules/jobs/tasks.py` |
| All thresholds (incl. recent-window) | `app/config.py` |
| Shared logging setup | `app/logging_config.py` |
| DB schema | `app/core/db/models/` |
| Diagnostic + fix scripts | `danger/` |
| Contamination cleanup | `danger/clean_contaminated_embeddings.py` |
| Historical backfill merge | `danger/merge_recent_window_duplicates.py` |
| Recent-window + contamination design | `docs/RECENT_WINDOW_AND_CONTAMINATION_FIX.md` |
