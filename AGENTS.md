# AGENTS.md — Retail Eye Insights AI Agent Instructions

> Every AI session working on this codebase MUST read these instructions first.

---

## First Action: Read CONTEXT.md

**Before ANY code change, debugging, or analysis, read `/gmr/CONTEXT.md`.** This file contains:

- The complete project architecture and AI pipeline
- All model choices and why they were made (SigLIP2 for gender, MiVOLO for age, etc.)
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

---

## Key Files to Know

| When working on... | Start here |
|---|---|
| Gender classification | `app/modules/reid/siglip2_analyzer.py` |
| Age prediction | `app/modules/reid/mivolo_analyzer.py` |
| Face detection | `app/modules/reid/insightface_analyzer.py` |
| Identity matching | `app/modules/reid/identity_decision_engine.py` |
| Camera pipeline | `app/modules/ai_runtime/camera_worker.py` |
| Analytics | `app/modules/analytics/service.py` |
| Background jobs | `app/modules/jobs/tasks.py` |
| All thresholds | `app/config.py` |
| DB schema | `app/core/db/models/` |
| Diagnostic scripts | `danger/` |
