# Crop Lifecycle & MinIO Cleanup Fixes

> **Date:** July 7, 2026  
> **Status:** FIXED  
> **Files changed:** `app/modules/ai_runtime/camera_worker.py`

---

## Problem Summary

The debug panel was showing **404s on face and body crop URLs** stored in `PersonFaceEmbedding.face_crop_path` and in the active-tracks real-time view. Simultaneously, MinIO storage was growing unboundedly — particularly with `curr_face_*` prefixed files accumulating at ~1GB+/hour per camera.

Three distinct crop lifetime problems were identified, each caused by a different gap in `_run_reid`.

---

## Crop Taxonomy

Every `_run_reid` call produces up to **three** different MinIO uploads:

| Variable | Prefix | Purpose | Added to `accum_list`? |
|---|---|---|---|
| `crop_path` | `crop_*` | Body crop for ReID quality + OSNet embedding | Yes (if `should_accumulate=True`) |
| `current_face_path` | `curr_face_*` | Real-time face visible in active-tracks debug view | **No** |
| `face_crop_path` | `face_*` | High-quality frontal face for identity storage | Yes (as `item[5]`) |

Understanding this separation is critical. The `accum_list` window cleanup (every `REID_ACCUMULATION_FRAMES=5` frames) only touches files that were added to `accum_list`. Anything outside the window is invisible to the cleanup.

---

## Bug 1 — `PersonFaceEmbedding` rows stored paths that were already deleted (404s)

### Root Cause

In `_run_reid`, the lifecycle for face crops added to `face_embedding_list` was:

```
Frame N:  face_crop_path saved to MinIO
          → added to accum_list (item[5])
          → also added to face_embedding_list (line 835-836)  ← BEFORE window fires

Frame N+4: window fires
          → cleanup loop: keeps best face (window_face_crop_path)
          → deletes all OTHER face crops in the window  ← DELETES some paths in face_embedding_list

Later:    decide_identity() stores face_embedding_list to PersonFaceEmbedding
          → face_crop_path in the DB points to file that no longer exists  → 404
```

The `face_embedding_list` was populated **before** the window cleanup ran, so it accumulated paths that the cleanup would later delete.

### Fix

Before the cleanup loop, build a **protected set** from `face_embedding_list`:

```python
_protected_face_paths = {
    entry[2] for entry in track.face_embedding_list if entry[2] is not None
}
```

In the cleanup condition:
```python
# Was:
if other_face_path and other_face_path != window_face_crop_path:
    minio_delete(...)

# Now:
if (other_face_path
        and other_face_path != window_face_crop_path
        and other_face_path not in _protected_face_paths):
    minio_delete(...)
```

Face crops already referenced in `face_embedding_list` are preserved in MinIO so the later `PersonFaceEmbedding` DB insert points to an existing file.

---

## Bug 2 — `curr_face_*` files accumulated forever (~1 GB+/hour/camera)

### Root Cause

Every frame with any face detection (regardless of quality) triggered:
```python
current_face_path = save_image(face_result.face_crop, ..., prefix=f"curr_face_{camera_id}")
track.current_face_crop_path = current_face_path
```

The **old** `track.current_face_crop_path` value was silently overwritten. The MinIO file it pointed to was **never deleted**. At 10 fps with ~70% face detection rate:

- ~7 orphaned `curr_face_*` files / second / person
- With 5 cameras × 10 people visible = **350 orphaned files / second**
- Each file ≈ 30–80 KB → **~1 GB / hour** leaked purely from this source

Additionally, when a track went stale (`cleanup_stale_tracks`), its final `current_face_crop_path` was never cleaned up — it lasted indefinitely.

### Fix

**Delete-before-overwrite**: capture the previous path before resetting, then delete it after the new upload:

```python
_prev_curr_face = track.current_face_crop_path   # capture before reset
track.current_face_crop_path = None
...
track.current_face_crop_path = current_face_path  # assign new
# delete old
if _prev_curr_face and _prev_curr_face != current_face_path:
    self._minio_cleanup(_prev_curr_face)
```

**On stale track eviction** (`_process_frame` stale loop):
```python
if t.current_face_crop_path:
    self._minio_cleanup(t.current_face_crop_path)
```

---

## Bug 3 — Body crops from non-accumulated frames leaked indefinitely

### Root Cause

`crop_path = save_image(...)` runs at line 723 **before** the quality gate. When `should_accumulate = False` (body quality too low and no frontal face), the function returns early without adding `crop_path` to `accum_list`. The file is never in any cleanup window.

`track.current_crop_path` is also set to this `crop_path` and overwritten on the next frame — leaving the previous file orphaned.

At 10 fps with ~20% quality-rejection rate: **2 leaked body crops / second / person**.

### Fix

At the **start of `_run_reid`** (before uploading a new body crop), check if the previous `current_crop_path` is still needed:

```python
_prev_body_crop = track.current_crop_path
if _prev_body_crop:
    _accum_body_paths = {item[2] for item in self.track_embeddings.get(track.local_track_id, [])}
    if _prev_body_crop not in _accum_body_paths and _prev_body_crop != track.best_crop_path:
        self._minio_cleanup(_prev_body_crop)
```

**Guards:**
- Not in `accum_list` → safe to delete (not pending for ReID window)
- Not equal to `best_crop_path` → safe to delete (not the overall best quality crop for this track)

---

## Bug 4 — `current_crop_path` briefly 404 after each window fires (transient)

### Root Cause

At the 5th frame of a window, `track.current_crop_path` = that frame's `crop_path`. When the window fires and the cleanup deletes non-best body crops, the 5th frame's crop might be deleted. `track.current_crop_path` now points to a deleted file for ~100 ms (until the next frame updates it). The active-tracks debug view polling during this window would return a 404 URL.

### Fix

After the cleanup loop, before `accum_list.clear()`:

```python
_window_body_paths = {item[2] for item in accum_list if item[2]}
if track.current_crop_path in _window_body_paths and track.current_crop_path != best_crop_path:
    track.current_crop_path = best_crop_path
```

Forces `current_crop_path` to the surviving best crop immediately after cleanup.

---

## Bug 5 — Incomplete window crops leaked on stale track eviction

### Root Cause

A track can go stale mid-window (< 5 frames accumulated, never fires). At stale-track cleanup time:
```python
self.track_embeddings.pop(t.local_track_id, None)  # old code — just discards the list
```

The body/face crops in the partial window — already uploaded to MinIO — are simply abandoned with no reference and no cleanup.

### Fix

Before the pop, delete partial window crops while protecting any paths still referenced by persistent track state that `_close_track_session` is about to write to DB:

```python
_partial_window = self.track_embeddings.pop(t.local_track_id, [])
if _partial_window:
    _protected = {t.best_crop_path, t.best_face_crop_path_for_id}
    if t.best_demographics:
        _protected.add(t.best_demographics.get("face_crop_path"))
    for _fe, _fs, _fp in t.face_embedding_list:
        if _fp:
            _protected.add(_fp)
    _protected.discard(None)
    for _item in _partial_window:
        for _path in (_item[2], _item[5]):
            if _path and _path not in _protected:
                self._minio_cleanup(_path)
```

---

## Helper: `_minio_cleanup()`

All five fixes share a new static helper method on `CameraWorker` (replaces the old inline `_extract_object_name` function that was re-defined on every loop iteration):

```python
@staticmethod
def _minio_cleanup(full_path: str) -> None:
    """Delete a MinIO object by full bucket/key path. Swallows all errors."""
    try:
        from app.modules.storage.minio_client import delete_object as _minio_del
        key = full_path.split("/", 1)[1] if "/" in full_path else full_path
        _minio_del(key)
    except Exception as _e:
        logger.warning(f"MinIO cleanup failed for {full_path}: {_e}")
```

---

## What `TrackSession.best_crop_path` and `PersonIdentity.face_crop_path` Are NOT Affected

`TrackSession.best_crop_path` is safe because it is set by `_create_track_session` (its own separate `save_image` call, **never in `accum_list`**) and only overwritten by `_close_track_session` with `track.best_crop_path` (the overall best quality crop, always the window-best so always preserved by cleanup).

`PersonIdentity.face_crop_path` is safe because it is always `track.best_demographics["face_crop_path"]`, which is the face with the globally highest face score seen so far. By definition this is the maximum across all windows, which is always the window-best for whichever window it came from — and thus always preserved.

---

## Diagnostic Verification

After these fixes, MinIO should stop growing from `curr_face_*` uploads:

```bash
# List curr_face objects in MinIO (should be ~1 per active track, not thousands)
mc ls minio/retail/crops/ | grep curr_face | wc -l
```

Expected: ≈ (number of active tracks) × 1. Before fix: tens of thousands accumulating hourly.
