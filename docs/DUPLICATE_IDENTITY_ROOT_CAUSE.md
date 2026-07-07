# Duplicate Person Registration — Root Cause Analysis & Fixes

> **Date:** July 7, 2026  
> **Status:** FIXED  
> **Files changed:** `config.py`, `reid/identity_decision_engine.py`, `jobs/tasks.py`, `jobs/scheduler.py`, `reid/insightface_analyzer.py`, `reid/crop_quality.py`, `ai_runtime/camera_worker.py`, `danger/dedup_faces.py`

---

## Observed Symptoms

After 24 hours of operation with 2 cameras:

| Metric | Value | Expected |
|---|---|---|
| PersonIdentity rows created | **885** | ~100–250 for a pharmacy |
| Persons with face similarity > 0.48 to another person | **1,554 unique pairs** | ~0 |
| Persons created within 5 s of another person | **351** | ~0 |
| Persons correctly seen on both cameras | 146 | should be majority |
| New persons/hour at peak | 111 | should be ~10–30 |

The data was severely corrupted: visit counts inflated 3–6×, analytics meaningless, unique-person reports showing ~5× real headcount.

---

## Root Cause: Four Interacting Bugs

### Bug 1 — Same threshold used for matching AND disassociation *(Critical)*

`FACE_MATCH_THRESHOLD = 0.48` was used in two opposite roles:

```python
# Positive match — needs a HIGH similarity to confirm same person
if face_sim >= FACE_MATCH_THRESHOLD:        # 0.48
    used_face = True

# Contradiction / disassociation — needs a LOW similarity to confirm different person  
if best_face_sim < FACE_MATCH_THRESHOLD:    # 0.48 — same value!
    current_id_contradicted = True
```

**Why this causes duplicates:**

ArcFace (buffalo_l) cross-angle cosine similarity for the **same physical person** viewed from two fixed store cameras often lands in the **0.40–0.47 range** — below 0.48 but clearly not a different person. With 0.48 as both gates:

- Positive match fails (0.44 < 0.48) → Camera B does not recognise Camera A's identity
- Contradiction fires (0.44 < 0.48) → Camera B disassociates from its own temporary identity

The same face angle gap produces both a failed match AND a forced disassociation. The person gets a new identity every time they are seen from a slightly different angle.

**Evidence:** 1,554 cross-identity pairs with face similarity > 0.48 existed in the DB — these pairs *should* have been matched but weren't because at the moment of registration (when only 1 embedding existed per identity) the initial similarity was 0.44–0.47, just below the gate.

---

### Bug 2 — Body ReID face-exclusion gate used the same threshold *(Critical)*

When a face match failed, the code fell through to body ReID (OSNet). But the body-candidate loop also used `FACE_MATCH_THRESHOLD`:

```python
for candidate in body_candidates:
    candidate_faces = await _get_person_face_embeddings(candidate_id)
    if candidate_faces:
        best_f_sim = max(np.dot(f, face_embedding) for f in candidate_faces)
        if best_f_sim < FACE_MATCH_THRESHOLD:   # 0.48
            continue   # Skip this candidate!
```

Result: Camera B detects person P1 (Camera A's identity, stored with Camera A's frontal face). Cross-angle similarity = 0.44 < 0.48 → P1 is **excluded** from body candidates. No candidates survive → P2 created → duplicate.

The body ReID was supposed to be a fallback for exactly this situation, but it was gatekept by the same threshold that caused the face search to fail.

---

### Bug 3 — `skip_body_reid` logic abandoned the fallback entirely *(Major)*

```python
# When face search fails but face quality is high, skip body search "to avoid false merges"
if face_score >= FACE_SEARCH_THRESHOLD:   # 0.65
    skip_body_reid = True
```

**Intent vs. reality:**

The intent was conservative: a high-confidence face that doesn't match anything should not be falsely merged via body appearance alone. But this assumed the face DB is comprehensive at the moment of search.

In practice, when Camera B processes its first ReID window for a person, Camera A may have registered that person 0.5–2 seconds ago with only one face embedding (frontal). Camera B's face (profile, score 0.75 ≥ 0.65) searches for Camera A's face — similarity 0.44 < 0.48 — finds no match — `skip_body_reid = True` — never tries OSNet body matching — creates P2.

46 persons in the dataset had **no body embeddings at all** because they were created via this path. Every single one is likely a duplicate.

---

### Bug 4 — Confirmed identities never merge post-track *(Systemic)*

The temporary identity merge only fires for `is_temporary=True`:

```python
prune_old_id = current_person_id if is_temporary else None
if prune_old_id:
    await self._delete_person(db, prune_old_id, matched_id)
```

An identity becomes non-temporary (confident) quickly: once a camera reliably matches its own angle (high similarity on same-camera embeddings), `is_confident = True` fires. After that, even if a later ReID window finds a cross-camera better match, `prune_old_id = None` → the old identity is not deleted. Both identities survive independently.

This means the "early phase rescue" mechanism only works within the first 1–2 accumulation windows. After ~10 seconds of tracking, permanently separate identities are created.

---

## Why The Advisory Lock Doesn't Solve This

The existing `pg_advisory_xact_lock(1001)` in `decide_identity()` **is working correctly**. It serializes identity creation across cameras: Camera B waits for Camera A's transaction to commit, then searches and can see Camera A's newly created person.

The lock prevents race-condition duplicates at the exact same timestamp. It cannot prevent duplicates caused by **face similarity below threshold** — Camera B can see Camera A's person but still fails to match it because the angular difference produces a cosine similarity of 0.44.

---

## Fixes Applied

### Fix 1 — Separate match and contradiction thresholds

Two new config values:

```python
FACE_CONTRADICTION_THRESHOLD: float = 0.25
# Triggers disassociation only when face is DEFINITELY a different person.
# Same person cross-angle: 0.40–0.47 → safely above 0.25, no disassociation.
# Genuinely different person: typically 0.10–0.35 → below 0.25, disassociates correctly.

FACE_BODY_EXCLUSION_THRESHOLD: float = 0.30
# Body candidate exclusion gate. More permissive than match threshold.
# Allows body ReID to proceed for cross-angle same-person (0.40–0.47 > 0.30 → not excluded).
# Still excludes genuinely different people (typically < 0.25 < 0.30 → excluded).
```

`FACE_MATCH_THRESHOLD = 0.48` is kept for positive face matching (confirming same person).

**In `decide_identity()`:**
```python
# Contradiction check — now uses FACE_CONTRADICTION_THRESHOLD (0.25)
if best_face_sim < self.settings.FACE_CONTRADICTION_THRESHOLD:
    current_id_contradicted = True

# Body candidate exclusion — now uses FACE_BODY_EXCLUSION_THRESHOLD (0.30)
if best_f_sim < self.settings.FACE_BODY_EXCLUSION_THRESHOLD:
    continue
```

**Effect for the same person at different camera angles (similarity 0.44):**

| Gate | Old threshold | Old result | New threshold | New result |
|---|---|---|---|---|
| Face positive match | 0.48 | FAIL (0.44 < 0.48) | 0.48 | FAIL (unchanged) |
| Contradiction | 0.48 | FIRES (0.44 < 0.48) | 0.25 | silent (0.44 > 0.25) ✓ |
| Body exclusion | 0.48 | EXCLUDED (0.44 < 0.48) | 0.30 | NOT excluded (0.44 > 0.30) ✓ |
| Body match (OSNet) | — | never reached | 0.80 | fires → P1 found ✓ |

### Fix 2 — Remove `skip_body_reid`

The entire `skip_body_reid` block was deleted. When face search fails, body ReID is always attempted as fallback. Body-only matches are marked `is_confident = False` (non-permanent), which is the correct behavior: a body-based match is held tentatively until a face match confirms it.

### Fix 3 — Periodic deduplication job (every 10 minutes)

Added `deduplicate_persons()` to APScheduler. Handles identities that slip through (e.g., camera-angle similarity just at the border, or both identities go confident before the other camera's embedding arrives).

**Algorithm:**

1. Find duplicate pairs using pgvector `LATERAL` query (O(N·log N), not O(N²)):
```sql
SELECT DISTINCT pid_a, pid_b, MAX(1 - dist) AS max_sim
FROM person_face_embeddings a
CROSS JOIN LATERAL (
    SELECT person_identity_id, embedding <=> a.embedding AS dist
    FROM person_face_embeddings
    WHERE person_identity_id != a.person_identity_id
      AND (1 - (embedding <=> a.embedding)) >= :threshold
    ORDER BY dist LIMIT 5
) b_near
GROUP BY pid_a, pid_b
HAVING MAX(1 - b_near.dist) >= :threshold
```

2. Resolve connected components via union-find (handles A=B, B=C → all merge into one winner).

3. Winner selection: highest `best_face_score`, tiebreak by earliest `first_seen_at`.

4. Loser merge: reassign `track_sessions`, `events`, `billing_interactions`, `storage_objects` FK; absorb `visit_count` and `first_seen_at`; CASCADE-delete loser; clean up loser MinIO files.

**Why not O(N²)?** The old `dedup_faces.py` script timed out (>120s) because it did one `CROSS JOIN` per identity pair. The `LATERAL` approach lets pgvector's IVFFlat index handle each probe in O(log N). `ivfflat.probes = 50` ensures sufficient recall.

### Fix 4 — Log orphaned confident identities on switch

When a non-temporary (confident) identity is superseded by a better cross-camera match, a log entry is now emitted so engineers can track how often this occurs:

```
[ReID Refined] Non-temporary ID abc12345 superseded by def67890 (score=0.83).
Old identity left for dedup job to clean up.
```

The dedup job at 10-minute intervals will pick up these orphaned identities automatically.

---

## Face Frontality Quality Scoring

### Problem (Pre-fix)

`assess_face_quality()` returned `face_score * 1.1` if both eyes were detected — a minimal 10% boost that didn't meaningfully differentiate frontal vs. profile faces. The face score used for identity-critical decisions (`FACE_IDENTITY_MIN_SCORE` gating, `best_face_score` comparisons) was therefore mostly the raw InsightFace detection confidence, which can be high even for profile shots.

Profile faces stored as "best face" embeddings hurt cross-camera matching: the stored embedding of a profile-angle face has low cosine similarity to a frontal-angle embedding of the same person, making the face search less effective.

### Fix

`assess_face_quality()` now computes a **geometric frontality score** from InsightFace's 5-point keypoints (`kps`):

```
kps[0] = left_eye     kps[1] = right_eye     kps[2] = nose
kps[3] = left_mouth   kps[4] = right_mouth
```

Three sub-scores:

| Sub-score | Formula | Frontal | Profile |
|---|---|---|---|
| **Eye spread** | `abs(rx - lx) / face_width` | ~0.35–0.50 | ~0.00–0.10 |
| **Nose centering** | `1 - 2*abs(nose_x - face_cx) / face_width` | ~1.0 | ~0.0–0.5 |
| **Eye symmetry** | `1 - abs(ry - ly) / face_height * 4` | ~1.0 | ~0.5–1.0 |

```
frontality = 0.55 × eye_spread_score + 0.30 × nose_score + 0.15 × sym_score
face_quality = (1 - FACE_FRONTALITY_WEIGHT) × det_score + FACE_FRONTALITY_WEIGHT × frontality
```

Default `FACE_FRONTALITY_WEIGHT = 0.35`. Effect:

| Face angle | det_score | frontality | face_quality |
|---|---|---|---|
| Fully frontal | 0.85 | 0.95 | **0.89** |
| 30° (3/4 view) | 0.82 | 0.65 | **0.76** |
| 60° (side) | 0.80 | 0.30 | **0.63** |
| Profile | 0.78 | 0.05 | **0.53** |

A clean frontal detection now always outranks an angled shot even if the latter has higher raw detection confidence. The `best_face_score` on `PersonIdentity` will accumulate frontal embeddings preferentially, making cross-camera face matching more reliable.

### Re-enabled Eye-Spread Gate in `camera_worker.py`

The `FACE_MIN_EYE_SPREAD` gate was previously disabled with a comment:
```python
# FACE_MIN_EYE_SPREAD check disabled — too aggressive for non-frontal shots
```

It is now **re-enabled** at a relaxed threshold (`FACE_MIN_EYE_SPREAD = 0.25`):

| eye_spread | Angle | Treatment |
|---|---|---|
| ≥ 0.35 | Frontal–near-frontal | Accepted |
| 0.25–0.34 | 3/4 view | Accepted |
| 0.15–0.24 | Side angle | **Rejected** |
| < 0.15 | Profile | **Rejected** |

Rejected faces are not used for ReID accumulation (`face_frontal = False`). They can still contribute body ReID embeddings. This prevents profile-only face embeddings from being stored as the "canonical" face for an identity.

---

## Configuration Reference (Post-Fix)

| Setting | Value | Notes |
|---|---|---|
| `FACE_MATCH_THRESHOLD` | `0.48` | Positive face match gate (unchanged) |
| `FACE_CONTRADICTION_THRESHOLD` | `0.25` | **New** — disassociation gate (much lower) |
| `FACE_BODY_EXCLUSION_THRESHOLD` | `0.30` | **New** — body candidate face gate |
| `FACE_MIN_EYE_SPREAD` | `0.25` | Re-enabled frontal gate (was disabled) |
| `FACE_FRONTALITY_WEIGHT` | `0.35` | **New** — weight of frontality in face_quality |
| `FACE_IDENTITY_MIN_SCORE` | `0.60` | Unchanged; now applied to face_quality (includes frontality) |

---

## Data Reset

After deploying these fixes, a **full data reset** is required to clear the corrupted identity data:

```bash
systemctl stop retail-ai.service
cd /gmr/gmr
PYTHONPATH=. venv/bin/python danger/reset_tracking_data.py --yes
systemctl start retail-ai.service
```

This wipes all person identities, track sessions, events, embeddings, analytics, and MinIO crops. Configuration tables (cameras, zones, rules, stores, users) are preserved.

The PoC data prior to this reset had ~600+ spurious identity records and should not be used for any analysis.

---

## Ongoing Monitoring

The dedup job runs every 10 minutes and logs its results. Check for residual duplicates:

```bash
# Check dedup job output in logs
journalctl -u retail-ai.service | grep "Dedup job"

# Run the read-only diagnostic script
cd /gmr/gmr
PYTHONPATH=. venv/bin/python danger/dedup_faces.py
```

After the reset and with these fixes, the expected dedup rate should be **near zero** (0–3 pairs per run in a busy environment). If it consistently finds 10+ pairs per run, the thresholds may need further tuning for the specific camera angles in this installation.
