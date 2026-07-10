# Session Summary — 2026-07-09: Contamination Cleanup + Recent-Window Matching

> Compact handoff for next session. Full detail in `/gmr/CONTEXT.md` issues #17-#24.

---

## What was broken (before this session)

1. **Broken OSNet weights** (issue #16, resolved earlier 2026-07-09): `models/osnet_x1_0.pth` didn't exist; FeatureExtractor fell back to ImageNet backbone + random fc head → body embeddings encoded scene appearance, not identity. Fixed: MSMT17 checkpoint downloaded, `recompute_body_embeddings.py` ran, thresholds retuned (REID_MATCH 0.85→0.50, BODY_CONTAMINATION 0.60→0.50).

2. **Broken union-find `find()`** (issue #13, resolved earlier 2026-07-09): iterative path-compression reverted every union → dedup job found pairs but merged ~0 (silent no-op since inception). Fixed: recursive path compression.

3. **`FACE_MATCH_THRESHOLD` lowered** 0.48→0.40 (earlier 2026-07-09): match dedup threshold, prevent duplicate creation at real-time.

4. **Services not restarted** after the above fixes — both `retail-ai.service` and `retail-ai-worker.service` were running stale code. User restarted mid-session.

---

## What this session found and fixed

### Step A — Worker logging (issue #22)
- `app/worker.py` never called loguru setup → dedup activity invisible (only in `sudo journalctl`).
- **Fix:** `app/logging_config.py::setup_logging()` (shared, idempotent). `worker.py` now calls it. Dedup runs visible in `logs/ai_processing.log`.

### Step B — Contamination cleanup (issues #17, #18)

**Root cause 1 — absorb had no contamination gate (issue #17):**
`_absorb_face_embeddings` / `_absorb_body_embeddings` (`jobs/tasks.py`) moved ALL loser embeddings into winner with only a >0.95 duplicate-angle check — zero cluster-fit validation. Every false merge (~5-6 per busy window, issue #14) directly injected a stranger's face+body into the winner. Staff identities hit hardest (they win most merges via highest face_score/visit_count).

**Root cause 2 — face cleanup used greedy single-linkage (issue #18):**
`_clean_contaminated_face_embeddings` used "keep if sim ≥0.35 to ANY already-kept embedding". Hand-verified this FAILED on both real contaminated identities (`9b6053ac`, `cf793282`): a borderline "bridge" embedding (similar ~0.35-0.49 to BOTH real people) let contamination chain through undetected. The body version already used iterative median-outlier removal correctly.

**Fixes applied:**
- `_clean_contaminated_face_embeddings` (both `tasks.py` and `danger/clean_contaminated_embeddings.py`): replaced greedy with iterative median-outlier removal (same as body version). Aggressive-reject: any embedding whose median sim to the rest < threshold is removed.
- `_absorb_face_embeddings` / `_absorb_body_embeddings`: added contamination gate — a loser embedding is only moved if its median sim to the winner's EXISTING cluster clears threshold (0.35 face / 0.50 body); otherwise DROPPED (logged as rejected).
- `danger/clean_contaminated_embeddings.py`: rewritten to use median approach, cleans BOTH face AND body (`--apply` / `--face-only` / `--body-only`).

**Applied to live DB:** removed 2 faces + 2 bodies. Face contamination below gate: 2 → 0. Then `danger/uncontaminate_tracks.py` disassociated 17 stranger tracks from `9b6053ac` (16 tracks) + `bcb6ab47` (1 track).

### Step C — Historical backfill merge (issue #21)

New script `danger/merge_recent_window_duplicates.py` — one-off cleanup applying the combined recent-window rule to ALL historical persons. Uses fixed union-find + fixed contamination-gated absorb.

**Merge rule (combined, aggressive-reject):**
- Candidate pairs: persons whose track windows are within 5 min of each other.
- **Non-overlap gate (camera-aware):** only blocks if tracks overlap on the SAME camera. Cross-camera overlap (entry + counter simultaneously) is expected for the same person — does NOT block. (Original global overlap check was a bug — blocked true same-person pairs visible on different cameras.)
- **Face path:** `face_max ≥ 0.35` (best cross-pair cosine sim, matching dedup job MAX() semantics).
- **Body-only path:** `body_median ≥ 0.55` AND `≥2 bodies each side` AND non-overlap on same camera AND faces don't contradict (faceless side OR `face_max ≥ 0.25`).
- Anything not matching exactly → not merged.

**Non-contradiction gate (key safety mechanism):** body-alone is NOT trustworthy even in a short window. A "body-chameleon" person (`f9392afa`) matched 5+ strangers at body_median 0.6-0.7 while faces contradicted (<0.20). The non-contradiction gate (faces must not contradict at 0.25) is what prevents those false merges. Lowered from 0.30 to 0.25 to match `FACE_CONTRADICTION_THRESHOLD` — catches cross-angle faces in [0.25, 0.30) that the body path needs.

**Applied:** 100→95 persons (5 merges in first run), then 103→101 (2 more after restart). Acid-test triple `d9ab4071`+`da526bb8`+`2ce51a66` merged into one identity.

### Step D — Live recent-window matching (issue #20)

Implemented in `identity_decision_engine.py`, feature-flagged via `ENABLE_RECENT_WINDOW_MATCHING`.

**Config (`config.py`):**
```
ENABLE_RECENT_WINDOW_MATCHING = True
RECENT_WINDOW_MINUTES = 5
FACE_MATCH_THRESHOLD_RECENT = 0.35
RECENT_BODY_SINGLE_MATCH_THRESHOLD = 0.55
```

**Identity decision flow (updated):**

```
Step 1: Face matching (highest priority)
  for each accumulated face angle:
    search_similar_face() → returns best per-person candidate + first_seen_at
    if face_sim ≥ FACE_MATCH_THRESHOLD (0.40):
      → MATCH (strict, any age of candidate)
    elif ENABLE_RECENT_WINDOW_MATCHING
         AND face_sim ≥ FACE_MATCH_THRESHOLD_RECENT (0.35)
         AND candidate.first_seen_at within RECENT_WINDOW_MINUTES (5):
      → MATCH (recent-window relaxed face)
      logs as [Face Match RECENT]

Step 2: Body ReID (if face didn't match)
  search_similar() → top-5 unique-person candidates + first_seen_at
  for each candidate:
    face-exclusion gate:
      if candidate is RECENT (first_seen within window):
        exclusion_bar = FACE_CONTRADICTION_THRESHOLD (0.25)  ← relaxed for recent
      else:
        exclusion_bar = FACE_BODY_EXCLUSION_THRESHOLD (0.30) ← strict for old
      if face_sim < exclusion_bar: skip candidate (face contradicts)

    collect consensus votes (sim ≥ REID_MATCH_THRESHOLD 0.50)

  Consensus: if 2+ of top-3 agree → MATCH (strict, any age)

  Recent-window single-candidate override:
    if ENABLE_RECENT_WINDOW_MATCHING
       AND no consensus reached
       AND best_candidate is recent (first_seen within window)
       AND body_median ≥ RECENT_BODY_SINGLE_MATCH_THRESHOLD (0.55)
       AND candidate has ≥2 body embeddings:
      → MATCH (bypasses 2-of-3 consensus — impossible when store is quiet)
      logs as [Body RECENT single]
      used_face stays False → demoted to non-confident

Outside the window: strict 0.40 face / 0.50 body + 2-of-3 consensus (unchanged).
```

**Key design points:**
- Face uses MAX (best cross-pair) — cross-angle faces need the best angle, not average.
- Body uses MEDIAN (consistency check) — a single lucky crop is not enough.
- Body-only valid ONLY within the 5-min window (clothing constant → body reliable). Outside: strict consensus.
- Camera-aware overlap: cross-camera overlap is expected, only same-camera overlap blocks.
- Recency check: `_is_recent(first_seen_at)` — candidate's first_seen within window. NOT last_seen (first_seen captures "same visit" better).

**Helper methods added to IdentityDecisionEngine:**
- `_is_recent(first_seen_at)` — checks against `RECENT_WINDOW_MINUTES`
- `_person_body_count(db, person_id)` — count body embeddings
- `_person_body_median_sim(db, person_id, query_embedding)` — median cosine sim to all stored body embeddings (L2-normalized)

### Step E — Dedup job robustness (issue #19)

`jobs/tasks.py` dedup merge loop: each merge now runs in its own SAVEPOINT (`async with db.begin_nested()`). One bad pair rolls back ONLY that merge; batch continues. Previously one failure → `rollback() + return` → aborted ALL merges + skipped cleanup/sweep/staff steps for the whole 10-min cycle.

MinIO deletion of merged losers' crops deferred to AFTER commit (so a savepoint rollback can't delete a MinIO object whose DB reference survived).

---

## All thresholds (current)

| Setting | Value | Where |
|---|---|---|
| `FACE_MATCH_THRESHOLD` | 0.40 | Real-time strict face match |
| `FACE_MATCH_THRESHOLD_RECENT` | 0.35 | Recent-window relaxed face |
| `REID_MATCH_THRESHOLD` | 0.50 | Body ReID consensus (strict) |
| `RECENT_BODY_SINGLE_MATCH_THRESHOLD` | 0.55 | Recent-window body-only override (median) |
| `FACE_CONTRADICTION_THRESHOLD` | 0.25 | Face disassociation (also: recent non-contradiction bar) |
| `FACE_BODY_EXCLUSION_THRESHOLD` | 0.30 | Body candidate face exclusion (strict, old candidates) |
| `FACE_CONTAMINATION_THRESHOLD` | 0.35 | Face contamination cleanup (median) |
| `BODY_CONTAMINATION_THRESHOLD` | 0.50 | Body contamination cleanup (median) |
| `RECENT_WINDOW_MINUTES` | 5 | Recent window |
| `DEDUP_THRESHOLD` (dedup job) | 0.40 | Older-pairs face-only merge (strict, outside window) |
| `MAX_FACE_EMBEDDINGS_PER_PERSON` | 5 | Face storage cap |
| `MAX_EMBEDDINGS_PER_PERSON` (body) | 10 | Body storage cap |

---

## Body ReID distribution (MSMT17 weights, live data)

```
same-person:  median=0.680  p10=0.393  p90=0.845
diff-person:  median=0.386  p10=0.294  p90=0.534  max=0.900
```

Median separation is clean (0.386 vs 0.680). BUT the **tail** of diff-person reaches 0.6-0.9 (~10% of pairs). These are people in similar clothing/lighting — body ReID can't distinguish them. The non-contradiction gate (faces must not contradict at 0.25) is what handles the tail. Body threshold is just a coarse filter.

---

## Files changed this session

| File | Change |
|---|---|
| `app/logging_config.py` | NEW — shared loguru setup |
| `app/worker.py` | calls `setup_logging()` |
| `app/config.py` | +ENABLE_RECENT_WINDOW_MATCHING, RECENT_WINDOW_MINUTES, FACE_MATCH_THRESHOLD_RECENT, RECENT_BODY_SINGLE_MATCH_THRESHOLD |
| `app/modules/reid/identity_decision_engine.py` | recent-window face path + body single-candidate override + `_is_recent` / `_person_body_median_sim` / `_person_body_count` helpers + camera-aware first_seen_at in queries + face-exclusion relaxation for recent candidates |
| `app/modules/jobs/tasks.py` | median face cleanup + absorb contamination gate + per-merge SAVEPOINT isolation + deferred MinIO |
| `danger/clean_contaminated_embeddings.py` | rewritten: median-based, face+body, --apply/--face-only/--body-only |
| `danger/merge_recent_window_duplicates.py` | NEW — historical backfill, camera-aware overlap, combined rule |

Commits: `bd61c8c` (code), `20b44fa` (docs).

---

## Open issues for future work (CONTEXT.md #23, #24)

### #23 — Body crop padding (OPEN)
Body ReID crop uses **10% padding** (default `extract_crop` `padding_pct=0.10`). Adds ~20px horizontal + ~60px vertical background for a typical person bbox. Could include shelf edges, adjacent people, store fixtures — noise that may degrade OSNet. **Potential fix:** reduce to 0% (tight) or 5%. Validate with `danger/measure_body_reid.py` first. Padding is baked into MinIO crops — any change requires `recompute_body_embeddings.py` re-run.

### #24 — Face-to-track misassignment (OPEN, the ONLY remaining contamination source)
After absorb-gate fix, the sole contamination path is the global face-to-track assignment (`camera_worker.py:402-427`). Face **centre** must be inside original body bbox (no expansion). Failure mode: shoulder-to-shoulder proximity — person A's face centre inside person B's body bbox. First frame for a new track has no `last_face_center` anchor → two people entering simultaneously can get faces swapped. Store-time gates catch this for ≥2 faces/bodies, but single contaminated face is blind spot.

**Potential improvements:**
- (a) Stricter first-frame assignment: require full face bbox inside body bbox (not just centre), or higher score threshold when no temporal anchor.
- (b) Inter-face consensus at identity creation: don't create `PersonIdentity` until ≥2 accumulated face embeddings agree with each other (currently `FACE_IDENTITY_MIN_DETECTIONS=2` counts good faces but doesn't check they're the SAME person).
- (c) Reduce body crop padding (issue #23) to reduce body bbox area that can "capture" an adjacent face.

### Other open items
- **SigLIP2 gender misclassification**: woman `64abfee7` consistently classified as M despite clear face (quality 0.69-0.88). Fixed manually (person + tracks → F) but new visits will still get M from SigLIP2 → person-level may flip back via dedup re-vote as new M-tracks accumulate. Needs either a gender override column or classifier-level fix.
- **Step B5 completed**: `uncontaminate_tracks.py` ran on `9b6053ac`, `cf793282`, `bcb6ab47` — 17 tracks disassociated.
- **Data reset planned**: user is doing a full data reset on fresh (uncontaminated) data to validate the new pipeline from scratch.

---

## Key scripts (danger/)

| Script | What it does |
|---|---|
| `clean_contaminated_embeddings.py` | Median-based face+body contamination cleanup. `--apply` / `--face-only` / `--body-only`. |
| `merge_recent_window_duplicates.py` | Historical backfill — merge same-visit duplicates within 5-min window. Camera-aware overlap. Dry-run-first. `--apply` / `--ids`. |
| `uncontaminate_tracks.py` | Re-run InsightFace+OSNet on track crops, disassociate tracks whose face/body contradicts the assigned person. GPU-heavy. `--apply` / `--ids`. |
| `recompute_body_embeddings.py` | Recompute all person_embeddings from MinIO crops with fixed OSNet weights + rebuild IVFFlat index. |
| `measure_body_reid.py` | Measure OSNet same/diff body sim distributions from fresh crop embeddings. Use to validate threshold changes. |
| `diagnose_persons.py` | Deep-dive: cross-person face/body sim, track metadata, contamination check. |

---

## Process architecture (unchanged)

```
retail-ai.service (API + cameras, ~3.4 GB GPU)
  → identity_decision_engine.py (now with recent-window matching)
  → camera_worker.py (face-to-track assignment, body crop extraction)

retail-ai-worker.service (background jobs, ~100 MB, NO GPU)
  → deduplicate_persons (10 min) — now with per-merge isolation + median cleanup + absorb gate
  → close_stale_tracks (5 min)
  → probe_cameras (2 min)

reextract-faces.timer (GPU oneshot, every 20 min)
  → reextract_or_delete_faceless.py
```

All three share PostgreSQL + MinIO. Worker now logs to `logs/ai_processing.log` (same as API).
