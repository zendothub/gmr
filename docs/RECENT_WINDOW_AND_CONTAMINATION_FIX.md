# Recent-Window Matching + Contamination Cleanup + Backfill Merge

> **Date:** July 9, 2026
> **Status:** Implemented, committed as `bd61c8c`
> **Files changed:** `config.py`, `identity_decision_engine.py`, `jobs/tasks.py`, `worker.py`, `logging_config.py` (new), `danger/clean_contaminated_embeddings.py`, `danger/merge_recent_window_duplicates.py` (new)

Full rationale, threshold values, and empirical data: see `/gmr/CONTEXT.md` issues #17–#22.

---

## Problem

40% of persons were single-visit fragments (median track-session = 13s). Same person seen briefly per camera got registered as separate identities because:
- Cross-angle face best-pair falls in [0.35, 0.40) — just under strict 0.40
- 2-of-3 body consensus is impossible when the store is quiet (only 1 candidate exists)

Additionally, contamination was silently polluting winner identities via two bugs:
- `_absorb_*_embeddings` moved ALL loser embeddings with no cluster-fit check
- `_clean_contaminated_face_embeddings` used greedy single-linkage that chained through bridge embeddings

---

## Solution: Two-tier recent-window matching

### Thresholds (in `config.py`)

| Setting | Value | Purpose |
|---|---|---|
| `ENABLE_RECENT_WINDOW_MATCHING` | `True` | Feature flag |
| `RECENT_WINDOW_MINUTES` | `5` | Same-visit window |
| `FACE_MATCH_THRESHOLD_RECENT` | `0.35` | Relaxed face (strict 0.40 outside) |
| `RECENT_BODY_SINGLE_MATCH_THRESHOLD` | `0.55` | Body-only override (median, ≥2 bodies/side) |

### Within 5-min window of candidate's `first_seen_at`:
- **Face path**: `face_max ≥ 0.35` (best cross-pair, matches dedup job MAX() semantics)
- **Body-only path**: `body_median ≥ 0.55` AND `≥2 bodies each side` AND `non-overlapping on SAME camera` AND `faces don't contradict` (faceless side OR `face_max ≥ 0.25`)

### Outside the window:
- Strict 0.40 face / 0.50 body + 2-of-3 consensus (unchanged)

### Camera-aware overlap
Cross-camera overlap (entry + counter simultaneously) is **expected** for the same person — it does NOT block a merge. Only same-camera overlap blocks (two different people cannot occupy the same camera at the same time).

### Non-contradiction gate
Recent candidates use `FACE_CONTRADICTION_THRESHOLD` (0.25) instead of `FACE_BODY_EXCLUSION_THRESHOLD` (0.30) for the face-exclusion gate. This allows cross-angle faces in [0.25, 0.30) to use the body path. Older candidates still use 0.30.

### Critical safety finding
Body-alone is NOT trustworthy even in a short window. A "body-chameleon" person matched 5+ strangers at body_median 0.6–0.7 while faces contradicted (<0.20). The non-contradiction gate is what prevents those false merges — body threshold is just a coarse filter.

---

## Contamination cleanup fixes

### 1. Face cleanup: greedy → median (`tasks.py`)
`_clean_contaminated_face_embeddings` replaced greedy single-linkage ("keep if sim ≥0.35 to ANY kept") with iterative median-outlier removal (same approach as body version). Greedy failed on real contaminated identities where a borderline bridge embedding chained contamination through.

### 2. Absorb contamination gate (`tasks.py`)
`_absorb_face_embeddings` / `_absorb_body_embeddings` now DROP a loser embedding if its median similarity to the winner's existing cluster is below threshold (0.35 face / 0.50 body). Previously moved ALL with only a >0.95 duplicate check — silently polluted winners on every false merge.

### 3. Dedup per-merge isolation (`tasks.py`)
Each merge runs in its own SAVEPOINT (`async with db.begin_nested()`). One bad pair rolls back ONLY that merge; batch continues. Previously one failure aborted ALL merges + downstream cleanup.

---

## New scripts

| Script | Purpose |
|---|---|
| `danger/merge_recent_window_duplicates.py` | Historical backfill — merge same-visit duplicates within 5-min window. Uses fixed contamination-gated absorb. Dry-run-first. |
| `danger/clean_contaminated_embeddings.py` | Rewritten: median-based, cleans BOTH face AND body. `--apply` / `--face-only` / `--body-only`. |
| `app/logging_config.py` | Shared logging setup. `worker.py` now calls it so dedup activity is visible in `logs/ai_processing.log`. |

---

## Validation

- Acid-test: `d9ab4071` + `da526bb8` + `2ce51a66` (same person, face 0.388 + zero-face body_median 0.622) merged into one identity.
- Contamination cleanup applied: 2 faces + 2 bodies removed. Face contamination below gate: 2 → 0.
- 17 stranger tracks disassociated from contaminated staff identity `9b6053ac`.
- Dedup absorb gate verified live: rejected contaminated face (0.336 < 0.35) and body (0.309 < 0.50) during a real merge.
