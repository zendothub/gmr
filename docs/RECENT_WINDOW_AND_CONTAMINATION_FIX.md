# Recent-Window Matching + Contamination Cleanup + Backfill Merge

> **Date:** July 9–10, 2026 (initial); **updated July 17, 2026 (P0–P5 identity live path)**  
> **Status:** Live. See also repo `CONTEXT.md` issues **#20, #25, #26**.  
> **Files:** `config.py`, `identity_decision_engine.py`, `camera_worker.py`, `jobs/tasks.py`, `worker.py`, `logging_config.py`, `danger/clean_contaminated_embeddings.py`, `danger/merge_recent_window_duplicates.py`, `danger/reextract_or_delete_faceless.py`

Full rationale, threshold values, and empirical data: see `CONTEXT.md` (repo root) / `/gmr/CONTEXT.md`.

---

## Problem

40% of persons were single-visit fragments (median track-session = 13s). Same person seen briefly per camera got registered as separate identities because:
- Cross-angle face best-pair falls in [0.35, 0.40) — just under strict 0.40
- Historical **unique-person 2-of-3 body consensus** was **structurally impossible** (search returns one row per person); failure path nulling `best_candidate` also killed the recent body override (fixed 2026-07-17, CONTEXT #25)

Additionally, contamination was silently polluting winner identities via two bugs:
- `_absorb_*_embeddings` moved ALL loser embeddings with no cluster-fit check
- `_clean_contaminated_face_embeddings` used greedy single-linkage that chained through bridge embeddings

---

## Solution: Two-tier recent-window matching

### Thresholds (in `config.py`)

| Setting | Value | Purpose |
|---|---|---|
| `ENABLE_RECENT_WINDOW_MATCHING` | `True` | Feature flag |
| `RECENT_WINDOW_MINUTES` | `5` | Same-visit window (`last_seen_at`) |
| `FACE_MATCH_THRESHOLD` | `0.40` | Strict face (live = dedup) |
| `FACE_MATCH_THRESHOLD_RECENT` | `0.35` | Relaxed face inside window; accept via `match_tier` |
| `RECENT_BODY_SINGLE_MATCH_THRESHOLD` | `0.55` | Body median override inside window (n_bodies≥2) |
| `REID_MATCH_THRESHOLD` | `0.50` | Body **median** (not 2-of-3 votes) outside/strict |
| `BODY_MATCH_AMBIGUITY` | `0.03` | Reject if top-2 body medians too close |
| `FACE_MATCH_MEDIAN_THRESHOLD` | `0.30` | Grey-zone [0.35, 0.40) face median gate |

### Within 5-min window of candidate's `last_seen_at`:
- **Face path**: best-pair ≥ 0.35; grey zone requires median ≥ 0.30 when ≥3 cross-pairs; `match_tier=face_recent` so CASE1 accepts at 0.35 (not re-gated at 0.40)
- **Body path**: gallery **median** ≥ 0.55, n_bodies≥2, face non-contradiction; log `[Body RECENT single]` / or strict median ≥0.50 as `[Body Match]`

### Outside the window:
- Face ≥ **0.40** (strict), body **median ≥ 0.50**, top-2 ambiguity reject  
- **No** unique-person 2-of-3 body vote (dead by design of `_search_similar` dedupe)

### SAME_CAM / create (2026-07-17)
After same-camera concurrent-track overlap reject: **do not create** a new person — leave track unassigned.

### Persistence / FK (P5, 2026-07-17)
Attach stores use SAVEPOINT + `person_identities` exist/`FOR SHARE`. Do not create after MATCH STALE or IntegrityError. Faceless delete takes advisory lock **1001**.

### Camera-aware overlap
Cross-camera overlap (entry + counter) is expected. Only same-camera overlap blocks.

### Non-contradiction gate
Recent candidates use `FACE_CONTRADICTION_THRESHOLD` (0.25) for body face-exclusion gate. Older candidates use 0.30.

### Critical safety findings
1. Body-alone is NOT trustworthy even in a short window — non-contradiction + ambiguity gates.
2. Uniformed staff hard on OSNet — face is authority; staff reattach separate (body med ≥0.70).
3. `_is_recent` must use `last_seen_at` not `first_seen_at`.
4. Face best-pair grey zone needs median check (0.30).

---

## Contamination cleanup fixes

### 1. Face cleanup: greedy → median (`tasks.py`)
`_clean_contaminated_face_embeddings` replaced greedy single-linkage with iterative median-outlier removal.

### 2. Absorb contamination-gated
Loser embeddings moved only if they fit winner cluster.

### 3. Historical backfill
`danger/merge_recent_window_duplicates.py` — dry-run default; recent-window combined face/body rule.

---

## Tests / deploy

- `tests/test_identity_decision_p0_p3.py` — body median, face recent, SAME_CAM no-create  
- `tests/test_identity_persistence_p5.py` — FK/store/stale gates  
- Restart: `sudo systemctl restart retail-ai.service`
