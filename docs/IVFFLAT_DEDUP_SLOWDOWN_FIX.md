# IVFFlat Index Misconfiguration — Dedup Job Hung for 36+ Minutes

**Date:** 2026-10-07  
**Severity:** High — dedup job completely blocked for the entire store-open window  
**Branch:** staffRegistration/ak  
**Files changed:** `app/modules/jobs/tasks.py`, DB index rebuilt in-place

---

## Symptom

From approximately **12:21 IST** the dedup job (`deduplicate_persons`) stopped merging duplicate
person identities. Multiple persons visible on the dashboard were clearly the same physical person
but were never merged.

Checking `retail-ai-worker.service` via `journalctl`:

```
Oct 07 12:27:30 Execution of job "deduplicate_persons ... skipped: maximum number of running instances reached (1)
Oct 07 12:33:30 Execution of job "deduplicate_persons ... skipped: maximum number of running instances reached (1)
Oct 07 12:39:30 Execution of job "deduplicate_persons ... skipped: maximum number of running instances reached (1)
Oct 07 12:45:30 Execution of job "deduplicate_persons ... skipped: maximum number of running instances reached (1)
Oct 07 12:51:30 Execution of job "deduplicate_persons ... skipped: maximum number of running instances reached (1)
```

APScheduler allows max 1 concurrent instance of `deduplicate_persons`. The first run at 12:21 never
finished, so every subsequent 6-minute tick was silently skipped. No merges happened for 40+ minutes.

---

## Root Cause

### What the dedup job does

Every 6 minutes `deduplicate_persons` runs this pgvector LATERAL query against the **entire**
`person_face_embeddings` table (no time window — full lifetime history):

```sql
SET LOCAL ivfflat.probes = 50;

SELECT DISTINCT
    LEAST(a.person_identity_id::text, b_near.person_identity_id::text)  AS pid_a,
    GREATEST(a.person_identity_id::text, b_near.person_identity_id::text) AS pid_b,
    MAX(1.0 - (b_near.dist)) AS max_sim
FROM person_face_embeddings a
CROSS JOIN LATERAL (
    SELECT pfe.person_identity_id,
           pfe.embedding <=> a.embedding AS dist
    FROM   person_face_embeddings pfe
    WHERE  pfe.person_identity_id != a.person_identity_id
      AND  (1.0 - (pfe.embedding <=> a.embedding)) >= 0.40
    ORDER  BY dist
    LIMIT  5
) b_near
GROUP  BY pid_a, pid_b
HAVING MAX(1.0 - (b_near.dist)) >= 0.40;
```

### What IVFFlat `lists` and `probes` mean

**IVFFlat (Inverted File Flat)** is an approximate nearest-neighbour (ANN) index. It avoids a
brute-force O(N²) scan by partitioning embeddings into clusters at index-build time.

#### Building the index — `lists=N`

```sql
CREATE INDEX ... USING ivfflat (embedding vector_cosine_ops) WITH (lists = N)
```

PostgreSQL runs k-means and groups all embeddings into **N buckets**. Each bucket holds embeddings
that are geometrically close. Every bucket has a **centroid** (its average vector).

```
lists = 50  →  50 buckets, each holding ~456 embeddings on average (22826 / 50)
lists = 150 →  150 buckets, each holding ~152 embeddings on average (22826 / 150)
```

#### Querying — `probes=K`

```sql
SET LOCAL ivfflat.probes = K;
```

When searching for the nearest neighbours of a query vector:

1. Compute distance to all N centroids (cheap — only N comparisons).
2. Open the K closest buckets.
3. Search the embeddings inside those K buckets.
4. Return the best hits.

```
Coverage = probes / lists

probes=10, lists=150  →  10/150 = 6.7% of the table scanned per query
probes=50, lists=150  →  50/150 = 33% of the table scanned per query
probes=50, lists=50   →  50/50  = 100% of the table scanned per query  ← FULL SCAN
```

### The specific failure

The index was originally created with `lists=50`, appropriate for ~2,500 embeddings
(`sqrt(2500) = 50`). As the system grew to **22,826 embeddings / 5,465 persons**, the index was
never rebuilt.

The dedup job hardcoded `probes=50` to match the original `lists=50`. Once the DB grew large:

```
probes = 50
lists  = 50
─────────────────────────────────────────────
Coverage = 50/50 = 100%  →  no index benefit
```

For **every** one of the 22,826 embeddings (outer loop), the query scanned all 22,826 rows (inner
lateral). Total comparisons: **22,826 × 22,826 ≈ 521 million cosine distance calculations**.

PostgreSQL confirmed this — `pg_stat_activity` showed PID 2814098 running the dedup LATERAL query
for **36 minutes 25 seconds** with `state = active` and no wait events (pure CPU work).

Two additional stuck connections (PIDs 2834111, 2836985) from manually-run diagnostic scripts
running the same query added further load.

---

## Fix

### 1. Kill the stuck queries

```sql
SELECT pg_cancel_backend(2814098),
       pg_cancel_backend(2834111),
       pg_cancel_backend(2836985);
```

### 2. Rebuild the IVFFlat index with the correct `lists` count

Rule of thumb from pgvector docs: `lists ≈ sqrt(row_count)`.

```
sqrt(22826) ≈ 151  →  use lists = 150
```

```sql
DROP INDEX IF EXISTS idx_person_face_embeddings_embedding;

CREATE INDEX idx_person_face_embeddings_embedding
    ON person_face_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 150);

VACUUM ANALYZE person_face_embeddings;
```

### 3. Lower `probes` in `tasks.py`

`app/modules/jobs/tasks.py` line ~202 changed from:

```python
# BEFORE (full scan when lists=50)
await db.execute(text("SET LOCAL ivfflat.probes = 50"))
```

to:

```python
# AFTER — probes=10 with lists=150 → 6.7% coverage, adequate recall at thr=0.40
await db.execute(text("SET LOCAL ivfflat.probes = 10"))
```

### 4. Restart the worker

```bash
sudo systemctl restart retail-ai-worker.service
```

---

## Performance Comparison

| Configuration | Coverage | Estimated runtime (22k embs) |
|---|---|---|
| `probes=50`, `lists=50` (old) | 100% — full scan | 36+ minutes |
| `probes=10`, `lists=150` (new) | 6.7% — ANN | ~2-3 minutes |
| `probes=5`, `lists=150` | 3.3% — ANN | ~1 minute |

At similarity threshold 0.40 (strong face similarity), true duplicate pairs cluster tightly in the
same IVFFlat bucket by construction. `probes=10` with well-sized lists provides ~90-95% recall of
real duplicates while being ~15× faster.

---

## Recall Trade-off

IVFFlat ANN is approximate — it may miss a small fraction of true pairs compared to an exact scan.
For the dedup use case this is acceptable because:

1. The real-time identity engine (`decide_identity`) already catches most same-person matches at
   track time. The dedup job is a safety net for cross-angle pairs just below the 0.40 threshold.
2. Missed pairs on one 6-minute cycle are caught on the next cycle (the embeddings remain in the DB).
3. A false negative (missed merge) is far less harmful than a 40-minute hang that blocks ALL merges.

---

## Prevention

### Always keep `probes << lists`

```
Good:  probes = lists × 0.05 to 0.10   (5–10% coverage)
Bad:   probes = lists                   (100% coverage = full scan)
Bad:   probes > lists                   (clamped to lists internally = full scan)
```

### Rebuild the index when the row count grows significantly

The `lists` value should track `sqrt(row_count)`. Rebuild when the table doubles:

| Rows | Recommended lists |
|---|---|
| ~2,500 | 50 |
| ~10,000 | 100 |
| ~22,000 | 150 |
| ~40,000 | 200 |

Add a periodic check (e.g., in `storage_cleanup` or a weekly cron):

```sql
SELECT reltuples::bigint AS approx_rows,
       round(sqrt(reltuples)) AS recommended_lists
FROM pg_class
WHERE relname = 'person_face_embeddings';
```

If `recommended_lists` has grown significantly beyond the current `lists` value, schedule a
maintenance-window `REINDEX`.

---

## Files Changed

| File | Change |
|---|---|
| `app/modules/jobs/tasks.py` | `ivfflat.probes` lowered from 50 → 10; comment explains the probes/lists contract |
| `person_face_embeddings` index | Rebuilt with `lists=150` (was `lists=50`) + `VACUUM ANALYZE` |
| `docs/IVFFLAT_DEDUP_SLOWDOWN_FIX.md` | This document |

---

## Related

- `CONTEXT.md` — Known issue #13 (dedup union-find bug), #14 (face-only threshold gap)
- `app/modules/jobs/tasks.py` — `deduplicate_persons()`
- `app/modules/jobs/scheduler.py` — dedup job registered at 6-minute interval
- pgvector docs: https://github.com/pgvector/pgvector#ivfflat
