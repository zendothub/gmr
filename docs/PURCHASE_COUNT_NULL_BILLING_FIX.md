# Purchase Count Under-count — Null Billing Person Fix

**Branch:** `bugfix/purchase-count/abdur`  
**Date:** 2026-07-29  
**Status:** Live fix landed in `camera_worker.py`; historical backfill script available (dry-run default)

---

## 1. Main issue

### What the store saw

Purchase / billing counts in Retail Eye were **lower than the pharmacy’s reported sales**.

Example pattern:

| Source | Count (example day) |
|--------|--------------------:|
| Store reported purchases | 40 |
| DB guest DISTINCT purchasers | 21 |
| Gap | −19 |

Longer multi-day gaps (e.g. Jul 16 store **106** vs DB **46**) showed the same class of problem: the system often **under-counts unique paying customers**.

Analytics purchase metric:

```text
COUNT(DISTINCT billing_interactions.person_identity_id)
WHERE person is not staff
```

**Rows with `person_identity_id = NULL` never count.**

---

### Root cause A — Billing fires before identity (primary fix)

Pipeline order on each frame:

```text
1. Update zones / dwell
2. Evaluate billing rule  ← snapshots track.person_identity_id (often NULL)
3. ReID / decide_identity ← may assign person_id later same frame or later window
4. Persist billing_interaction with the snapshot
```

If a guest stands at the counter long enough to hit the billing dwell threshold (**50s**) **before** face/body identity is resolved:

1. A `billing_interactions` row is written with **`person_identity_id = NULL`**
2. ReID later sets `track_sessions.person_identity_id` and updates `person_entered_view`
3. **The BI row was never updated**
4. Analytics ignore the NULL → **silent under-count**

This is a correctness bug independent of “how long they stayed.”

```text
Physical visit ≥ 50s at counter
        │
        ▼
  BI row written (person_id = NULL)
        │
        ▼
  Identity resolves on same or later frame
        │
        ▼
  Session has person_id; BI still NULL  ← purchase lost
```

---

### Root cause B — Fragmented counter tracks (related, not fully fixed here)

ByteTrack often **splits one continuous counter visit** into several track IDs (occlusion, 5s stale timeout, staff blocking the view).

Effects:

| Effect | Result |
|--------|--------|
| Dwell timer resets on each new track | Fragments may never each hit 50s alone |
| Identity may attach only to short slices | Long unassigned fragments stay `person_id` null |
| Billing may attach to unassigned track | BI with NULL person or wrong identity ceiling |

Case study (`3235f0e9`, ~Jul 24): visual counter dwell >30s, no purchase credit; assigned tracks were only short slices; **body ReID** matched nearby long **unassigned** counter tracks as the same person.

Jul 27 read-only audit: gated same-camera stitch (gap ≤60s, body ≥0.80, bbox near) recovered only modest day-level guest purchases (~+4–6). Lowering dwell 50→30 recovered more on already-identified people but still did not match store bill totals.

**This branch fixes root cause A fully in live code.** Track stitch (root cause B) remains deferred because of staff-uniform body false-positive risk and lower proven recovery vs null-BI repair.

---

## 2. Approach

### Goals

1. Every `billing_interaction` that belongs to a resolved track session should carry that session’s `person_identity_id`.
2. Same-frame order race (rules before ReID) must not permanently stamp NULL on new BIs.
3. Historical NULL BIs (session already has person) must be repairable offline without changing live rules or dwell threshold.
4. Do **not** lower purchase dwell from 50s solely to chase store totals (store “bills” ≠ always “unique people once”).

### Non-goals (this change)

- Collapsing two ByteTrack IDs into one live track / carrying dwell across gaps  
- Changing `dwell_threshold_seconds` (still **50s** in DB rule)  
- Counting NULL persons in analytics  
- Automated staff/guest body merge without face gates  

---

### Live fix (`app/modules/ai_runtime/camera_worker.py`)

#### (1) Same-frame snap — `_refresh_event_person_ids`

After `_run_reid` finishes for the batch, and **before** `_persist_events`:

- Build `track_session_id → person_identity_id` from active tracks  
- For each pending rule/zone event with `person_identity_id is None`, fill from that map  

So if identity resolves in the same persist batch that first fires billing, the BI insert gets the person.

#### (2) Deferred backfill — `_backfill_null_person_fks`

When identity is assigned on ReID window fire **or** track close:

```sql
UPDATE billing_interactions
SET person_identity_id = :pid
WHERE track_session_id = :sid
  AND person_identity_id IS NULL;

UPDATE events
SET person_identity_id = :pid
WHERE track_session_id = :sid
  AND person_identity_id IS NULL;
```

Also run on close even if identity was already known earlier (safety net for any earlier null BIs on that session).

```text
Frame N: dwell ≥ 50s, person still None → (eval) rule event pid=None
Frame N: ReID resolves person P
Frame N: refresh event → pid=P → BI insert with P     ✓ same-frame

Frame N: dwell ≥ 50s, person None → BI insert NULL
Frame N+k: ReID resolves P → backfill BI → person P   ✓ deferred
```

Only **NULL** person rows are updated (never overwrite an existing person_id).

---

### Historical repair (`danger/backfill_null_billing_person.py`)

For rows already in the DB before the live fix:

```text
null BI  JOIN  track_sessions
WHERE bi.person_identity_id IS NULL
  AND ts.person_identity_id IS NOT NULL
→ SET bi.person_identity_id = ts.person_identity_id
```

| Command | Effect |
|---------|--------|
| `python -m danger.backfill_null_billing_person` | Dry-run report |
| `… --days 14` | Limit to last 14 days |
| `… --since 2026-07-20` | Since date (IST if naive) |
| `… --apply` | Write updates |

**Review dry-run first.** Then apply in a maintenance window if desired.

---

### What we deliberately did not change

| Item | Why |
|------|-----|
| Dwell threshold 50s | CONTEXT #27: thr alone cannot hit store bill counts; purity for real checkout |
| Analytics formula | Still DISTINCT non-staff person — intentional (staff rain) |
| Live track stitch | Needs strong gates; body alone unsafe for uniformed staff |

---

## 3. Deploy & verify

```bash
# 1) Deploy code
sudo systemctl restart retail-ai.service

# 2) Optional historical repair
cd /gmr/gmr
./venv/bin/python -m danger.backfill_null_billing_person --days 14          # dry-run
./venv/bin/python -m danger.backfill_null_billing_person --days 14 --apply   # after review
```

**Log signals after restart:**

```text
Backfilled null person FKs session=… person=… billing=N events=M
```

**DB checks:**

```sql
-- Remaining null BIs where session already has a person (should go to 0 after apply)
SELECT COUNT(*)
FROM billing_interactions bi
JOIN track_sessions ts ON ts.id = bi.track_session_id
WHERE bi.person_identity_id IS NULL
  AND ts.person_identity_id IS NOT NULL;

-- Still-null BIs (identity never resolved on that track — different problem)
SELECT COUNT(*)
FROM billing_interactions
WHERE person_identity_id IS NULL;
```

---

## 4. Files touched

| File | Role |
|------|------|
| `app/modules/ai_runtime/camera_worker.py` | Live snap + backfill |
| `danger/backfill_null_billing_person.py` | Historical repair |
| `tests/test_camera_worker.py` | Unit tests for refresh / backfill guards |
| `CONTEXT.md` §28 | Cross-session memory |
| `docs/PURCHASE_COUNT_NULL_BILLING_FIX.md` | This document |

---

## 5. Residual gaps (future work)

1. **Same-camera track stitch** — reconnect fragments within ~60s with high body sim + bbox proximity + anti-staff gates; optionally **carry zone dwell** so split 28s+25s visits can still hit 50s.  
2. **Identity coverage** — unassigned long counter tracks with no person ever (body-only create / attach); partially addressed by body-only path + backfill scripts.  
3. **Metric definition** — store “number of bills” vs system “unique people with ≥1 counted purchase”; imperfect alignment even with perfect vision.

---

## 6. One-line summary

**Main issue:** purchases fired with `person_id=NULL` because billing ran before (or without backfill of) identity — analytics dropped those rows.  
**Approach:** after ReID, fill person on pending events and UPDATE any prior null BI/events for that track session; offline script for history.
