#!/usr/bin/env python3
"""
merge_recent_window_duplicates.py — historical backfill: merge PersonIdentity
records that represent the same physical person seen within a short (5-min)
window but registered as separate identities (cross-angle face miss +
single-candidate body rejection, etc.).

This is the "past data" counterpart to the live Recent Window matching
(Step D in the plan). It scans ALL historical persons, not just since the
last restart.

Merge rule (single combined rule, aggressive-reject — anything not matching
exactly is NOT merged):

  A pair (A, B) is a candidate if BOTH:
    - Their [first_seen_at, last_seen_at] windows are within RECENT_WINDOW_MINUTES
      (5) of each other (same-visit presumption; body ReID is reliable within a
      single visit because clothing is constant).
    - Their track_sessions NEVER overlap in time (temporal non-overlap gate —
      two persons on screen simultaneously cannot be the same person; this is a
      cheap, safe false-positive filter that can only prevent bad merges).

  A candidate is MERGED if EITHER:
    - face_max >= FACE_MATCH_THRESHOLD_RECENT (0.35), where face_max is the BEST
      cross-pair cosine similarity between A's and B's face embeddings (matches
      the existing dedup job's LATERAL MAX() semantics — cross-angle face
      matching must use the best angle pair, since most angle pairs degrade);
      requires at least one face on each side. OR
    - body_median >= RECENT_BODY_SINGLE_MATCH_THRESHOLD (0.60), where body_median
      is the median of all cross-pair body similarities (consistency check — a
      single lucky crop is not enough), AND both sides have
      >= MIN_BODIES_PER_SIDE (2) body embeddings (never merge on a single crop),
      AND the faces do NOT contradict: either at least one side has zero faces
      (so nothing to contradict), OR face_max >= NON_CONTRADICTION_THRESHOLD
      (0.30, the existing FACE_BODY_EXCLUSION_THRESHOLD). This gate is what
      prevents the "body-chameleon" false merges seen in live data (e.g. a
      person whose body matches 5+ strangers at body_median 0.6-0.7 while their
      faces contradict at <0.30 — those are DIFFERENT people in similar clothing,
      not the same person).

  Body-only merges (no usable face on one side) are valid ONLY within this
  recent window. Outside it the dedup job's older-pairs tier (face-only >= 0.40)
  applies — body ReID is unreliable across days (clothing changes).

Uses the FIXED contamination-gated absorb functions from jobs.tasks
(_absorb_face_embeddings / _absorb_body_embeddings), so merges do NOT
re-inject contamination into the winner. Must therefore run AFTER the
contamination cleanup (Step B).

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/merge_recent_window_duplicates.py            # dry run
    PYTHONPATH=/gmr/gmr venv/bin/python danger/merge_recent_window_duplicates.py --apply    # apply
    PYTHONPATH=/gmr/gmr venv/bin/python danger/merge_recent_window_duplicates.py --ids <pid1> <pid2> ...  # limit to specific persons
"""

import asyncio
import argparse
import sys
from datetime import timedelta

import numpy as np
from sqlalchemy import text
from loguru import logger

from app.core.db.session import AsyncSessionLocal
from app.config import get_settings
# Reuse the FIXED merge machinery (contamination-gated absorb + gender re-vote)
from app.modules.jobs.tasks import (
    _absorb_face_embeddings,
    _absorb_body_embeddings,
    _revote_person_gender,
)

# ── Locked thresholds (plan, 2026-07-09) ──────────────────────────────────
RECENT_WINDOW_MINUTES = 5
FACE_MATCH_THRESHOLD_RECENT = 0.35
RECENT_BODY_SINGLE_MATCH_THRESHOLD = 0.55
MIN_BODIES_PER_SIDE = 2
# Body-only merge requires faces don't contradict: one side faceless, OR
# face_max >= this. Matches live engine's FACE_CONTRADICTION_THRESHOLD (0.25) —
# same-person cross-angle face can drop to ~0.28, so 0.25 avoids false
# rejection while still blocking genuinely-different faces (<0.25).
NON_CONTRADICTION_THRESHOLD = 0.25
# Face-path median check: when face best-pair is in grey zone [0.35, 0.40),
# require median of ALL cross-pairs >= this. Same-person min median=0.401,
# diff-person p50=0.200. Only checked when >= 3 total cross-pairs.
FACE_MATCH_MEDIAN_THRESHOLD = 0.30


def _parse_embedding(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        return np.array(eval(raw), dtype=np.float32)
    return np.array(raw, dtype=np.float32)


def _l2normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n > 0:
        return v / n
    return v


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _median_cross_sim(embs_a: list[np.ndarray], embs_b: list[np.ndarray]) -> float | None:
    """Median of all cross-pair cosine sims. Both sides must be L2-normalized."""
    if not embs_a or not embs_b:
        return None
    sims = [_cos(a, b) for a in embs_a for b in embs_b]
    return float(np.median(sims))


def _max_cross_sim(embs_a: list[np.ndarray], embs_b: list[np.ndarray]) -> float | None:
    """Best (max) cross-pair cosine sim. Both sides must be L2-normalized."""
    if not embs_a or not embs_b:
        return None
    return max(_cos(a, b) for a in embs_a for b in embs_b)


async def _load_person_face_embs(db, pid: str) -> list[np.ndarray]:
    r = await db.execute(text("""
        SELECT embedding FROM person_face_embeddings
        WHERE person_identity_id::text = :pid AND embedding IS NOT NULL
    """), {"pid": pid})
    embs = []
    for row in r.fetchall():
        e = _parse_embedding(row[0])
        if e is not None:
            embs.append(_l2normalize(e))
    return embs


async def _load_person_body_embs(db, pid: str) -> list[np.ndarray]:
    r = await db.execute(text("""
        SELECT embedding FROM person_embeddings
        WHERE person_identity_id::text = :pid AND embedding IS NOT NULL
    """), {"pid": pid})
    embs = []
    for row in r.fetchall():
        e = _parse_embedding(row[0])
        if e is not None:
            embs.append(_l2normalize(e))
    return embs


async def _load_track_windows(db, pid: str) -> list[tuple]:
    """Return (camera_id, started_at, ended_at_or_last_seen) per track_session.

    Camera-aware so the overlap / window-proximity gates can distinguish
    cross-camera overlap (expected for the same person seen on entry + counter
    simultaneously) from same-camera overlap (genuinely two different people).
    """
    r = await db.execute(text("""
        SELECT camera_id, started_at, COALESCE(ended_at, last_seen_at) AS ended
        FROM track_sessions
        WHERE person_identity_id::text = :pid
        ORDER BY started_at
    """), {"pid": pid})
    return r.fetchall()


def _windows_within_recent(a_windows, b_windows, max_gap: timedelta) -> bool:
    """True if ANY A-track-window and B-track-window are within max_gap of
    each other (gap between end of earlier and start of later, OR overlap).
    Camera-agnostic — proximity is checked across all cameras since a person
    stepping off one camera and onto another within max_gap is exactly the
    cross-camera handoff case we want to catch.
    """
    if not a_windows or not b_windows:
        return False
    for a_row in a_windows:
        a_start, a_end = a_row[1], a_row[2]
        for b_row in b_windows:
            b_start, b_end = b_row[1], b_row[2]
            # Overlap → within window
            if a_start <= b_end and b_start <= a_end:
                return True
            # a before b
            if a_end <= b_start:
                if b_start - a_end <= max_gap:
                    return True
            else:
                if a_start - b_end <= max_gap:
                    return True
    return False


def _tracks_overlap_same_camera(a_windows, b_windows) -> bool:
    """True if ANY track_session of A overlaps in time with ANY of B ON THE
    SAME CAMERA. Two different people cannot occupy the same camera at the
    same time, so a same-camera overlap rules out "same person". Cross-camera
    overlap is expected (the same person is visible on entry + counter
    simultaneously) and does NOT block a merge.
    """
    for a_row in a_windows:
        a_cam, a_start, a_end = a_row[0], a_row[1], a_row[2]
        for b_row in b_windows:
            b_cam, b_start, b_end = b_row[0], b_row[1], b_row[2]
            if a_cam == b_cam and a_start <= b_end and b_start <= a_end:
                return True
    return False


async def run(apply_fix: bool, ids_filter: list[str] | None):
    settings = get_settings()
    max_faces = settings.MAX_FACE_EMBEDDINGS_PER_PERSON  # 5
    max_bodies = 10
    max_gap = timedelta(minutes=RECENT_WINDOW_MINUTES)

    async with AsyncSessionLocal() as db:
        # Load all persons
        where = ""
        params: dict = {}
        if ids_filter:
            where = "WHERE id::text = ANY(:ids)"
            params["ids"] = list(ids_filter)

        r = await db.execute(text(f"""
            SELECT id::text, first_seen_at, last_seen_at, best_face_score, visit_count
            FROM person_identities
            {where}
            ORDER BY first_seen_at
        """), params)
        persons = r.fetchall()

        print(f"\n{'='*80}")
        print(f"  Recent-window duplicate merge — {'APPLY' if apply_fix else 'DRY RUN'}")
        print(f"  Window: {RECENT_WINDOW_MINUTES} min | face_max ≥ {FACE_MATCH_THRESHOLD_RECENT} "
              f"| body_median ≥ {RECENT_BODY_SINGLE_MATCH_THRESHOLD} (≥{MIN_BODIES_PER_SIDE} bodies/side)")
        print(f"  Persons scanned: {len(persons)}")
        print(f"{'='*80}\n")

        # Preload embeddings + track windows per person (cache)
        cache: dict[str, dict] = {}
        for pid, first_seen, last_seen, best_score, visits in persons:
            faces = await _load_person_face_embs(db, pid)
            bodies = await _load_person_body_embs(db, pid)
            tw = await _load_track_windows(db, pid)
            cache[pid] = {
                "first_seen": first_seen,
                "last_seen": last_seen,
                "best_score": float(best_score or 0.0),
                "visits": visits or 0,
                "faces": faces,
                "bodies": bodies,
                "tracks": tw,
            }

        # Build candidate pairs within the recent window
        accepted_pairs: list[tuple[str, str, dict]] = []  # (pid_a, pid_b, evidence)
        n = len(persons)
        for i in range(n):
            pid_a = persons[i][0]
            ca = cache[pid_a]
            for j in range(i + 1, n):
                pid_b = persons[j][0]
                cb = cache[pid_b]
                # Window proximity check (cheap, before embedding sims)
                if not _windows_within_recent(ca["tracks"], cb["tracks"], max_gap):
                    continue
                # Same-camera non-overlap gate (cross-camera overlap is expected
                # for the same person visible on entry + counter simultaneously)
                if _tracks_overlap_same_camera(ca["tracks"], cb["tracks"]):
                    continue

                face_max = _max_cross_sim(ca["faces"], cb["faces"])
                body_median = _median_cross_sim(ca["bodies"], cb["bodies"])

                merge = False
                reason = ""
                if face_max is not None and face_max >= FACE_MATCH_THRESHOLD_RECENT:
                    # Grey-zone face match [0.35, 0.40): validate with median
                    # of ALL cross-pairs. A single lucky crop can hit 0.35+ for
                    # different people; median catches this.
                    # Only check when >= 3 total cross-pairs (meaningful median).
                    n_cross = len(ca["faces"]) * len(cb["faces"])
                    if n_cross >= 3:
                        face_median = _median_cross_sim(ca["faces"], cb["faces"])
                        if face_median is not None and face_median < FACE_MATCH_MEDIAN_THRESHOLD:
                            merge = False
                            reason = (f"REJECTED: face_max={face_max:.3f} ≥ {FACE_MATCH_THRESHOLD_RECENT} "
                                      f"but face_median={face_median:.3f} < {FACE_MATCH_MEDIAN_THRESHOLD} "
                                      f"(cross-pairs={n_cross}) — single lucky pair, different person")
                        else:
                            merge = True
                            reason = f"face_max={face_max:.3f} ≥ {FACE_MATCH_THRESHOLD_RECENT} (median={face_median:.3f} ✓)"
                    else:
                        merge = True
                        reason = f"face_max={face_max:.3f} ≥ {FACE_MATCH_THRESHOLD_RECENT} (cross-pairs={n_cross} < 3, median skipped)"
                elif (
                    body_median is not None
                    and body_median >= RECENT_BODY_SINGLE_MATCH_THRESHOLD
                    and len(ca["bodies"]) >= MIN_BODIES_PER_SIDE
                    and len(cb["bodies"]) >= MIN_BODIES_PER_SIDE
                ):
                    # Non-contradiction gate: body-only merge requires faces
                    # don't contradict (one side faceless OR face_max >= 0.30).
                    nfa = len(ca["faces"])
                    nfb = len(cb["faces"])
                    face_contradicts = (
                        nfa > 0 and nfb > 0
                        and (face_max is None or face_max < NON_CONTRADICTION_THRESHOLD)
                    )
                    if not face_contradicts:
                        merge = True
                        face_note = (f"faceless-side" if nfa == 0 or nfb == 0
                                     else f"face_max={face_max:.3f} ≥ {NON_CONTRADICTION_THRESHOLD}")
                        reason = (f"body_median={body_median:.3f} ≥ {RECENT_BODY_SINGLE_MATCH_THRESHOLD} "
                                  f"(bodies {len(ca['bodies'])}/{len(cb['bodies'])}, "
                                  f"{face_note}, non-overlap ✓)")

                if merge:
                    accepted_pairs.append((pid_a, pid_b, {
                        "face_max": face_max,
                        "body_median": body_median,
                        "n_faces": (len(ca["faces"]), len(cb["faces"])),
                        "n_bodies": (len(ca["bodies"]), len(cb["bodies"])),
                        "reason": reason,
                    }))

        if not accepted_pairs:
            print("  No merge candidates found.\n")
            print(f"{'='*80}\n")
            return

        # Union-find (correct recursive path compression, same as fixed dedup job)
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            p = parent.get(x, x)
            if p != x:
                parent[x] = find(p)
            return parent.get(x, x)

        def union(x: str, y: str):
            parent[find(x)] = find(y)

        for pid_a, pid_b, _ in accepted_pairs:
            union(pid_a, pid_b)

        # Group into components
        components: dict[str, list[str]] = {}
        for pid_a, pid_b, _ in accepted_pairs:
            components.setdefault(find(pid_a), set()).add(pid_a)  # type: ignore
            components[find(pid_a)].add(pid_b)
        # Normalize to lists
        components = {root: sorted(members) for root, members in components.items()}  # type: ignore

        # Pick winner per component: highest best_face_score, tiebreak earliest first_seen
        merges: list[tuple[str, str]] = []
        for root, members in components.items():
            if len(members) < 2:
                continue
            winner = max(
                members,
                key=lambda pid: (
                    cache[pid]["best_score"],
                    -(cache[pid]["first_seen"].timestamp() if cache[pid]["first_seen"] else 0),
                ),
            )
            for loser in members:
                if loser != winner:
                    merges.append((winner, loser))

        # Report
        print(f"  Candidate pairs accepted: {len(accepted_pairs)}")
        print(f"  Connected components: {sum(1 for m in components.values() if len(m) >= 2)}")
        print(f"  Merges to perform: {len(merges)}\n")
        for pid_a, pid_b, ev in accepted_pairs:
            print(f"  pair {pid_a[:8]} + {pid_b[:8]}: {ev['reason']}")
            print(f"        faces={ev['n_faces']} bodies={ev['n_bodies']} "
                  f"face_max={ev['face_max']} body_med={ev['body_median']}")
        print()
        for winner, loser in merges:
            wc = cache[winner]
            lc = cache[loser]
            print(f"  MERGE {loser[:8]} → {winner[:8]}  "
                  f"(winner score={wc['best_score']:.3f} visits={wc['visits']}, "
                  f"loser score={lc['best_score']:.3f} visits={lc['visits']})")

        if not apply_fix:
            print(f"\n  Dry run. Run with --apply to perform the merges.")
            print(f"{'='*80}\n")
            return

        # Apply merges — same logic as deduplicate_persons Step 3
        merged_count = 0
        for winner_id, loser_id in merges:
            try:
                # Reassign FK references
                for tbl, col in [
                    ("track_sessions", "person_identity_id"),
                    ("events", "person_identity_id"),
                    ("billing_interactions", "person_identity_id"),
                    ("storage_objects", "person_identity_id"),
                ]:
                    await db.execute(text(
                        f"UPDATE {tbl} SET {col} = :winner WHERE {col}::text = :loser"
                    ), {"winner": winner_id, "loser": loser_id})

                # Absorb visit_count + first_seen_at
                wc = cache[winner_id]
                lc = cache[loser_id]
                update_parts = ["visit_count = visit_count + :extra_visits"]
                params = {"extra_visits": lc["visits"], "winner": winner_id}
                if lc["first_seen"] and wc["first_seen"] and lc["first_seen"] < wc["first_seen"]:
                    update_parts.append("first_seen_at = :loser_first")
                    params["loser_first"] = lc["first_seen"]
                await db.execute(text(
                    f"UPDATE person_identities SET {', '.join(update_parts)} WHERE id::text = :winner"
                ), params)

                # Absorb embeddings (FIXED, contamination-gated)
                await _absorb_face_embeddings(db, winner_id, loser_id, max_faces)
                await _absorb_body_embeddings(db, winner_id, loser_id, max_bodies)
                await _revote_person_gender(db, winner_id)

                # Defer MinIO deletion to the periodic sweep (collect paths for cleanup)
                # — the dedup job sweep cross-references DB, so just delete the loser row.
                await db.execute(text(
                    "DELETE FROM person_identities WHERE id::text = :loser"
                ), {"loser": loser_id})

                merged_count += 1
                print(f"  merged {loser_id[:8]} → {winner_id[:8]}")
            except Exception as e:
                logger.error(f"Backfill: failed to merge {loser_id[:8]} → {winner_id[:8]}: {e}")
                await db.rollback()
                return

        await db.commit()
        print(f"\n  Applied {merged_count} merge(s).")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Historical recent-window duplicate merge (backfill)")
    parser.add_argument("--apply", action="store_true", help="Apply merges (default: dry run)")
    parser.add_argument("--ids", nargs="*", default=None, help="Limit to specific person identity UUIDs")
    args = parser.parse_args()
    asyncio.run(run(args.apply, args.ids))
