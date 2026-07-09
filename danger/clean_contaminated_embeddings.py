#!/usr/bin/env python3
"""
clean_contaminated_embeddings.py — remove face AND body embeddings from different
people stored under the same PersonIdentity.

Strategy (median-based, replaces the old greedy single-linkage version that
chained contamination through borderline "bridge" embeddings):

FACE: for each person with >=2 face embeddings, iteratively remove the
embedding whose MEDIAN cosine similarity to the rest of the cluster is lowest,
until the worst remaining median >= FACE_CONTAMINATION_THRESHOLD (0.35) or the
cluster drops below 2.

BODY: for each person with >=3 body embeddings, iteratively remove the
embedding whose median similarity to the rest is lowest, until the worst
remaining median >= BODY_CONTAMINATION_THRESHOLD (0.50) or the cluster drops
below 3.

Aggressive-reject tuning: any embedding whose median similarity to the rest
is below the threshold is removed — no leniency. Deleting a borderline
same-person embedding is recoverable (reextract_or_delete_faceless.py re-extracts
from track crops); storing a contaminated one is not (it pollutes the identity
and causes future false merges).

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/clean_contaminated_embeddings.py [--apply] [--face-only] [--body-only]
"""

import asyncio
import argparse
import sys
import numpy as np
from sqlalchemy import text
from app.core.db.session import AsyncSessionLocal
from app.config import get_settings


def _parse_embedding(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        return np.array(eval(raw), dtype=np.float32)
    return np.array(raw, dtype=np.float32)


def _iterative_median_removals(embs: list[np.ndarray], threshold: float, min_cluster: int) -> list[int]:
    """Return indices of embeddings to remove, using iterative median outlier removal.

    Repeatedly removes the embedding with the lowest median similarity to the
    rest, until all remaining embeddings have median >= threshold or the
    cluster drops below `min_cluster` size.

    `embs` MUST already be L2-normalized for the dot product to equal cosine sim.
    """
    N = len(embs)
    remove_idx = set()
    active = set(range(N))

    while len(active) >= min_cluster:
        medians = []
        for i in active:
            sims = []
            for j in active:
                if i != j:
                    sims.append(float(np.dot(embs[i], embs[j])))
            medians.append((i, float(np.median(sims)) if sims else 0.0))

        worst_idx, worst_median = min(medians, key=lambda x: x[1])

        if worst_median >= threshold:
            break

        remove_idx.add(worst_idx)
        active.discard(worst_idx)

    return sorted(remove_idx)


async def clean_faces(apply_fix: bool) -> int:
    settings = get_settings()
    threshold = settings.FACE_CONTAMINATION_THRESHOLD

    async with AsyncSessionLocal() as db:
        r = await db.execute(text("""
            SELECT pi.id FROM person_identities pi
            WHERE (SELECT COUNT(*) FROM person_face_embeddings WHERE person_identity_id = pi.id) >= 2
        """))
        person_ids = [row[0] for row in r.fetchall()]

        print(f"\n{'='*70}")
        print(f"  FACE contamination cleanup — {'APPLY' if apply_fix else 'DRY RUN'}")
        print(f"  Persons with >=2 face embeddings: {len(person_ids)}")
        print(f"  Threshold (median sim): {threshold}")
        print(f"{'='*70}\n")

        total_removed = 0
        persons_touched = 0
        for pid in person_ids:
            r2 = await db.execute(text("""
                SELECT id::text, embedding, face_score FROM person_face_embeddings
                WHERE person_identity_id = :pid AND embedding IS NOT NULL
                ORDER BY face_score DESC
            """), {"pid": str(pid)})
            rows = r2.fetchall()
            if len(rows) < 2:
                continue

            ids = [r[0] for r in rows]
            embs = [_parse_embedding(r[1]) for r in rows]

            # L2-normalize (InsightFace embeddings are NOT normalized at extract)
            for emb in embs:
                _n = np.linalg.norm(emb)
                if _n > 0:
                    emb /= _n

            remove_idx = _iterative_median_removals(embs, threshold, min_cluster=2)

            if not remove_idx:
                continue

            persons_touched += 1
            remove_ids = [ids[i] for i in remove_idx]
            keep_ids = [ids[i] for i in range(len(rows)) if i not in remove_idx]

            print(f"  {str(pid)[:12]}  total={len(rows)}  keep={len(keep_ids)}  remove={len(remove_idx)}")
            # Show why each removed embedding was rejected
            for ci in remove_idx:
                sims_to_keep = [f"{float(np.dot(embs[ci], embs[k])):.2f}" for k in range(len(rows)) if k not in remove_idx]
                print(f"    emb[{ci}] score={rows[ci][2]:.3f}  sims_to_cluster={sims_to_keep}")

            if apply_fix:
                await db.execute(text(
                    "DELETE FROM person_face_embeddings WHERE id = ANY(:ids)"
                ), {"ids": remove_ids})
                total_removed += len(remove_idx)

        if apply_fix and total_removed > 0:
            await db.commit()
            print(f"\n  Removed {total_removed} contaminated face embedding(s) across {persons_touched} person(s).")
        elif apply_fix:
            print(f"\n  No contamination found.")
        else:
            print(f"\n  Dry run: would remove {total_removed if False else sum(1 for _ in [])} — see report above. Run with --apply to fix.")

        print(f"{'='*70}\n")
    return total_removed


async def clean_bodies(apply_fix: bool) -> int:
    settings = get_settings()
    threshold = settings.BODY_CONTAMINATION_THRESHOLD

    async with AsyncSessionLocal() as db:
        r = await db.execute(text("""
            SELECT pi.id FROM person_identities pi
            WHERE (SELECT COUNT(*) FROM person_embeddings WHERE person_identity_id = pi.id) >= 3
        """))
        person_ids = [row[0] for row in r.fetchall()]

        print(f"\n{'='*70}")
        print(f"  BODY contamination cleanup — {'APPLY' if apply_fix else 'DRY RUN'}")
        print(f"  Persons with >=3 body embeddings: {len(person_ids)}")
        print(f"  Threshold (median sim): {threshold}")
        print(f"{'='*70}\n")

        total_removed = 0
        persons_touched = 0
        for pid in person_ids:
            r2 = await db.execute(text("""
                SELECT id::text, embedding, crop_quality FROM person_embeddings
                WHERE person_identity_id = :pid AND embedding IS NOT NULL
                ORDER BY crop_quality DESC
            """), {"pid": str(pid)})
            rows = r2.fetchall()
            if len(rows) < 3:
                continue

            ids = [r[0] for r in rows]
            embs = [_parse_embedding(r[1]) for r in rows]

            # OSNet embeddings are L2-normalized at extract; be defensive
            for emb in embs:
                _n = np.linalg.norm(emb)
                if _n > 0:
                    emb /= _n

            remove_idx = _iterative_median_removals(embs, threshold, min_cluster=3)

            if not remove_idx:
                continue

            persons_touched += 1
            remove_ids = [ids[i] for i in remove_idx]

            print(f"  {str(pid)[:12]}  total={len(rows)}  keep={len(rows)-len(remove_idx)}  remove={len(remove_idx)}")
            for ci in remove_idx:
                sims_to_keep = [f"{float(np.dot(embs[ci], embs[k])):.2f}" for k in range(len(rows)) if k not in remove_idx]
                print(f"    emb[{ci}] quality={rows[ci][2]:.3f}  sims_to_cluster={sims_to_keep}")

            if apply_fix:
                await db.execute(text(
                    "DELETE FROM person_embeddings WHERE id = ANY(:ids)"
                ), {"ids": remove_ids})
                total_removed += len(remove_idx)

        if apply_fix and total_removed > 0:
            await db.commit()
            print(f"\n  Removed {total_removed} contaminated body embedding(s) across {persons_touched} person(s).")
        elif apply_fix:
            print(f"\n  No contamination found.")
        else:
            print(f"\n  Dry run: see report above. Run with --apply to fix.")

        print(f"{'='*70}\n")
    return total_removed


async def main(apply_fix: bool, face_only: bool, body_only: bool):
    total = 0
    if not body_only:
        total += await clean_faces(apply_fix)
    if not face_only:
        total += await clean_bodies(apply_fix)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply fixes (default: dry run)")
    parser.add_argument("--face-only", action="store_true", help="Only clean face embeddings")
    parser.add_argument("--body-only", action="store_true", help="Only clean body embeddings")
    args = parser.parse_args()
    asyncio.run(main(args.apply, args.face_only, args.body_only))
