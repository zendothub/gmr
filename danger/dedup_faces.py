#!/usr/bin/env python3
"""
dedup_faces.py
--------------
READ-ONLY script that scans person_identities for duplicate faces.

Uses an efficient pgvector LATERAL query (O(N·log N) via IVFFlat index) instead
of the old O(N²) cross-join approach that timed out on large datasets.

Finds pairs of identities whose maximum cross-face-similarity exceeds
FACE_MATCH_THRESHOLD (default 0.48), meaning they are likely the same
person registered multiple times from different camera angles.

Output: prints identity pairs with their IDs, face embedding counts,
max similarity, and demographics.  Does NOT modify any data.

Usage:
    PYTHONPATH=/gmr/gmr .venv/bin/python dedup_faces.py [--threshold 0.48]
"""
import asyncio
import sys
import argparse

from sqlalchemy import text
from app.core.db.session import AsyncSessionLocal
from app.config import get_settings


async def find_duplicates(threshold: float):
    print(f"\n{'='*100}")
    print(f"  DUPLICATE FACE DETECTION  (threshold: {threshold})")
    print(f"{'='*100}\n")

    async with AsyncSessionLocal() as db:
        # ── Person count ────────────────────────────────────────────────────────
        count_result = await db.execute(text(
            "SELECT COUNT(*) FROM person_identities WHERE EXISTS "
            "(SELECT 1 FROM person_face_embeddings pfe WHERE pfe.person_identity_id = person_identities.id)"
        ))
        total_with_face = count_result.scalar()
        print(f"  Persons with face embeddings: {total_with_face}\n")

        if total_with_face == 0:
            print("  No persons with face embeddings found.\n")
            return

        # ── Efficient duplicate discovery via LATERAL ───────────────────────────
        # Probes=50 ensures the IVFFlat index scans enough buckets for high recall.
        await db.execute(text("SET LOCAL ivfflat.probes = 50"))

        pairs_result = await db.execute(text("""
            SELECT
                LEAST(a.person_identity_id::text, b_near.person_identity_id::text)   AS pid_a,
                GREATEST(a.person_identity_id::text, b_near.person_identity_id::text) AS pid_b,
                MAX(1.0 - b_near.dist) AS max_sim
            FROM person_face_embeddings a
            CROSS JOIN LATERAL (
                SELECT pfe.person_identity_id,
                       pfe.embedding <=> a.embedding AS dist
                FROM   person_face_embeddings pfe
                WHERE  pfe.person_identity_id != a.person_identity_id
                  AND  (1.0 - (pfe.embedding <=> a.embedding)) >= :threshold
                ORDER  BY dist
                LIMIT  5
            ) b_near
            GROUP  BY pid_a, pid_b
            HAVING MAX(1.0 - b_near.dist) >= :threshold
            ORDER  BY max_sim DESC
        """), {"threshold": threshold})

        pairs = pairs_result.fetchall()

        if not pairs:
            print(f"  No duplicate pairs found (threshold={threshold}).\n")
            print(f"{'='*100}\n")
            return

        # ── Fetch metadata for all involved IDs ─────────────────────────────────
        all_ids = list({str(r[0]) for r in pairs} | {str(r[1]) for r in pairs})

        meta_result = await db.execute(text("""
            SELECT
                pi.id::text,
                pi.first_seen_at,
                pi.last_seen_at,
                pi.visit_count,
                pi.gender,
                pi.estimated_age,
                pi.age_group,
                pi.best_face_score,
                COUNT(pfe.id) AS face_emb_count,
                COUNT(ts.id)  AS track_count
            FROM person_identities pi
            LEFT JOIN person_face_embeddings pfe ON pfe.person_identity_id = pi.id
            LEFT JOIN track_sessions          ts  ON ts.person_identity_id  = pi.id
            WHERE pi.id::text = ANY(:ids)
            GROUP BY pi.id
        """), {"ids": all_ids})

        meta = {r[0]: r for r in meta_result.fetchall()}

        # ── Print results ────────────────────────────────────────────────────────
        print(f"  Found {len(pairs)} duplicate pair(s):\n")
        fmt = "  {:<38} {:<38} {:>8}  {:>7}  {:>7}  {:<6}/{:<6}  {}/{}"
        print(fmt.format("ID A", "ID B", "MaxSim", "A_faces", "B_faces",
                         "GenderA", "GenderB", "AgeA", "AgeB"))
        print("  " + "-" * 98)

        for row in pairs:
            pid_a, pid_b, max_sim = str(row[0]), str(row[1]), float(row[2])
            ia = meta.get(pid_a)
            ib = meta.get(pid_b)
            if not ia or not ib:
                continue

            print(fmt.format(
                pid_a[:36], pid_b[:36], f"{max_sim:.4f}",
                int(ia[8]), int(ib[8]),
                str(ia[4] or "?"), str(ib[4] or "?"),
                str(ia[5] or "?"), str(ib[5] or "?"),
            ))
            score_a = f"{ia[7]:.3f}" if ia[7] else "n/a"
            score_b = f"{ib[7]:.3f}" if ib[7] else "n/a"
            print(f"    A: first={ia[1]}  visits={ia[3]}  tracks={ia[9]}  face_score={score_a}")
            print(f"    B: first={ib[1]}  visits={ib[3]}  tracks={ib[9]}  face_score={score_b}")
            print()

        print(f"{'='*100}")
        print(f"  Total duplicate pairs: {len(pairs)}")
        print(f"  These identities SHOULD be merged (same person, multiple registrations).")
        print(f"  The periodic dedup job (every 10 min) will merge them automatically.")
        print(f"  Run reset_tracking_data.py --yes for a full clean slate.")
        print(f"{'='*100}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan for duplicate person identities.")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Face similarity threshold (default: FACE_MATCH_THRESHOLD from config)"
    )
    args = parser.parse_args()

    settings = get_settings()
    threshold = args.threshold if args.threshold is not None else settings.FACE_MATCH_THRESHOLD

    asyncio.run(find_duplicates(threshold))
