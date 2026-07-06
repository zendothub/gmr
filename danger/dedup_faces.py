#!/usr/bin/env python3
"""
dedup_faces.py
--------------
READ-ONLY script that scans person_identities for duplicate faces.
Finds pairs of identities whose maximum cross-face-similarity exceeds
the FACE_MATCH_THRESHOLD (0.48), meaning they are likely the same
person registered multiple times.

Output: prints identity pairs with their IDs, face embedding counts,
max similarity, and demographics. Does NOT modify any data.

Usage:
    .venv/bin/python dedup_faces.py
"""
import asyncio
import sys

from sqlalchemy import text
from app.core.db.session import AsyncSessionLocal
from app.config import get_settings


THRESHOLD = None  # set in main()


async def find_duplicates():
    global THRESHOLD
    settings = get_settings()
    THRESHOLD = settings.FACE_MATCH_THRESHOLD

    print(f"\n{'='*90}")
    print(f"  DUPLICATE FACE DETECTION (threshold: {THRESHOLD})")
    print(f"{'='*90}\n")

    async with AsyncSessionLocal() as db:
        # Get all persons with face embeddings, ordered by first_seen
        result = await db.execute(text("""
            SELECT pi.id,
                   pi.first_seen_at,
                   pi.last_seen_at,
                   pi.visit_count,
                   pi.gender,
                   pi.estimated_age,
                   pi.age_group,
                   pi.best_face_score,
                   (SELECT count(*) FROM person_face_embeddings pfe
                    WHERE pfe.person_identity_id = pi.id) as face_emb_count
            FROM person_identities pi
            WHERE EXISTS (
                SELECT 1 FROM person_face_embeddings pfe
                WHERE pfe.person_identity_id = pi.id
            )
            ORDER BY pi.first_seen_at
        """))

        persons = result.fetchall()

        if not persons:
            print("  No persons with face embeddings found.\n")
            return

        print(f"  Scanning {len(persons)} persons with face embeddings...\n")

        duplicates = []
        checked = 0

        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                pid_a = persons[i][0]
                pid_b = persons[j][0]

                # Cross-compare ALL face embeddings between the two identities
                sim_result = await db.execute(text("""
                    SELECT max(1.0 - (a.embedding <=> b.embedding)) as max_sim,
                           count(*) as pairs
                    FROM person_face_embeddings a
                    CROSS JOIN person_face_embeddings b
                    WHERE a.person_identity_id = :pid_a
                      AND b.person_identity_id = :pid_b
                """), {"pid_a": str(pid_a), "pid_b": str(pid_b)})

                row = sim_result.fetchone()
                max_sim = float(row[0]) if row[0] is not None else 0.0
                pair_count = int(row[1]) if row[1] is not None else 0

                checked += 1

                if max_sim >= THRESHOLD:
                    duplicates.append({
                        "pid_a": pid_a,
                        "pid_b": pid_b,
                        "max_sim": max_sim,
                        "pairs": pair_count,
                        "info_a": persons[i],
                        "info_b": persons[j],
                    })

        # Print results
        if not duplicates:
            print(f"  No duplicate pairs found (checked {checked} pairs).\n")
            print(f"{'='*90}\n")
            return

        print(f"  Found {len(duplicates)} duplicate pair(s) out of {checked} checked:\n")
        print(f"  {'ID A':<38} {'ID B':<38} {'MaxSim':>8} {'A_faces':>8} {'B_faces':>8}  Gender  Age")
        print(f"  {'-'*38} {'-'*38} {'-'*8} {'-'*8} {'-'*8}  {'-'*7}  {'-'*3}")

        for d in duplicates:
            ia = d["info_a"]
            ib = d["info_b"]
            print(f"  {str(ia[0])[:36]:<38} {str(ib[0])[:36]:<38} {d['max_sim']:>8.4f} "
                  f"{ia[8]:>8} {ib[8]:>8}  "
                  f"{str(ia[4] or '?'):<5}/{str(ib[4] or '?'):<5}  "
                  f"{str(ia[5] or '?')}/{str(ib[5] or '?')}")
            print(f"    A: first_seen={ia[1]}, visits={ia[3]}, face_score={ia[7]:.3f}" if ia[7] else f"    A: first_seen={ia[1]}, visits={ia[3]}")
            print(f"    B: first_seen={ib[1]}, visits={ib[3]}, face_score={ib[7]:.3f}" if ib[7] else f"    B: first_seen={ib[1]}, visits={ib[3]}")
            print()

        print(f"{'='*90}")
        print(f"  Total duplicates: {len(duplicates)}")
        print(f"  These identities SHOULD be merged (same person, multiple registrations).")
        print(f"  Run reset_tracking_data.py to wipe all and start fresh, or merge manually.")
        print(f"{'='*90}\n")


if __name__ == "__main__":
    asyncio.run(find_duplicates())
