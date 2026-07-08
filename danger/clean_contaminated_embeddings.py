#!/usr/bin/env python3
"""
clean_contaminated_embeddings.py — remove face embeddings from different people
stored under the same PersonIdentity.

Strategy: for each person with ≥2 embeddings, compute all pairwise cosine similarities.
Any embedding that has negative similarity (< 0) to any other embedding for the same
identity is contaminated — it belongs to a DIFFERENT person whose body crop overlapped
the tracked person's during detection.

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/clean_contaminated_embeddings.py [--apply]
"""

import asyncio, argparse, sys, numpy as np
from sqlalchemy import text
from app.core.db.session import AsyncSessionLocal


async def clean(apply_fix: bool) -> int:
    async with AsyncSessionLocal() as db:
        r = await db.execute(text("""
            SELECT pi.id FROM person_identities pi
            WHERE (SELECT COUNT(*) FROM person_face_embeddings WHERE person_identity_id = pi.id) >= 2
        """))
        person_ids = [row[0] for row in r.fetchall()]

        print(f"\n{'='*60}")
        print(f"  Face contamination cleaner — {'APPLY' if apply_fix else 'DRY RUN'}")
        print(f"  Persons with ≥2 embeddings: {len(person_ids)}")
        print(f"{'='*60}\n")

        total_removed = 0
        for pid in person_ids:
            r2 = await db.execute(text("""
                SELECT id::text, embedding FROM person_face_embeddings
                WHERE person_identity_id = :pid AND embedding IS NOT NULL
            """), {"pid": str(pid)})
            rows = r2.fetchall()
            if len(rows) < 2:
                continue

            N = len(rows)
            ids = [r[0] for r in rows]
            embs = []
            for r in rows:
                if isinstance(r[1], str):
                    embs.append(np.array(eval(r[1]), dtype=np.float32))
                else:
                    embs.append(np.array(r[1], dtype=np.float32))

            # Find contaminated: any embedding with negative sim to another
            contaminated = set()
            for i in range(N):
                for j in range(i + 1, N):
                    sim = float(np.dot(embs[i], embs[j]))
                    if sim < 0.0:
                        # The one with lower avg similarity to the rest is the contaminant
                        i_avg = np.mean([float(np.dot(embs[i], embs[k])) for k in range(N) if k != i])
                        j_avg = np.mean([float(np.dot(embs[j], embs[k])) for k in range(N) if k != j])
                        if i_avg < j_avg:
                            contaminated.add(i)
                        else:
                            contaminated.add(j)

            if not contaminated:
                continue

            remove_ids = [ids[i] for i in contaminated]
            print(f"  {str(pid)[:12]}  total={N}  remove={len(contaminated)}")
            for ci in sorted(contaminated):
                sims = [f"{float(np.dot(embs[ci], embs[k])):.2f}" for k in range(N) if k != ci]
                print(f"    emb[{ci}]: sims_to_others={sims}")

            if apply_fix:
                await db.execute(text(
                    "DELETE FROM person_face_embeddings WHERE id = ANY(:ids)"
                ), {"ids": remove_ids})
                total_removed += len(contaminated)

        if apply_fix and total_removed > 0:
            await db.commit()
            print(f"\n  Removed {total_removed} contaminated face embedding(s).")
        elif apply_fix:
            print(f"\n  No contamination found.")
        else:
            print(f"\n  Dry run. Run with --apply to remove.")

        print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    sys.exit(asyncio.run(clean(args.apply)))
