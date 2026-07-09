#!/usr/bin/env python3
"""
clean_contaminated_embeddings.py — remove face embeddings from different people
stored under the same PersonIdentity.

Strategy: for each person with >=2 embeddings, compute all pairwise cosine similarities.
Any embedding that has low similarity (< FACE_CONTAMINATION_THRESHOLD, default 0.35) to
the majority cluster is contaminated — it belongs to a DIFFERENT person whose body crop
overlapped the tracked person's during detection.

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/clean_contaminated_embeddings.py [--apply]
"""

import asyncio, argparse, sys, numpy as np
from sqlalchemy import text
from app.core.db.session import AsyncSessionLocal
from app.config import get_settings


async def clean(apply_fix: bool) -> int:
    settings = get_settings()
    threshold = settings.FACE_CONTAMINATION_THRESHOLD  # 0.35

    async with AsyncSessionLocal() as db:
        r = await db.execute(text("""
            SELECT pi.id FROM person_identities pi
            WHERE (SELECT COUNT(*) FROM person_face_embeddings WHERE person_identity_id = pi.id) >= 2
        """))
        person_ids = [row[0] for row in r.fetchall()]

        print(f"\n{'='*60}")
        print(f"  Face contamination cleaner — {'APPLY' if apply_fix else 'DRY RUN'}")
        print(f"  Persons with >=2 embeddings: {len(person_ids)}")
        print(f"  Contamination threshold: {threshold}")
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

            # Normalize embeddings before comparison — InsightFace face
            # embeddings are NOT L2-normalized (norms 12-27).
            for emb in embs:
                _n = np.linalg.norm(emb)
                if _n > 0:
                    emb /= _n

            # Build similarity matrix
            sims = np.zeros((N, N), dtype=np.float32)
            for i in range(N):
                for j in range(i + 1, N):
                    s = float(np.dot(embs[i], embs[j]))
                    sims[i][j] = s
                    sims[j][i] = s

            # Find the largest cluster (mutual sim >= threshold)
            keep_idx = {0}  # always keep top-scored face (rows sorted by score)
            for i in range(1, N):
                compatible = any(sims[i][k] >= threshold for k in keep_idx)
                if compatible:
                    keep_idx.add(i)

            contaminated = set(range(N)) - keep_idx

            if not contaminated:
                continue

            remove_ids = [ids[i] for i in contaminated]
            print(f"  {str(pid)[:12]}  total={N}  keep={len(keep_idx)}  remove={len(contaminated)}")
            for ci in sorted(contaminated):
                sims_to_keep = [f"{sims[ci][k]:.2f}" for k in sorted(keep_idx)]
                print(f"    emb[{ci}]: sims_to_cluster={sims_to_keep}")

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
