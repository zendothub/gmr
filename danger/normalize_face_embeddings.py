#!/usr/bin/env python3
"""
normalize_face_embeddings.py — L2-normalize all existing face embeddings in DB.

InsightFace buffalo_l embeddings were stored un-normalized (norms 12-27).
This one-time migration script normalizes all existing person_face_embeddings
rows to unit length (norm=1.0) so that:
  1. Future np.dot() comparisons work directly as cosine similarity
  2. pgvector index queries remain unaffected (they normalize internally)

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/normalize_face_embeddings.py [--apply]
"""

import asyncio
import sys
import argparse
import numpy as np
from sqlalchemy import text
from app.core.db.session import AsyncSessionLocal


async def normalize(apply_fix: bool):
    async with AsyncSessionLocal() as db:
        r = await db.execute(text("""
            SELECT id::text, embedding FROM person_face_embeddings
            WHERE embedding IS NOT NULL
        """))
        rows = r.fetchall()

        print(f"\n{'='*60}")
        print(f"  Face embedding normalization — {'APPLY' if apply_fix else 'DRY RUN'}")
        print(f"  Total face embeddings: {len(rows)}")
        print(f"{'='*60}\n")

        normalized = 0
        already_normalized = 0
        for row in rows:
            row_id = row[0]
            raw = row[1]
            if isinstance(raw, str):
                emb = np.array(eval(raw), dtype=np.float32)
            else:
                emb = np.array(raw, dtype=np.float32)

            norm = float(np.linalg.norm(emb))
            if abs(norm - 1.0) < 0.01:
                already_normalized += 1
                continue

            emb_norm = emb / norm if norm > 0 else emb
            normalized += 1

            if apply_fix:
                await db.execute(text(
                    "UPDATE person_face_embeddings SET embedding = :emb WHERE id = :id"
                ), {"emb": str(emb_norm.tolist()), "id": row_id})

            if normalized <= 10 or normalized % 100 == 0:
                print(f"  {row_id[:12]}  norm={norm:.4f} → 1.0000")

        if apply_fix and normalized > 0:
            await db.commit()

            # Rebuild the IVFFlat index — the embedding values changed so the
            # index's cluster assignments are stale. Without this, pgvector
            # LATERAL queries return non-deterministic results.
            print("\n  Rebuilding IVFFlat index...")
            from app.core.db.session import sync_engine
            with sync_engine.connect() as conn:
                conn.execution_options(isolation_level="AUTOCOMMIT")
                conn.execute(text("REINDEX INDEX idx_person_face_embeddings_embedding"))
                conn.execute(text("VACUUM ANALYZE person_face_embeddings"))
            print("  Index rebuilt + vacuumed.")

        print(f"\n  Normalized: {normalized}")
        print(f"  Already normalized: {already_normalized}")
        if not apply_fix:
            print(f"\n  Dry run. Run with --apply to normalize.")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    sys.exit(asyncio.run(normalize(args.apply)))
