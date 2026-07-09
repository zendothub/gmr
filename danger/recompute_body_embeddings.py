#!/usr/bin/env python3
"""
recompute_body_embeddings.py — Recompute all person_embeddings from MinIO
crops using the fixed OSNet (MSMT17) weights.

All existing body embeddings were computed with broken weights (ImageNet
backbone + random fc head). This script downloads each embedding's source
crop from MinIO, re-extracts the embedding with the trained model, and
UPDATEs the row in place. Preserves person_identity_id, camera_id,
crop_quality, crop_path, captured_at — only the embedding vector changes.

After recomputation, rebuilds the IVFFlat index (CONTEXT.md issue #9).

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/recompute_body_embeddings.py [--dry-run]
"""

import asyncio
import sys
import argparse
import numpy as np
from sqlalchemy import text

from app.core.db.session import AsyncSessionLocal
from app.modules.storage.minio_client import get_client, BUCKET_PREFIX
from app.modules.reid.osnet_extractor import get_shared_extractor


def _download_crop(minio_client, crop_path: str):
    if not crop_path:
        return None
    try:
        key = crop_path
        if key.startswith(f"{BUCKET_PREFIX}/"):
            key = key[len(BUCKET_PREFIX) + 1:]
        if "/" in key and not key.startswith("crops/"):
            key = key.split("/", 1)[1]
        resp = minio_client.get_object(BUCKET_PREFIX, key)
        data = resp.read()
        resp.close()
        resp.release_conn()
        import cv2
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


async def recompute(dry_run: bool = False):
    print("\n  Loading OSNet (fixed weights)...")
    osnet = get_shared_extractor()
    minio_client = get_client()
    print("  OSNet loaded.\n")

    async with AsyncSessionLocal() as db:
        r = await db.execute(text("""
            SELECT id::text, person_identity_id::text, crop_path, camera_id::text
            FROM person_embeddings
            WHERE embedding IS NOT NULL AND crop_path IS NOT NULL
            ORDER BY id
        """))
        rows = r.fetchall()

    total = len(rows)
    print(f"  Total body embeddings to recompute: {total}")
    if dry_run:
        print("  [DRY RUN — no updates will be written]\n")

    updated = 0
    skipped_missing = 0
    skipped_failed = 0

    async with AsyncSessionLocal() as db:
        for i, (emb_id, pid, crop_path, cam_id) in enumerate(rows):
            img = _download_crop(minio_client, crop_path)
            if img is None:
                skipped_missing += 1
                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{total}] updated={updated} missing={skipped_missing} failed={skipped_failed}")
                continue

            new_emb = osnet.extract(img)
            if new_emb is None:
                skipped_failed += 1
                continue

            if not dry_run:
                await db.execute(text(
                    "UPDATE person_embeddings SET embedding = :emb WHERE id::text = :eid"
                ), {"emb": str(new_emb.tolist()), "eid": emb_id})

            updated += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{total}] updated={updated} missing={skipped_missing} failed={skipped_failed}")
                if not dry_run:
                    await db.commit()

        if not dry_run:
            await db.commit()

    print(f"\n  Done: updated={updated}  missing_crop={skipped_missing}  failed={skipped_failed}")

    if not dry_run and updated > 0:
        print("\n  Rebuilding IVFFlat index (CONTEXT.md issue #9)...")
        from app.core.db.session import sync_engine
        with sync_engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(text("REINDEX INDEX idx_person_embeddings_embedding"))
            conn.execute(text("VACUUM ANALYZE person_embeddings"))
        print("  IVFFlat index rebuilt.\n")
    else:
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recompute all body embeddings from MinIO crops with fixed OSNet weights.")
    parser.add_argument("--dry-run", action="store_true", help="Don't write updates, just report stats.")
    args = parser.parse_args()
    asyncio.run(recompute(args.dry_run))
