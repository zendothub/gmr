#!/usr/bin/env python3
"""
reextract_or_delete_faceless.py — Fix persons left faceless by contamination cleanup.

Runs as a SEPARATE PROCESS (via systemd timer, not inside the FastAPI event loop)
to avoid blocking the API and contending for GPU memory with live camera workers.

For each person with 0 face embeddings:
  1. Download their track crops (best_crop_path) from MinIO
  2. Run InsightFace on each crop to extract a face embedding
  3. If a valid face is found → store it (normalized) in person_face_embeddings
  4. If NO face is found in ANY crop → DELETE the person entirely:
     - Orphan all tracks (set person_identity_id = NULL)
     - Orphan events and billing_interactions
     - DELETE the person (cascades to person_embeddings + person_face_embeddings)
     - MinIO crops are left for the periodic sweep to clean up

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/reextract_or_delete_faceless.py
"""

import asyncio
import sys
import numpy as np
from sqlalchemy import text
from loguru import logger
from app.core.db.session import AsyncSessionLocal
from app.config import get_settings


def _download_crop(minio_client, crop_path: str):
    if not crop_path:
        return None
    try:
        from app.modules.storage.minio_client import BUCKET_PREFIX
        key = crop_path
        if key.startswith(f"{BUCKET_PREFIX}/"):
            key = key[len(BUCKET_PREFIX) + 1:]
        resp = minio_client.get_object(BUCKET_PREFIX, key)
        data = resp.read()
        resp.close()
        resp.release_conn()
        import cv2
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.debug(f"Failed to download {crop_path[:50]}: {e}")
        return None


async def run():
    settings = get_settings()
    logger.info("=== Face Re-extraction / Deletion Job ===")

    # ── 1. Find faceless persons with tracks ──────────────────────────────
    async with AsyncSessionLocal() as db:
        r = await db.execute(text("""
            SELECT pi.id::text
            FROM person_identities pi
            WHERE (SELECT COUNT(*) FROM person_face_embeddings WHERE person_identity_id = pi.id) = 0
              AND EXISTS (
                  SELECT 1 FROM track_sessions ts
                  WHERE ts.person_identity_id = pi.id AND ts.best_crop_path IS NOT NULL
              )
        """))
        faceless_pids = [row[0] for row in r.fetchall()]

    if not faceless_pids:
        logger.info("No faceless persons found. Exiting.")
        return

    logger.info(f"Found {len(faceless_pids)} faceless person(s) to process.")

    # ── 2. Load InsightFace (separate process — own GPU memory) ──────────
    logger.info("Loading InsightFace buffalo_l...")
    from app.modules.reid.insightface_analyzer import get_shared_analyzer
    from app.modules.storage.minio_client import get_client
    analyzer = get_shared_analyzer()
    minio_client = get_client()
    logger.info("InsightFace loaded.")

    reextracted = 0
    deleted = 0

    for pid in faceless_pids:
        async with AsyncSessionLocal() as db:
            # Get up to 5 track crops ordered by duration (longest = most frames = best crop)
            r2 = await db.execute(text("""
                SELECT ts.id::text, ts.best_crop_path, ts.total_frames
                FROM track_sessions ts
                WHERE ts.person_identity_id::text = :pid
                  AND ts.best_crop_path IS NOT NULL
                ORDER BY ts.total_frames DESC
                LIMIT 5
            """), {"pid": pid})
            crops = r2.fetchall()

            if not crops:
                logger.warning(f"Person {pid[:12]}: no crops found despite EXISTS check — skipping")
                continue

            # Try each crop to find a valid face
            face_found = False
            for crop_row in crops:
                tid_short = crop_row[0][:12]
                crop_path = crop_row[1]
                img = _download_crop(minio_client, crop_path)
                if img is None:
                    continue

                try:
                    result = analyzer.analyze(img)
                    if result and result.embedding is not None and result.face_score >= settings.FACE_MIN_DET_SCORE:
                        # Normalize the embedding before storing
                        emb = np.array(result.embedding, dtype=np.float32)
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            emb = emb / norm

                        from app.core.db.models.person import PersonFaceEmbedding
                        face_emb = PersonFaceEmbedding(
                            person_identity_id=pid,
                            embedding=emb.tolist(),
                            face_score=result.face_quality,
                            face_crop_path=None,
                            captured_at=crop_row[0],
                        )
                        db.add(face_emb)
                        await db.commit()
                        face_found = True
                        reextracted += 1
                        logger.info(
                            f"Person {pid[:12]}: re-extracted face from track {tid_short} "
                            f"(det={result.face_score:.2f}, quality={result.face_quality:.2f})"
                        )
                        break
                except Exception as e:
                    logger.debug(f"Face extraction failed for track {tid_short}: {e}")
                    continue

            if not face_found:
                # ── 3. Delete the faceless person ───────────────────────────
                # Take the SAME advisory lock as live decide_identity so we never
                # DELETE a person_id that a camera worker is mid-store attaching to
                # (FK race → session poison storm). See CONTEXT.md issue #26 / P5.
                from app.modules.reid.identity_decision_engine import IDENTITY_ADVISORY_LOCK_KEY
                await db.execute(text(f"SELECT pg_advisory_xact_lock({IDENTITY_ADVISORY_LOCK_KEY})"))

                logger.warning(
                    f"Person {pid[:12]}: no face found in {len(crops)} crop(s) — deleting person"
                )

                # Orphan tracks
                await db.execute(text(
                    "UPDATE track_sessions SET person_identity_id = NULL "
                    "WHERE person_identity_id::text = :pid"
                ), {"pid": pid})

                # Orphan events
                await db.execute(text(
                    "UPDATE events SET person_identity_id = NULL "
                    "WHERE person_identity_id::text = :pid"
                ), {"pid": pid})

                # Orphan billing interactions
                await db.execute(text(
                    "UPDATE billing_interactions SET person_identity_id = NULL "
                    "WHERE person_identity_id::text = :pid"
                ), {"pid": pid})

                # Orphan storage objects
                await db.execute(text(
                    "UPDATE storage_objects SET person_identity_id = NULL "
                    "WHERE person_identity_id::text = :pid"
                ), {"pid": pid})

                # Delete the person (cascades to person_embeddings + person_face_embeddings)
                await db.execute(text(
                    "DELETE FROM person_identities WHERE id::text = :pid"
                ), {"pid": pid})

                await db.commit()
                deleted += 1
                logger.info(f"Person {pid[:12]}: deleted (tracks orphaned, MinIO cleanup deferred)")

    logger.info(
        f"=== Done: {reextracted} re-extracted, {deleted} deleted, "
        f"{len(faceless_pids)} total processed ==="
    )


if __name__ == "__main__":
    asyncio.run(run())
