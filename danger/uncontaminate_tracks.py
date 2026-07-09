#!/usr/bin/env python3
"""
uncontaminate_tracks.py — Disassociate tracks whose face contradicts the assigned person.

For each track with a person_identity_id and best_crop_path:
  1. Download the crop from MinIO
  2. Run InsightFace + OSNet on it
  3. Compute face/body embeddings
  4. Check if the track's face contradicts the person's stored face cluster
  5. If contradiction → set person_identity_id to NULL (disassociate)
  6. Optionally also remove contaminated body/face embeddings

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/uncontaminate_tracks.py [--apply] [--ids PID1 PID2 ...]
"""

import asyncio
import sys
import argparse
import io
import numpy as np
from sqlalchemy import text
from app.core.db.session import AsyncSessionLocal
from app.config import get_settings
from app.modules.storage.minio_client import get_client, BUCKET_PREFIX


def cos_sim(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _parse_embedding(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        return np.array(eval(raw), dtype=np.float32)
    return np.array(raw, dtype=np.float32)


def _download_crop(minio_client, crop_path: str):
    if not crop_path:
        return None
    try:
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
        return None


async def uncontaminate(person_ids: list[str] | None, apply_fix: bool):
    settings = get_settings()

    print(f"\n{'='*100}")
    print(f"  TRACK UNCONTAMINATION  {'— APPLY' if apply_fix else '— DRY RUN'}")
    print(f"{'='*100}")

    print("  Loading models...")
    from app.modules.reid.osnet_extractor import get_shared_extractor
    from app.modules.reid.insightface_analyzer import get_shared_analyzer
    osnet = get_shared_extractor()
    analyzer = get_shared_analyzer()
    minio_client = get_client()
    print("  OSNet + InsightFace loaded.\n")

    async with AsyncSessionLocal() as db:
        # Get all persons with tracks that have crops
        if person_ids:
            id_filter = "AND ts.person_identity_id::text = ANY(:ids)"
            params = {"ids": list(person_ids)}
        else:
            id_filter = ""
            params = {}

        r = await db.execute(text(f"""
            SELECT DISTINCT ts.person_identity_id::text
            FROM track_sessions ts
            WHERE ts.person_identity_id IS NOT NULL
              AND ts.best_crop_path IS NOT NULL
              {id_filter}
        """), params)
        all_pids = [row[0] for row in r.fetchall()]

        if not all_pids:
            print("  No tracks with crops found.")
            return

        print(f"  Persons to check: {len(all_pids)}\n")

        total_checked = 0
        total_disassociated = 0
        total_face_removed = 0
        total_body_removed = 0

        for pid in all_pids:
            # Get stored face embeddings for this person
            r = await db.execute(text("""
                SELECT embedding, face_score FROM person_face_embeddings
                WHERE person_identity_id::text = :pid AND embedding IS NOT NULL
                ORDER BY face_score DESC
            """), {"pid": pid})
            face_rows = r.fetchall()
            stored_faces = [_parse_embedding(r[0]) for r in face_rows]

            # Get stored body embeddings for this person
            r = await db.execute(text("""
                SELECT embedding, crop_quality FROM person_embeddings
                WHERE person_identity_id::text = :pid AND embedding IS NOT NULL
                ORDER BY crop_quality DESC
            """), {"pid": pid})
            body_rows = r.fetchall()
            stored_bodies = [_parse_embedding(r[0]) for r in body_rows]

            # Get tracks with crops
            r = await db.execute(text("""
                SELECT ts.id::text, ts.best_crop_path, ts.started_at
                FROM track_sessions ts
                WHERE ts.person_identity_id::text = :pid
                  AND ts.best_crop_path IS NOT NULL
                  AND ts.total_frames >= 5
                ORDER BY ts.started_at
            """), {"pid": pid})
            tracks = r.fetchall()

            if not tracks:
                continue

            face_contradictions = 0
            body_mismatches = 0

            for tid, crop_path, started in tracks:
                img = _download_crop(minio_client, crop_path)
                if img is None:
                    continue

                total_checked += 1
                tid_short = tid[:12]

                # Compute face embedding from crop
                face_emb = None
                try:
                    face_info = analyzer.analyze(img)
                    if face_info and face_info.get("embedding") is not None:
                        face_emb = np.array(face_info["embedding"], dtype=np.float32)
                except Exception:
                    pass

                # Compute body embedding from crop
                body_emb = None
                try:
                    body_emb = osnet.extract(img)
                except Exception:
                    pass

                # Check face contradiction
                face_contradicts = False
                if face_emb is not None and len(stored_faces) >= 2:
                    max_face_sim = max(cos_sim(face_emb, sf) for sf in stored_faces)
                    if max_face_sim < settings.FACE_CONTRADICTION_THRESHOLD:  # 0.25
                        face_contradicts = True
                        face_contradictions += 1

                # Check body mismatch
                body_mismatch = False
                if body_emb is not None and len(stored_bodies) >= 3:
                    max_body_sim = max(cos_sim(body_emb, sb) for sb in stored_bodies)
                    if max_body_sim < settings.BODY_CONTAMINATION_THRESHOLD:  # 0.40
                        body_mismatch = True
                        body_mismatches += 1

                if face_contradicts or body_mismatch:
                    flags = []
                    if face_contradicts:
                        flags.append(f"FACE-CONTRADICT")
                    if body_mismatch:
                        flags.append(f"BODY-MISMATCH")

                    face_sims_str = ""
                    if face_emb is not None and stored_faces:
                        face_sims_str = f" face_sims={[f'{cos_sim(face_emb, sf):.3f}' for sf in stored_faces[:3]]}"
                    body_sims_str = ""
                    if body_emb is not None and stored_bodies:
                        body_sims_str = f" body_sims={[f'{cos_sim(body_emb, sb):.3f}' for sb in stored_bodies[:3]]}"

                    print(f"  {' | '.join(flags):30s} trk={tid_short} person={pid[:12]}{face_sims_str}{body_sims_str}")

                    if apply_fix:
                        # Disassociate track from person
                        await db.execute(text(
                            "UPDATE track_sessions SET person_identity_id = NULL WHERE id::text = :tid"
                        ), {"tid": tid})
                        total_disassociated += 1

            # After checking tracks, also clean contaminated embeddings
            if apply_fix and face_contradictions > 0:
                # Remove face embeddings that don't match the cluster
                if len(stored_faces) >= 2:
                    from app.modules.jobs.tasks import _clean_contaminated_face_embeddings
                    removed = await _clean_contaminated_face_embeddings(db, settings)
                    total_face_removed += removed

            if apply_fix and body_mismatches > 0:
                if len(stored_bodies) >= 3:
                    from app.modules.jobs.tasks import _clean_contaminated_body_embeddings
                    removed = await _clean_contaminated_body_embeddings(db, settings)
                    total_body_removed += removed

            if face_contradictions > 0 or body_mismatches > 0:
                print(f"  Person {pid[:12]}: {face_contradictions} face contradictions, {body_mismatches} body mismatches among {len(tracks)} tracks\n")

        if apply_fix and (total_disassociated > 0 or total_face_removed > 0 or total_body_removed > 0):
            await db.commit()
            print(f"\n  Applied: {total_disassociated} tracks disassociated, {total_face_removed} faces removed, {total_body_removed} bodies removed")
        elif not apply_fix:
            print(f"\n  Dry run: {total_disassociated} tracks would be disassociated across {len(all_pids)} persons")
            print("  Run with --apply to fix.")

    print(f"{'='*100}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disassociate contaminated tracks from person identities")
    parser.add_argument("--apply", action="store_true", help="Apply fixes (default: dry run)")
    parser.add_argument("--ids", nargs="*", default=None, help="Person identity UUIDs to check (default: all)")
    args = parser.parse_args()
    asyncio.run(uncontaminate(args.ids, args.apply))
