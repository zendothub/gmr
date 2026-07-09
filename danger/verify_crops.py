#!/usr/bin/env python3
"""
verify_crops.py — Download crop images from MinIO and re-compute embeddings.

For the given person IDs, this script:
  1. Downloads each track's best_crop_path from MinIO
  2. Re-runs OSNet body ReID on each crop
  3. Re-runs InsightFace on each crop for face embedding
  4. Compares recomputed embeddings against stored person_embeddings/person_face_embeddings
  5. Identifies tracks whose crops produce embeddings that DON'T match the person

Usage:
    PYTHONPATH=/gmr/gmr .venv/bin/python danger/verify_crops.py \
        b7b7da22-593c-4153-908b-7fb41e284de3 \
        104ef438-5129-4853-913a-a069366f91c2 \
        86be736d-c5a3-4370-b934-a029519eb872 \
        90cd4658-0894-4177-9a59-6d9e0b9dd9d6
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
from app.modules.reid.osnet_extractor import get_shared_extractor
from app.modules.reid.insightface_analyzer import get_shared_analyzer


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
    arr = np.asarray(raw, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr


def _download_crop(minio_client, crop_path: str):
    """Download a crop image from MinIO, return as numpy BGR array."""
    if not crop_path:
        return None
    try:
        key = crop_path
        if key.startswith(f"{BUCKET_PREFIX}/"):
            key = key[len(BUCKET_PREFIX) + 1:]
        if "/" in key:
            key = key.split("/", 1)[1] if not key.startswith("crops/") else key
        resp = minio_client.get_object(BUCKET_PREFIX, key)
        data = resp.read()
        resp.close()
        resp.release_conn()
        import cv2
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"      ⚠ Failed to download {crop_path[:60]}: {e}")
        return None


async def verify(person_ids: list[str]):
    settings = get_settings()
    ids = list(set(person_ids))

    print(f"\n{'='*100}")
    print(f"  CROP VERIFICATION  ({len(ids)} persons)")
    print(f"{'='*100}")

    print("\n  Loading models...")
    osnet = get_shared_extractor()
    analyzer = get_shared_analyzer()
    minio_client = get_client()
    print("  OSNet + InsightFace loaded.\n")

    async with AsyncSessionLocal() as db:
        for pid in ids:
            # Get person summary
            r = await db.execute(text("""
                SELECT pi.gender, pi.estimated_age, pi.age_group,
                       (SELECT COUNT(*) FROM person_embeddings WHERE person_identity_id = pi.id) AS body_count,
                       (SELECT COUNT(*) FROM person_face_embeddings WHERE person_identity_id = pi.id) AS face_count
                FROM person_identities pi WHERE pi.id::text = :pid
            """), {"pid": pid})
            person = r.fetchone()
            if not person:
                print(f"  {pid[:36]} → NOT FOUND")
                continue

            gender = person[0] or "-"
            est_age = person[1] or "-"

            # Get stored body and face embeddings for this person
            r = await db.execute(text("""
                SELECT embedding FROM person_embeddings
                WHERE person_identity_id::text = :pid AND embedding IS NOT NULL
                ORDER BY crop_quality DESC
            """), {"pid": pid})
            stored_body_embs = [_parse_embedding(row[0]) for row in r.fetchall()]

            r = await db.execute(text("""
                SELECT embedding, face_score FROM person_face_embeddings
                WHERE person_identity_id::text = :pid AND embedding IS NOT NULL
                ORDER BY face_score DESC
            """), {"pid": pid})
            stored_face_rows = r.fetchall()
            stored_face_embs = [_parse_embedding(r[0]) for r in stored_face_rows]

            # Get all tracks with crops
            r = await db.execute(text("""
                SELECT ts.id::text, ts.best_crop_path, ts.started_at, ts.last_seen_at,
                       ts.gender, ts.age_group, ts.total_frames, ts.avg_confidence
                FROM track_sessions ts
                WHERE ts.person_identity_id::text = :pid
                  AND ts.best_crop_path IS NOT NULL
                  AND ts.total_frames >= 5
                ORDER BY ts.started_at
            """), {"pid": pid})
            tracks = r.fetchall()

            print(f"\n{'─'*100}")
            print(f"  Person: {pid}")
            print(f"  Stored: {len(stored_body_embs)} body emb, {len(stored_face_embs)} face emb, {gender=}, age={est_age}")
            print(f"  Tracks with crops to verify: {len(tracks)}")

            if not tracks:
                continue

            body_issues = 0
            face_issues = 0
            no_face = 0

            for t in tracks:
                tid = str(t[0])[:12]
                crop_path = t[1]
                started = str(t[3])[:19] if t[3] else "-"
                track_gender = t[4] or "-"
                track_age = t[5] or "-"
                frames = t[6] or 0
                conf = f"{t[7]:.2f}" if t[7] else "-"

                img = _download_crop(minio_client, crop_path)
                if img is None:
                    continue

                # Re-compute body embedding
                body_emb = None
                try:
                    body_emb = osnet.extract(img)
                except Exception as e:
                    print(f"    trk={tid}: OSNet failed: {e}")
                    continue

                # Re-compute face embedding
                face_emb = None
                try:
                    face_info = analyzer.analyze(img)
                    if face_info and face_info.get("embedding") is not None:
                        face_emb = np.array(face_info["embedding"], dtype=np.float32)
                except Exception as e:
                    pass

                # Compare body embedding against stored body embeddings
                body_sims = []
                if stored_body_embs:
                    body_sims = [cos_sim(body_emb, s) for s in stored_body_embs]
                    body_best = max(body_sims)
                    body_median = float(np.median(body_sims))
                else:
                    body_best = -1
                    body_median = -1

                # Compare face embedding against stored face embeddings
                face_sims = []
                face_best = -1
                if face_emb is not None and stored_face_embs:
                    face_sims = [cos_sim(face_emb, s) for s in stored_face_embs]
                    face_best = max(face_sims)

                # Flag issues
                flags = []
                if body_best < 0.30:
                    flags.append(f"BODY MISMATCH (best={body_best:.3f})")
                    body_issues += 1
                elif body_best < 0.50:
                    flags.append(f"BODY LOW (best={body_best:.3f})")

                if face_emb is not None and stored_face_embs:
                    if face_best < 0.25:
                        flags.append(f"FACE MISMATCH (best={face_best:.3f})")
                        face_issues += 1
                    elif face_best < 0.40:
                        flags.append(f"FACE LOW (best={face_best:.3f})")
                elif face_emb is None:
                    no_face += 1

                flag_str = " | ".join(flags) if flags else "✓"
                print(f"    trk={tid} frames={frames} conf={conf} g={track_gender} age={track_age} started={started}")
                print(f"          body→person: best={body_best:.4f} median={body_median:.4f} | face→person: best={face_best:.4f} | {flag_str}")
                if body_sims and len(body_sims) <= 10:
                    print(f"          body_sims: {[f'{s:.3f}' for s in body_sims]}")
                if face_sims and len(face_sims) <= 5:
                    print(f"          face_sims: {[f'{s:.3f}' for s in face_sims]}")

            print(f"\n  Summary: {body_issues} body mismatches, {face_issues} face mismatches, {no_face} tracks without detectable face")

    # ── Cross-person body comparison using re-computed track crops ──────────
    print(f"\n{'─'*100}")
    print("  CROSS-PERSON TRACK CROP COMPARISON")
    print(f"  (Re-computing body embeddings from each person's track crops,")
    print(f"  comparing across persons to see if OSNet can separate them)")

    # Collect first 3 track crops per person and compute body embeddings
    person_track_embs = {}
    async with AsyncSessionLocal() as db:
        for pid in ids:
            r = await db.execute(text("""
                SELECT ts.id::text, ts.best_crop_path
                FROM track_sessions ts
                WHERE ts.person_identity_id::text = :pid
                  AND ts.best_crop_path IS NOT NULL
                  AND ts.total_frames >= 5
                ORDER BY ts.started_at
                LIMIT 3
            """), {"pid": pid})
            tracks = r.fetchall()
            embs = []
            for t in tracks:
                img = _download_crop(minio_client, t[1])
                if img is not None:
                    try:
                        emb = osnet.extract(img)
                        embs.append((str(t[0])[:12], emb))
                    except Exception:
                        pass
            if embs:
                person_track_embs[pid] = embs

    if len(person_track_embs) >= 2:
        pids_sorted = sorted(person_track_embs.keys())
        print(f"\n  {'Pair A':<40} {'Pair B':<40} {'BestCross':>12} {'AvgCross':>10} {'SelfA_Avg':>10} {'SelfB_Avg':>10} {'Separation':>10}")
        print("  " + "-" * 98)
        for i in range(len(pids_sorted)):
            pa = pids_sorted[i]
            a_embs = person_track_embs[pa]
            a_self = [cos_sim(a_embs[j][1], a_embs[k][1]) for j in range(len(a_embs)) for k in range(j+1, len(a_embs))]
            a_self_avg = float(np.mean(a_self)) if a_self else 0

            for j in range(i + 1, len(pids_sorted)):
                pb = pids_sorted[j]
                b_embs = person_track_embs[pb]
                b_self = [cos_sim(b_embs[j2][1], b_embs[k2][1]) for j2 in range(len(b_embs)) for k2 in range(j2+1, len(b_embs))]
                b_self_avg = float(np.mean(b_self)) if b_self else 0

                cross = [cos_sim(ae, be) for _, ae in a_embs for _, be in b_embs]
                cross_best = max(cross) if cross else 0
                cross_avg = float(np.mean(cross)) if cross else 0
                separation = cross_best - max(a_self_avg, b_self_avg, 0.5)

                flag = ""
                if cross_best > 0.60:
                    flag = " ⚠ HIGH CROSS (likely same person)"
                if separation > 0:
                    flag += " ⚠ CROSS > SELF (OSNet confused)"
                print(f"  {pa[:38]:<40} {pb[:38]:<40} {cross_best:>10.4f}   {cross_avg:>8.4f}   {a_self_avg:>8.4f}   {b_self_avg:>8.4f}   {separation:>8.4f}{flag}")

    print(f"\n{'='*100}")
    print(f"  VERIFICATION COMPLETE")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-compute embeddings from crop images to verify identity")
    parser.add_argument("ids", nargs="+", help="Person identity UUIDs to verify")
    args = parser.parse_args()
    asyncio.run(verify(args.ids))
