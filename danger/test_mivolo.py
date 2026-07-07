#!/usr/bin/env python3
"""
test_mivolo.py — test MiVOLO gender predictions against existing person records.

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/test_mivolo.py [person_id ...]

Runs MiVOLO on every face crop stored for the given PersonIdentity and
prints the MiVOLO prediction vs the DB-stored gender.  Use this to validate
the model before relying on it in production.
"""

import asyncio
import sys

from sqlalchemy import text
from app.core.db.session import AsyncSessionLocal
from app.modules.reid.mivolo_analyzer import get_shared_mivolo
from app.modules.storage.minio_client import get_client, BUCKET_PREFIX


async def test_person(mivolo, db, person_id: str) -> None:
    print(f"\n{'='*70}")
    result = await db.execute(text(
        "SELECT id::text, gender, best_face_score"
        " FROM person_identities WHERE id::text = :pid"
    ), {"pid": person_id})
    pi = result.fetchone()
    if not pi:
        print(f"  Person NOT FOUND: {person_id}")
        return
    print(f"  DB gender: {pi[1]}   DB face_score: {pi[2]}")

    faces_result = await db.execute(text(
        "SELECT face_crop_path, face_score, captured_at"
        " FROM person_face_embeddings"
        " WHERE person_identity_id::text = :pid AND face_crop_path IS NOT NULL"
        " ORDER BY captured_at"
    ), {"pid": person_id})
    faces = faces_result.fetchall()

    if not faces:
        print("  No face embeddings with crop paths found.")
        return

    print(f"  Testing {len(faces)} face crop(s) …\n")
    client = get_client()
    votes: dict[str, int] = {"M": 0, "F": 0}

    for i, (path, db_score, cap_time) in enumerate(faces):
        if path.startswith(f"{BUCKET_PREFIX}/"):
            key = path[len(BUCKET_PREFIX) + 1:]
        elif "/" in path:
            key = path
        else:
            key = path

        try:
            resp = client.get_object(BUCKET_PREFIX, key)
            data = resp.read()
            resp.close()
            resp.release_conn()
        except Exception as e:
            print(f"  [{i+1}] 404: {key[:70]} — {e}")
            continue

        import cv2
        import numpy as np
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"  [{i+1}] Could not decode: {key[:70]}")
            continue

        pred = mivolo.analyze(frame)
        if pred is None:
            print(f"  [{i+1}] MiVOLO failed on crop {key[:60]}")
            continue

        votes[pred["gender"]] += 1
        marker = " ✓" if pred["gender"] == pi[1] else " ✗ MISMATCH" if pi[1] is not None else ""
        print(
            f"  [{i+1}] MiVOLO: gender={pred['gender']}  age={pred['age']}"
            f"  prob={pred['gender_prob']:.3f}  DB_score={db_score:.3f}{marker}"
        )

    majority = "M" if votes.get("M", 0) >= votes.get("F", 0) else "F"
    print(f"\n  MiVOLO majority: {majority}  (M={votes.get('M',0)} F={votes.get('F',0)})")
    if pi[1] and majority != pi[1]:
        print(f"  ⚠️  MiVOLO disagrees with stored gender ({pi[1]})!")


async def main(ids: list[str]):
    mivolo = get_shared_mivolo()
    if mivolo.model is None:
        print("MiVOLO failed to load.")
        return

    async with AsyncSessionLocal() as db:
        for pid in ids:
            await test_person(mivolo, db, pid)


if __name__ == "__main__":
    ids = sys.argv[1:] if len(sys.argv) > 1 else [
        "e4c812b1-53fc-4fff-b411-b7490a55fffb",
        "c12a1f55-20c9-4d1e-907b-d5ab325cd0b1",
        "53dc068d-832d-47cf-b6ee-d75d1c216a25",
    ]
    asyncio.run(main(ids))
