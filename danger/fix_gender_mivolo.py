#!/usr/bin/env python3
"""
fix_gender_mivolo.py — one-shot correction of misgendered person identities.

Re-runs MiVOLO on every stored face crop for each PersonIdentity and updates
the DB ``gender`` field if the majority vote disagrees with the stored value.

Usage:
    # Dry-run: print mismatches only (does NOT modify DB)
    PYTHONPATH=/gmr/gmr venv/bin/python danger/fix_gender_mivolo.py

    # Apply fix: actually update the DB
    PYTHONPATH=/gmr/gmr venv/bin/python danger/fix_gender_mivolo.py --apply
"""

import asyncio
import sys
import argparse

from sqlalchemy import text
from app.core.db.session import AsyncSessionLocal
from app.modules.reid.mivolo_analyzer import get_shared_mivolo
from app.modules.storage.minio_client import get_client, BUCKET_PREFIX


async def fix_all(apply_fix: bool) -> int:
    mivolo = get_shared_mivolo()
    if mivolo.model is None:
        print("FATAL: MiVOLO failed to load.")
        return 1

    client = get_client()

    async with AsyncSessionLocal() as db:
        # Fetch all persons with face embeddings + crop paths
        result = await db.execute(text("""
            SELECT pi.id::text, pi.gender, pi.best_face_score,
                   (SELECT COUNT(*) FROM person_face_embeddings WHERE person_identity_id = pi.id
                    AND face_crop_path IS NOT NULL) AS crop_count
            FROM person_identities pi
            WHERE EXISTS (SELECT 1 FROM person_face_embeddings
                          WHERE person_identity_id = pi.id AND face_crop_path IS NOT NULL)
            ORDER BY pi.created_at
        """))
        persons = result.fetchall()

        print(f"\n{'='*70}")
        mode = "APPLY" if apply_fix else "DRY RUN"
        print(f"  MiVOLO gender fix — {mode}")
        print(f"  Persons with face crops: {len(persons)}")
        print(f"{'='*70}\n")

        fixed = 0
        skipped = 0
        errors = 0

        for pid, stored_gender, _, crop_count in persons:
            # Get all face crop paths
            crop_result = await db.execute(text("""
                SELECT face_crop_path FROM person_face_embeddings
                WHERE person_identity_id::text = :pid AND face_crop_path IS NOT NULL
            """), {"pid": pid})
            crops = [r[0] for r in crop_result.fetchall()]

            votes: dict[str, int] = {"M": 0, "F": 0}

            for path in crops:
                key = path.split("/", 1)[1] if "/" in path and "/" in path else path
                if key.startswith(f"{BUCKET_PREFIX}/"):
                    key = key[len(BUCKET_PREFIX) + 1:]

                try:
                    resp = client.get_object(BUCKET_PREFIX, key)
                    data = resp.read()
                    resp.close()
                    resp.release_conn()
                except Exception:
                    continue  # crop file doesn't exist → skip

                import cv2
                import numpy as np
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                pred = mivolo.analyze(frame)
                if pred and pred.get("gender") in ("M", "F"):
                    votes[pred["gender"]] += 1

            m_votes = votes.get("M", 0)
            f_votes = votes.get("F", 0)
            if m_votes == 0 and f_votes == 0:
                skipped += 1
                continue

            majority = "F" if f_votes > m_votes else "M"

            if stored_gender != majority:
                status = "FIXED" if apply_fix else "MISMATCH"
                print(f"  [{status}] {pid[:12]}  stored={stored_gender} → MiVOLO={majority}  "
                      f"(M={m_votes} F={f_votes} from {len(crops)} crops)")
                if apply_fix:
                    await db.execute(text(
                        "UPDATE person_identities SET gender = :gender WHERE id::text = :pid"
                    ), {"gender": majority, "pid": pid})
                    fixed += 1
            else:
                skipped += 1

        if apply_fix and fixed > 0:
            await db.commit()
            print(f"\n  Committed {fixed} gender correction(s).")
        elif apply_fix:
            print(f"\n  No corrections needed.")

        print(f"\n{'='*70}")
        print(f"  Summary: {fixed} fixed, {skipped} skipped, {errors} errors")
        print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix misgendered person identities using MiVOLO.")
    parser.add_argument("--apply", action="store_true", help="Actually update the database (default: dry-run)")
    args = parser.parse_args()
    sys.exit(asyncio.run(fix_all(args.apply)))
