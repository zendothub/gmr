#!/usr/bin/env python3
"""
fix_gender_siglip2.py — cross-check all persons and update gender with SigLIP2.

SigLIP2 achieved 100% gender accuracy on clean retail CCTV face crops
(vs ~11% for MiVOLO and 0% for DeepFace).  This script re-analyses every
stored face crop and corrects PersonIdentity.gender where SigLIP2 disagrees
with the stored value.

Usage:
    # Dry-run: print mismatches only (does NOT modify DB)
    PYTHONPATH=/gmr/gmr venv/bin/python danger/fix_gender_siglip2.py

    # Apply fix: actually update the DB
    PYTHONPATH=/gmr/gmr venv/bin/python danger/fix_gender_siglip2.py --apply
"""

import asyncio, argparse, sys, cv2, numpy as np
from sqlalchemy import text
from app.core.db.session import AsyncSessionLocal
from app.modules.storage.minio_client import get_client, BUCKET_PREFIX


async def fix_all(apply_fix: bool) -> int:
    from app.modules.reid.siglip2_analyzer import get_shared_siglip2
    s2 = get_shared_siglip2()
    if s2.model is None:
        print("FATAL: SigLIP2 failed to load.")
        return 1

    client = get_client()

    async with AsyncSessionLocal() as db:
        result = await db.execute(text("""
            SELECT pi.id::text, pi.gender,
                   (SELECT COUNT(*) FROM person_face_embeddings
                    WHERE person_identity_id = pi.id AND face_crop_path IS NOT NULL) AS cc
            FROM person_identities pi
            WHERE EXISTS (SELECT 1 FROM person_face_embeddings
                          WHERE person_identity_id = pi.id AND face_crop_path IS NOT NULL)
            ORDER BY pi.created_at
        """))
        persons = result.fetchall()

        mode = "APPLY" if apply_fix else "DRY RUN"
        print(f"\n{'='*60}")
        print(f"  SigLIP2 gender fix — {mode}")
        print(f"  Persons with face crops: {len(persons)}")
        print(f"{'='*60}\n")

        fixed = skipped = errors = 0

        for pid, stored_gender, crop_count in persons:
            crop_result = await db.execute(text("""
                SELECT face_crop_path FROM person_face_embeddings
                WHERE person_identity_id::text = :pid AND face_crop_path IS NOT NULL
                ORDER BY face_score DESC
            """), {"pid": pid})
            crops = [r[0] for r in crop_result.fetchall()]

            votes = {"M": 0, "F": 0}

            for path in crops:
                key = path.split("/", 1)[1] if "/" in path else path
                if key.startswith(f"{BUCKET_PREFIX}/"):
                    key = key[len(BUCKET_PREFIX) + 1:]
                try:
                    resp = client.get_object(BUCKET_PREFIX, key)
                    data = resp.read(); resp.close(); resp.release_conn()
                except Exception:
                    continue

                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                pred = s2.analyze(frame)
                if pred and pred.get("gender") in ("M", "F"):
                    votes[pred["gender"]] += 1

            m, f = votes["M"], votes["F"]
            if m == 0 and f == 0:
                skipped += 1
                continue

            majority = "F" if f > m else "M"

            if stored_gender != majority:
                status = "FIXED" if apply_fix else "MISMATCH"
                print(f"  [{status}] {pid[:12]}  {stored_gender} → {majority}  (M={m} F={f})")
                if apply_fix:
                    await db.execute(text(
                        "UPDATE person_identities SET gender = :g WHERE id::text = :p"
                    ), {"g": majority, "p": pid})
                    fixed += 1
                errors += 1 if status == "MISMATCH" else 0
            else:
                skipped += 1

        if apply_fix and fixed > 0:
            await db.commit()
            print(f"\n  Committed {fixed} gender correction(s).")
        elif apply_fix:
            print(f"\n  No corrections needed.")

        print(f"\n{'='*60}")
        print(f"  Summary: {fixed} fixed, {skipped} skipped")
        if not apply_fix:
            print(f"  Run with --apply to actually update the database.")
        print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix gender using SigLIP2.")
    parser.add_argument("--apply", action="store_true", help="Update DB (default: dry-run)")
    args = parser.parse_args()
    sys.exit(asyncio.run(fix_all(args.apply)))
