#!/usr/bin/env python3
"""
fix_demographics_oneshot.py — backfill age (all) + fix known wrong genders.

Age: InsightFace buffalo_l genderage median over stored face crops.
Gender: only the confirmed F→M false-positive IDs → set to F.

Dry-run by default. Pass --apply to write.

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/fix_demographics_oneshot.py
    PYTHONPATH=/gmr/gmr venv/bin/python danger/fix_demographics_oneshot.py --apply
"""
from __future__ import annotations

import argparse
import asyncio
from typing import List, Optional

import cv2
import numpy as np
from sqlalchemy import text

from app.core.db.session import AsyncSessionLocal
from app.modules.reid.insightface_analyzer import get_shared_analyzer
from app.modules.storage.minio_client import BUCKET_PREFIX, get_client

# User-confirmed females previously stored as male
KNOWN_WRONG_FP_MALE = {
    "3656da5e-cb9a-4215-b898-b44f6db2d59a",
    "3fc4be2c-1996-4e73-838a-ebccf466292f",
    "880d395a-c8af-4f4e-9205-571ce0c0a268",
    "d9eb93f5-92c1-465e-bcf6-9a5fdc44d778",
    "b8565d0b-6021-41ac-87e6-d05da3e0a849",
    "fd027ea1-c52e-44df-83fa-bf86b962ae26",
    "6fc05b89-e359-4176-a986-4369b405eaa0",
    "c08bf35e-2876-4517-bcfc-a3929f30e7ad",
}


def minio_key(path: str) -> str:
    if path.startswith(f"{BUCKET_PREFIX}/"):
        return path[len(BUCKET_PREFIX) + 1 :]
    return path


def load_bgr(client, path: str) -> Optional[np.ndarray]:
    try:
        r = client.get_object(BUCKET_PREFIX, minio_key(path))
        d = r.read()
        r.close()
        r.release_conn()
    except Exception:
        return None
    return cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR)


def age_to_group(age: Optional[int]) -> Optional[str]:
    if age is None:
        return None
    if age < 12:
        return "child"
    if age < 25:
        return "young_adult"
    if age < 60:
        return "adult"
    return "senior"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Write changes (default dry-run)")
    ap.add_argument("--max-crops", type=int, default=6)
    args = ap.parse_args()
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"fix_demographics_oneshot — {mode}")

    analyzer = get_shared_analyzer()
    if analyzer.app is None:
        print("FATAL: InsightFace failed to load")
        return
    client = get_client()

    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                text(
                    """
                    SELECT pi.id::text,
                           pi.gender,
                           pi.estimated_age,
                           pi.age_group,
                           (SELECT array_agg(path ORDER BY s DESC NULLS LAST)
                            FROM (
                              SELECT face_crop_path AS path, face_score AS s
                              FROM person_face_embeddings
                              WHERE person_identity_id = pi.id
                                AND face_crop_path IS NOT NULL
                              ORDER BY face_score DESC NULLS LAST
                              LIMIT :lim
                            ) x) AS faces
                    FROM person_identities pi
                    ORDER BY pi.created_at NULLS LAST
                    """
                ),
                {"lim": args.max_crops},
            )
        ).fetchall()

        age_updates = 0
        age_skip = 0
        gender_updates = 0
        changes: List[str] = []

        for i, (pid, db_gender, db_age, db_group, faces) in enumerate(rows, 1):
            faces = list(faces or [])
            ages: List[int] = []
            for p in faces:
                img = load_bgr(client, p)
                if img is None:
                    continue
                a = analyzer.estimate_age_from_crop(img)
                if a is not None:
                    ages.append(int(a))

            new_age = int(round(float(np.median(ages)))) if ages else None
            new_group = age_to_group(new_age)

            new_gender = db_gender
            if pid in KNOWN_WRONG_FP_MALE:
                new_gender = "F"

            age_changed = new_age is not None and new_age != db_age
            group_changed = new_group is not None and new_group != db_group
            gender_changed = (
                pid in KNOWN_WRONG_FP_MALE
                and (db_gender or "").upper()[:1] != "F"
            )

            if new_age is None:
                age_skip += 1

            if not (age_changed or group_changed or gender_changed):
                if i % 25 == 0:
                    print(f"  … {i}/{len(rows)}")
                continue

            parts = [f"{pid[:12]}"]
            if age_changed or group_changed:
                parts.append(f"age {db_age}/{db_group} → {new_age}/{new_group} (n={len(ages)})")
                age_updates += 1
            if gender_changed:
                parts.append(f"gender {db_gender} → F")
                gender_updates += 1
            line = " | ".join(parts)
            changes.append(line)
            print(f"  {line}")

            if args.apply:
                await db.execute(
                    text(
                        """
                        UPDATE person_identities
                        SET estimated_age = COALESCE(:age, estimated_age),
                            age_group = COALESCE(:grp, age_group),
                            gender = CASE WHEN :fix_g THEN 'F' ELSE gender END,
                            updated_at = NOW()
                        WHERE id::text = :pid
                        """
                    ),
                    {
                        "pid": pid,
                        "age": new_age,
                        "grp": new_group,
                        "fix_g": gender_changed,
                    },
                )
                # Keep recent sessions loosely in sync for those we touch
                if gender_changed or age_changed or group_changed:
                    await db.execute(
                        text(
                            """
                            UPDATE track_sessions
                            SET gender = CASE WHEN :fix_g THEN 'F' ELSE gender END,
                                age_group = COALESCE(:grp, age_group),
                                updated_at = NOW()
                            WHERE person_identity_id::text = :pid
                            """
                        ),
                        {
                            "pid": pid,
                            "grp": new_group,
                            "fix_g": gender_changed,
                        },
                    )

            if i % 25 == 0:
                print(f"  … {i}/{len(rows)}")

        if args.apply:
            await db.commit()
            print("Committed.")
        else:
            print("Dry-run — no writes.")

    print(
        f"\nSummary: persons={len(rows)} age_settable_changes={age_updates} "
        f"age_no_det={age_skip} gender_fixes={gender_updates} change_rows={len(changes)}"
    )
    print("Known gender fix list size:", len(KNOWN_WRONG_FP_MALE))


if __name__ == "__main__":
    asyncio.run(main())
