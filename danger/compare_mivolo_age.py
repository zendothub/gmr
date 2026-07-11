#!/usr/bin/env python3
"""
compare_mivolo_age.py — dry-run FairFace (3-class) vs IMDB (regression) age
on existing MinIO face crops. NO DB writes.

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/compare_mivolo_age.py
    PYTHONPATH=/gmr/gmr venv/bin/python danger/compare_mivolo_age.py --max-persons 30
    PYTHONPATH=/gmr/gmr venv/bin/python danger/compare_mivolo_age.py --per-crop
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sqlalchemy import text

from app.core.db.session import AsyncSessionLocal
from app.modules.reid.mivolo_analyzer import MiVOLOAnalyzer
from app.modules.storage.minio_client import BUCKET_PREFIX, get_client

V2_BINS = [
    ("under_18", 0, 17),
    ("18_24", 18, 24),
    ("25_34", 25, 34),
    ("35_44", 35, 44),
    ("45_60", 45, 60),
    ("60_plus", 61, 999),
]


def age_bin(age: Optional[int]) -> str:
    if age is None:
        return "none"
    for key, lo, hi in V2_BINS:
        if lo <= age <= hi:
            return key
    return "none"


def coarse_group(age: Optional[int]) -> str:
    if age is None:
        return "none"
    if age < 12:
        return "child"
    if age < 25:
        return "young_adult"
    if age < 60:
        return "adult"
    return "senior"


def minio_key(path: str) -> str:
    if path.startswith(f"{BUCKET_PREFIX}/"):
        return path[len(BUCKET_PREFIX) + 1 :]
    return path


def load_bgr(client, path: str) -> Optional[np.ndarray]:
    key = minio_key(path)
    try:
        resp = client.get_object(BUCKET_PREFIX, key)
        data = resp.read()
        resp.close()
        resp.release_conn()
    except Exception:
        return None
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return frame


def print_hist(title: str, ages: List[int]) -> None:
    print(f"\n=== {title} (n={len(ages)}) ===")
    if not ages:
        print("  (empty)")
        return
    arr = np.array(ages, dtype=float)
    print(
        f"  mean={arr.mean():.1f}  median={np.median(arr):.1f}  "
        f"p10={np.percentile(arr,10):.0f}  p90={np.percentile(arr,90):.0f}  "
        f"min={arr.min():.0f}  max={arr.max():.0f}"
    )
    bins = Counter(age_bin(a) for a in ages)
    total = len(ages)
    print("  V2 bins:")
    for key, _, _ in V2_BINS:
        n = bins.get(key, 0)
        pct = 100.0 * n / total
        print(f"    {key:10s}  {n:4d}  ({pct:5.1f}%)")
    groups = Counter(coarse_group(a) for a in ages)
    print("  coarse groups:")
    for g in ("child", "young_adult", "adult", "senior"):
        n = groups.get(g, 0)
        pct = 100.0 * n / total
        print(f"    {g:12s}  {n:4d}  ({pct:5.1f}%)")


async def fetch_person_crops(
    db, max_persons: Optional[int]
) -> List[Tuple[str, Optional[int], Optional[str], List[str]]]:
    """Return list of (person_id, db_age, db_group, crop_paths)."""
    lim = f" LIMIT {int(max_persons)}" if max_persons else ""
    rows = (
        await db.execute(
            text(
                f"""
                SELECT pi.id::text,
                       pi.estimated_age,
                       pi.age_group,
                       COALESCE(
                         (SELECT array_agg(x.path ORDER BY x.ord)
                          FROM (
                            SELECT face_crop_path AS path, 0 AS ord
                            FROM person_identities
                            WHERE id = pi.id AND face_crop_path IS NOT NULL
                            UNION ALL
                            SELECT face_crop_path AS path, 1 AS ord
                            FROM person_face_embeddings
                            WHERE person_identity_id = pi.id
                              AND face_crop_path IS NOT NULL
                          ) x
                         ),
                         ARRAY[]::text[]
                       ) AS paths
                FROM person_identities pi
                WHERE EXISTS (
                    SELECT 1 FROM person_face_embeddings pfe
                    WHERE pfe.person_identity_id = pi.id
                      AND pfe.face_crop_path IS NOT NULL
                )
                   OR pi.face_crop_path IS NOT NULL
                ORDER BY pi.created_at NULLS LAST
                {lim}
                """
            )
        )
    ).fetchall()
    out = []
    for pid, age, group, paths in rows:
        # de-dupe preserve order
        seen = set()
        uniq = []
        for p in paths or []:
            if p and p not in seen:
                seen.add(p)
                uniq.append(p)
        out.append((pid, age, group, uniq))
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-persons", type=int, default=None)
    parser.add_argument(
        "--per-crop",
        action="store_true",
        help="Print every crop prediction (verbose)",
    )
    parser.add_argument(
        "--max-crops-per-person",
        type=int,
        default=5,
        help="Cap crops per person (default 5)",
    )
    args = parser.parse_args()

    fairface_path = "models/mivolo/mivolo_fairface.pth.tar"
    imdb_path = "models/mivolo/mivolo_imbd.pth.tar"

    print("Loading MiVOLO FairFace (3-class) …")
    ff = MiVOLOAnalyzer(model_path=fairface_path)
    print("Loading MiVOLO IMDB (regression) …")
    imdb = MiVOLOAnalyzer(model_path=imdb_path)
    if ff.model is None or imdb.model is None:
        print("FATAL: model load failed")
        return

    print(
        f"  FairFace heads: age={ff.model.num_age} gender={ff.model.num_gender} "
        f"range=[{ff._min_age},{ff._max_age}] avg={ff._avg_age}"
    )
    print(
        f"  IMDB     heads: age={imdb.model.num_age} gender={imdb.model.num_gender} "
        f"range=[{imdb._min_age},{imdb._max_age}] avg={imdb._avg_age}"
    )

    client = get_client()

    async with AsyncSessionLocal() as db:
        people = await fetch_person_crops(db, args.max_persons)

    print(f"\nPersons with face crops: {len(people)}")

    db_ages: List[int] = []
    ff_person_ages: List[int] = []  # median over person crops
    imdb_person_ages: List[int] = []
    ff_crop_ages: List[int] = []
    imdb_crop_ages: List[int] = []

    person_rows: List[Dict] = []
    skipped_no_crop = 0
    failed_decode = 0

    for i, (pid, db_age, db_group, paths) in enumerate(people, 1):
        paths = paths[: args.max_crops_per_person]
        ff_ages: List[int] = []
        imdb_ages: List[int] = []

        for path in paths:
            frame = load_bgr(client, path)
            if frame is None:
                failed_decode += 1
                continue
            pred_ff = ff.analyze(frame)
            pred_im = imdb.analyze(frame)
            if pred_ff is None or pred_im is None:
                continue
            a_ff = int(pred_ff["age"])
            a_im = int(pred_im["age"])
            ff_ages.append(a_ff)
            imdb_ages.append(a_im)
            ff_crop_ages.append(a_ff)
            imdb_crop_ages.append(a_im)
            if args.per_crop:
                print(
                    f"  {pid[:8]} crop={path[-40:]:40s}  "
                    f"FF={a_ff:3d}/{coarse_group(a_ff):12s}  "
                    f"IMDB={a_im:3d}/{coarse_group(a_im):12s}  "
                    f"DB={db_age}"
                )

        if not ff_ages or not imdb_ages:
            skipped_no_crop += 1
            continue

        ff_med = int(round(float(np.median(ff_ages))))
        im_med = int(round(float(np.median(imdb_ages))))
        ff_person_ages.append(ff_med)
        imdb_person_ages.append(im_med)
        if db_age is not None:
            db_ages.append(int(db_age))

        person_rows.append(
            {
                "pid": pid,
                "db": db_age,
                "db_g": db_group,
                "ff": ff_med,
                "imdb": im_med,
                "n_crops": len(ff_ages),
                "delta": im_med - ff_med,
            }
        )

        if i % 20 == 0:
            print(f"  … processed {i}/{len(people)} persons")

    # ---- summaries ----
    print_hist("DB stored estimated_age (persons with usable crops)", db_ages)
    print_hist("FairFace re-run (person median age)", ff_person_ages)
    print_hist("IMDB re-run (person median age)", imdb_person_ages)
    print_hist("FairFace all crops", ff_crop_ages)
    print_hist("IMDB all crops", imdb_crop_ages)

    print(f"\n=== Person-level FairFace → IMDB shift (n={len(person_rows)}) ===")
    if person_rows:
        deltas = np.array([r["delta"] for r in person_rows], dtype=float)
        print(
            f"  IMDB − FairFace: mean={deltas.mean():+.1f}  "
            f"median={np.median(deltas):+.1f}  "
            f"p10={np.percentile(deltas,10):+.0f}  p90={np.percentile(deltas,90):+.0f}"
        )
        shifted_up = sum(1 for d in deltas if d >= 10)
        shifted_down = sum(1 for d in deltas if d <= -10)
        print(f"  persons IMDB ≥+10y vs FF: {shifted_up}")
        print(f"  persons IMDB ≤−10y vs FF: {shifted_down}")

        # bin movement matrix
        print("\n  coarse-group movement (FairFace → IMDB person-median):")
        move = Counter()
        for r in person_rows:
            move[(coarse_group(r["ff"]), coarse_group(r["imdb"]))] += 1
        groups = ["child", "young_adult", "adult", "senior"]
        header = "from\\to".ljust(14) + "".join(g.rjust(14) for g in groups)
        print("  " + header)
        for src in groups:
            row = src.ljust(14)
            for dst in groups:
                row += str(move.get((src, dst), 0)).rjust(14)
            print("  " + row)

        # under_18 share comparison
        def under18(ages):
            return 100.0 * sum(1 for a in ages if a < 18) / max(len(ages), 1)

        print("\n  % under 18:")
        print(f"    DB       {under18(db_ages):5.1f}%")
        print(f"    FairFace {under18(ff_person_ages):5.1f}%")
        print(f"    IMDB     {under18(imdb_person_ages):5.1f}%")

        print("\n=== Sample persons (sorted by |IMDB−FF| desc, top 25) ===")
        sample = sorted(person_rows, key=lambda r: abs(r["delta"]), reverse=True)[:25]
        print(
            f"  {'pid':12s}  {'DB':>4s}  {'FF':>4s}  {'IMDB':>4s}  {'Δ':>4s}  "
            f"{'DB_bin':10s} {'FF_bin':10s} {'IMDB_bin':10s}  n"
        )
        for r in sample:
            print(
                f"  {r['pid'][:12]:12s}  "
                f"{str(r['db'] or '-'):>4s}  {r['ff']:4d}  {r['imdb']:4d}  "
                f"{r['delta']:+4d}  "
                f"{age_bin(r['db']):10s} {age_bin(r['ff']):10s} {age_bin(r['imdb']):10s}  "
                f"{r['n_crops']}"
            )

    print(
        f"\nDone. persons={len(people)} usable={len(person_rows)} "
        f"skipped_no_decode={skipped_no_crop} failed_object_loads={failed_decode}"
    )
    print("No DB or config changes were made.")


if __name__ == "__main__":
    asyncio.run(main())
