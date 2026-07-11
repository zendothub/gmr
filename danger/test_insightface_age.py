#!/usr/bin/env python3
"""
test_insightface_age.py — dry-run InsightFace buffalo_l genderage head on stored face crops.

Majority-vote age bins (+ median years) per person. NO DB / config writes.

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/test_insightface_age.py
    PYTHONPATH=/gmr/gmr venv/bin/python danger/test_insightface_age.py --max-persons 20 --verbose
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from typing import List, Optional, Tuple

import cv2
import numpy as np
from sqlalchemy import text

from app.core.db.session import AsyncSessionLocal
from app.modules.storage.minio_client import BUCKET_PREFIX, get_client

V2_BINS = [
    ("under_18", 0, 17),
    ("age_18_24", 18, 24),
    ("age_25_34", 25, 34),
    ("age_35_44", 35, 44),
    ("age_45_60", 45, 60),
    ("age_60_plus", 61, 999),
]
COARSE = [
    ("child", 0, 11),
    ("young_adult", 12, 24),
    ("adult", 25, 59),
    ("senior", 60, 999),
]


def v2_bin(age: int) -> str:
    for k, lo, hi in V2_BINS:
        if lo <= age <= hi:
            return k
    return "none"


def coarse(age: int) -> str:
    for k, lo, hi in COARSE:
        if lo <= age <= hi:
            return k
    return "none"


def maj(labels: List[str], order: List[str]) -> Optional[str]:
    if not labels:
        return None
    c = Counter(labels)
    top_n = c.most_common(1)[0][1]
    tied = [k for k, n in c.items() if n == top_n]
    if len(tied) == 1:
        return tied[0]
    tied.sort(key=lambda x: order.index(x) if x in order else 99)
    return tied[0]


def minio_key(path: str) -> str:
    if path.startswith(f"{BUCKET_PREFIX}/"):
        return path[len(BUCKET_PREFIX) + 1 :]
    return path


def load_bgr(client, path: str) -> Optional[np.ndarray]:
    try:
        resp = client.get_object(BUCKET_PREFIX, minio_key(path))
        data = resp.read()
        resp.close()
        resp.release_conn()
    except Exception:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def pad_square(img: np.ndarray, pad_frac: float = 0.25) -> np.ndarray:
    """Pad crop so SCRFD still finds a face on tight face crops."""
    h, w = img.shape[:2]
    ph, pw = int(h * pad_frac), int(w * pad_frac)
    return cv2.copyMakeBorder(img, ph, ph, pw, pw, cv2.BORDER_REPLICATE)


class GenderAgeRunner:
    def __init__(self):
        from insightface.app import FaceAnalysis
        from app.utils.device import get_device, get_insightface_providers, insightface_ctx_id

        ctx_id = insightface_ctx_id()
        providers = get_insightface_providers()
        print(f"Loading InsightFace buffalo_l with genderage (ctx={ctx_id}, providers={providers})")
        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "genderage"],
            providers=providers,
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        print(f"  models: {[m for m in self.app.models.keys()]}")
        print(f"  device={get_device()}")

    def predict(self, bgr: np.ndarray) -> Optional[Tuple[int, str, float]]:
        """Return (age, gender M/F, det_score) for best face, or None."""
        if bgr is None or bgr.size == 0:
            return None
        # InsightFace expects BGR already for app.get in some versions; buffalo typically BGR via cv2
        # Official demo uses BGR ndarray directly with cv2.imread
        faces = self.app.get(bgr)
        if not faces:
            # try padded once
            faces = self.app.get(pad_square(bgr, 0.35))
        if not faces:
            return None
        # best det score
        f = max(faces, key=lambda x: float(x.det_score))
        age = int(round(float(f.age)))
        # gender: 0=F 1=M in insightface
        g = int(f.gender)
        gender = "M" if g == 1 else "F"
        return age, gender, float(f.det_score)


async def fetch(db, max_persons: Optional[int]):
    lim = f" LIMIT {int(max_persons)}" if max_persons else ""
    return (
        await db.execute(
            text(
                f"""
                SELECT pi.id::text,
                       pi.estimated_age,
                       pi.age_group,
                       pi.gender,
                       COALESCE(pi.is_staff, false),
                       (SELECT array_agg(path ORDER BY score DESC NULLS LAST)
                        FROM (
                          SELECT face_crop_path AS path, face_score AS score
                          FROM person_face_embeddings
                          WHERE person_identity_id = pi.id
                            AND face_crop_path IS NOT NULL
                          ORDER BY face_score DESC NULLS LAST
                          LIMIT 6
                        ) x) AS faces
                FROM person_identities pi
                WHERE EXISTS (
                    SELECT 1 FROM person_face_embeddings e
                    WHERE e.person_identity_id = pi.id AND e.face_crop_path IS NOT NULL
                )
                ORDER BY pi.created_at NULLS LAST
                {lim}
                """
            )
        )
    ).fetchall()


def print_hist(title: str, ages: List[int]) -> None:
    print(f"\n=== {title} (n={len(ages)}) ===")
    if not ages:
        print("  empty")
        return
    arr = np.array(ages, dtype=float)
    print(
        f"  mean={arr.mean():.1f}  median={np.median(arr):.1f}  "
        f"p10={np.percentile(arr,10):.0f}  p90={np.percentile(arr,90):.0f}  "
        f"min={int(arr.min())}  max={int(arr.max())}"
    )
    bins = Counter(v2_bin(int(a)) for a in ages)
    tot = len(ages)
    print("  V2 bins:")
    for k, _, _ in V2_BINS:
        n = bins.get(k, 0)
        print(f"    {k:12s} {n:4d} ({100.0*n/tot:5.1f}%)")
    cg = Counter(coarse(int(a)) for a in ages)
    print("  coarse:")
    for k, _, _ in COARSE:
        n = cg.get(k, 0)
        print(f"    {k:12s} {n:4d} ({100.0*n/tot:5.1f}%)")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-persons", type=int, default=None)
    ap.add_argument("--max-crops", type=int, default=6)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    runner = GenderAgeRunner()
    client = get_client()

    async with AsyncSessionLocal() as db:
        rows = await fetch(db, args.max_persons)

    print(f"Persons with face crops: {len(rows)}")

    db_ages: List[int] = []
    best_ages: List[int] = []
    median_ages: List[int] = []
    maj_v2: Counter = Counter()
    maj_coarse: Counter = Counter()
    n_ok = 0
    n_no_det = 0
    crop_fail = 0
    crop_ok = 0
    gender_agree = gender_total = 0
    samples_u18 = []
    person_rows = []

    v2_order = [k for k, _, _ in V2_BINS]
    coarse_order = [k for k, _, _ in COARSE]

    for i, (pid, db_age, db_group, db_gender, is_staff, faces) in enumerate(rows, 1):
        faces = list(faces or [])[: args.max_crops]
        ages: List[int] = []
        genders: List[str] = []
        scores: List[float] = []

        for path in faces:
            img = load_bgr(client, path)
            if img is None:
                crop_fail += 1
                continue
            pred = runner.predict(img)
            if pred is None:
                crop_fail += 1
                continue
            age, gender, sc = pred
            ages.append(age)
            genders.append(gender)
            scores.append(sc)
            crop_ok += 1

        if not ages:
            n_no_det += 1
            if args.verbose:
                print(f"  {pid[:12]} NO_FACE_DET crops={len(faces)}")
            continue

        n_ok += 1
        # best = highest det score among predictions
        best_idx = int(np.argmax(scores))
        best_age = ages[best_idx]
        med_age = int(round(float(np.median(ages))))
        mv = maj([v2_bin(a) for a in ages], v2_order)
        mc = maj([coarse(a) for a in ages], coarse_order)
        maj_gender = maj(genders, ["M", "F"])

        best_ages.append(best_age)
        median_ages.append(med_age)
        maj_v2[mv] += 1
        maj_coarse[mc] += 1
        if db_age is not None:
            db_ages.append(int(db_age))

        if db_gender and maj_gender:
            gender_total += 1
            if db_gender.upper()[:1] == maj_gender:
                gender_agree += 1

        person_rows.append(
            {
                "pid": pid,
                "db": db_age,
                "best": best_age,
                "median": med_age,
                "maj_v2": mv,
                "maj_coarse": mc,
                "ages": ages,
                "genders": genders,
            }
        )

        if med_age < 18 and len(samples_u18) < 5:
            samples_u18.append(person_rows[-1])

        if args.verbose:
            print(
                f"  {pid[:12]} DB={db_age} best={best_age} med={med_age} "
                f"maj={mv}/{mc} ages={ages}"
            )

        if i % 15 == 0:
            print(f"  … {i}/{len(rows)}")

    print_hist("DB estimated_age", db_ages)
    print_hist("InsightFace BEST crop age", best_ages)
    print_hist("InsightFace MEDIAN age", median_ages)

    tot = n_ok or 1
    print(f"\n=== InsightFace MAJORITY V2 bin (n={n_ok}) ===")
    for k, _, _ in V2_BINS:
        n = maj_v2.get(k, 0)
        print(f"  {k:12s} {n:4d} ({100.0*n/tot:5.1f}%)")

    print(f"\n=== InsightFace MAJORITY coarse (n={n_ok}) ===")
    for k, _, _ in COARSE:
        n = maj_coarse.get(k, 0)
        print(f"  {k:12s} {n:4d} ({100.0*n/tot:5.1f}%)")

    def u18(ages: List[int]) -> str:
        if not ages:
            return "n/a"
        return f"{100.0 * sum(1 for a in ages if a < 18) / len(ages):.1f}%"

    print("\n% under_18:")
    print(f"  DB              {u18(db_ages)}")
    print(f"  IF best crop    {u18(best_ages)}")
    print(f"  IF median       {u18(median_ages)}")
    print(f"  IF maj under_18 {100.0 * maj_v2.get('under_18', 0) / tot:.1f}%")

    if gender_total:
        print(f"\nGender maj vs DB-stored gender: {gender_agree}/{gender_total} = {100*gender_agree/gender_total:.1f}%")

    # shift vs DB
    if person_rows:
        deltas = [r["median"] - int(r["db"]) for r in person_rows if r["db"] is not None]
        if deltas:
            d = np.array(deltas, dtype=float)
            print(
                f"\nMedian IF − DB age: mean={d.mean():+.1f} median={np.median(d):+.1f} "
                f"p10={np.percentile(d,10):+.0f} p90={np.percentile(d,90):+.0f}"
            )

        print("\nLargest |median−DB| (top 12):")
        top = sorted(
            [r for r in person_rows if r["db"] is not None],
            key=lambda r: abs(r["median"] - int(r["db"])),
            reverse=True,
        )[:12]
        for r in top:
            print(
                f"  {r['pid'][:12]} DB={r['db']:>3} med={r['median']:>3} best={r['best']:>3} "
                f"maj={r['maj_v2']:12s} ages={r['ages']}"
            )

    if samples_u18:
        print("\nSample IF-median under_18:")
        for r in samples_u18:
            print(f"  {r['pid']} DB={r['db']} ages={r['ages']} med={r['median']}")
    else:
        print("\nNo IF-median under_18 persons.")

    print(
        f"\nDone. persons={len(rows)} usable={n_ok} no_det={n_no_det} "
        f"crop_ok={crop_ok} crop_fail={crop_fail}"
    )
    print("No DB or config changes.")


if __name__ == "__main__":
    asyncio.run(main())
