#!/usr/bin/env python3
"""
compare_gender_models.py — SigLIP2 vs InsightFace genderage error rates.

GT:
  - known F→M false positives (CLI / hardcoded list) → Female
  - everyone else → stored PersonIdentity.gender

No DB writes.

Usage:
  PYTHONPATH=/gmr/gmr venv/bin/python danger/compare_gender_models.py
"""
from __future__ import annotations

import asyncio
from collections import Counter, defaultdict

import cv2
import numpy as np
from sqlalchemy import text

from app.core.db.session import AsyncSessionLocal
from app.modules.reid.siglip2_analyzer import get_shared_siglip2
from app.modules.storage.minio_client import BUCKET_PREFIX, get_client

KNOWN_FP_MALE = {
    "3656da5e-cb9a-4215-b898-b44f6db2d59a",
    "3fc4be2c-1996-4e73-838a-ebccf466292f",
    "880d395a-c8af-4f4e-9205-571ce0c0a268",
    "d9eb93f5-92c1-465e-bcf6-9a5fdc44d778",
    "b8565d0b-6021-41ac-87e6-d05da3e0a849",
    "fd027ea1-c52e-44df-83fa-bf86b962ae26",
    "6fc05b89-e359-4176-a986-4369b405eaa0",
    "c08bf35e-2876-4517-bcfc-a3929f30e7ad",
}

METHODS = [
    "sig_face_maj",
    "sig_body_maj",
    "sig_comb_eq",
    "sig_comb_b3",
    "if_best",
    "if_maj",
    "if_median_proxy",
]


def minio_key(path: str) -> str:
    if path.startswith(f"{BUCKET_PREFIX}/"):
        return path[len(BUCKET_PREFIX) + 1 :]
    return path


def load_bgr(client, path):
    try:
        r = client.get_object(BUCKET_PREFIX, minio_key(path))
        d = r.read()
        r.close()
        r.release_conn()
    except Exception:
        return None
    return cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR)


def pad_square(img, pad_frac=0.35):
    h, w = img.shape[:2]
    ph, pw = int(h * pad_frac), int(w * pad_frac)
    return cv2.copyMakeBorder(img, ph, ph, pw, pw, cv2.BORDER_REPLICATE)


def maj(votes):
    if not votes:
        return None
    c = Counter(votes)
    m, f = c.get("M", 0), c.get("F", 0)
    if m == f:
        return None
    return "F" if f > m else "M"


async def main():
    from insightface.app import FaceAnalysis
    from app.utils.device import get_insightface_providers, insightface_ctx_id

    print("Loading models…")
    sig = get_shared_siglip2()
    ifa = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "genderage"],
        providers=get_insightface_providers(),
    )
    ifa.prepare(ctx_id=insightface_ctx_id(), det_size=(640, 640))
    client = get_client()

    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                text(
                    """
                    SELECT pi.id::text, pi.gender,
                      (SELECT array_agg(path ORDER BY s DESC NULLS LAST) FROM (
                         SELECT face_crop_path AS path, face_score AS s
                         FROM person_face_embeddings
                         WHERE person_identity_id=pi.id AND face_crop_path IS NOT NULL
                         ORDER BY face_score DESC NULLS LAST LIMIT 6) x) faces,
                      (SELECT array_agg(path ORDER BY q DESC NULLS LAST) FROM (
                         SELECT crop_path AS path, crop_quality AS q
                         FROM person_embeddings
                         WHERE person_identity_id=pi.id AND crop_path IS NOT NULL
                         ORDER BY crop_quality DESC NULLS LAST LIMIT 4) x) bodies
                    FROM person_identities pi
                    WHERE pi.gender IS NOT NULL
                    """
                )
            )
        ).fetchall()

    def empty():
        return {"correct": 0, "wrong": 0, "abstain": 0, "F_as_M": 0, "M_as_F": 0, "n": 0}

    stats = defaultdict(lambda: {
        "all": empty(), "known_hard": empty(), "rest": empty(), "gt_F": empty(), "gt_M": empty()
    })

    for i, (pid, db_g, faces, bodies) in enumerate(rows, 1):
        db_g = (db_g or "").strip().upper()[:1]
        if db_g not in ("M", "F"):
            continue
        if pid in KNOWN_FP_MALE:
            gt, subset = "F", "known_hard"
        else:
            gt, subset = db_g, "rest"

        faces, bodies = list(faces or []), list(bodies or [])
        sig_f, sig_b = [], []
        for p in faces:
            img = load_bgr(client, p)
            if img is None:
                continue
            r = sig.analyze(img)
            if r:
                sig_f.append(r["gender"])
        for p in bodies:
            img = load_bgr(client, p)
            if img is None:
                continue
            r = sig.analyze(img)
            if r:
                sig_b.append(r["gender"])

        if_gens, if_scores, if_num = [], [], []
        for p in faces:
            img = load_bgr(client, p)
            if img is None:
                continue
            det = ifa.get(img) or ifa.get(pad_square(img))
            if not det:
                continue
            f = max(det, key=lambda x: float(x.det_score))
            g = "M" if int(f.gender) == 1 else "F"
            if_gens.append(g)
            if_scores.append(float(f.det_score))
            if_num.append(1.0 if g == "F" else 0.0)

        preds = {
            "sig_face_maj": maj(sig_f),
            "sig_body_maj": maj(sig_b),
            "sig_comb_eq": maj(sig_f + sig_b),
            "sig_comb_b3": maj(sig_f + sig_b * 3),
        }
        if if_gens:
            preds["if_best"] = if_gens[int(np.argmax(if_scores))]
            preds["if_maj"] = maj(if_gens)
            med = float(np.median(if_num))
            preds["if_median_proxy"] = "F" if med > 0.5 else ("M" if med < 0.5 else None)
        else:
            preds["if_best"] = preds["if_maj"] = preds["if_median_proxy"] = None

        for m, p in preds.items():
            for ss in ("all", subset, f"gt_{gt}"):
                st = stats[m][ss]
                st["n"] += 1
                if p is None:
                    st["abstain"] += 1
                elif p == gt:
                    st["correct"] += 1
                else:
                    st["wrong"] += 1
                    if gt == "F" and p == "M":
                        st["F_as_M"] += 1
                    if gt == "M" and p == "F":
                        st["M_as_F"] += 1
        if i % 20 == 0:
            print(f"  … {i}/{len(rows)}")

    def rate(st):
        decided = st["correct"] + st["wrong"]
        n = st["n"] or 1
        err_dec = 100.0 * st["wrong"] / decided if decided else float("nan")
        err_all = 100.0 * (st["wrong"] + st["abstain"]) / n
        acc_dec = 100.0 * st["correct"] / decided if decided else float("nan")
        return decided, err_dec, err_all, acc_dec

    print("\nGT: known hardcoded F→M IDs = Female; others = DB gender\n")
    for subset in ("all", "rest", "known_hard", "gt_F", "gt_M"):
        print(f"### {subset}")
        print(f"  {'method':18s} {'n':>4s} {'dec':>4s} {'acc':>7s} {'err':>7s} {'err+abs':>8s} {'F→M':>5s} {'M→F':>5s} {'abs':>4s}")
        out = []
        for m in METHODS:
            st = stats[m][subset]
            decided, err_dec, err_all, acc_dec = rate(st)
            out.append((err_dec if decided else 999, m, st, decided, err_dec, err_all, acc_dec))
        out.sort(key=lambda x: x[0])
        for _, m, st, decided, err_dec, err_all, acc_dec in out:
            print(
                f"  {m:18s} {st['n']:4d} {decided:4d} {acc_dec:6.1f}% {err_dec:6.1f}% "
                f"{err_all:7.1f}% {st['F_as_M']:5d} {st['M_as_F']:5d} {st['abstain']:4d}"
            )
        print()

    print("Lowest decided-error winners:")
    for subset in ("all", "rest", "known_hard", "gt_F"):
        best = None
        for m in METHODS:
            st = stats[m][subset]
            decided, err_dec, err_all, acc_dec = rate(st)
            if not decided:
                continue
            if best is None or err_dec < best[0]:
                best = (err_dec, acc_dec, m, st)
        if best:
            err_dec, acc_dec, m, st = best
            print(
                f"  {subset:12s} → {m:18s} err={err_dec:.1f}% acc={acc_dec:.1f}% "
                f"F→M={st['F_as_M']} M→F={st['M_as_F']}"
            )
    print("\nDone. No DB changes.")


if __name__ == "__main__":
    asyncio.run(main())
